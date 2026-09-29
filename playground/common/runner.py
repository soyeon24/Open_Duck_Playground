"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax
import numpy as np

from playground.common.export_onnx import export_onnx


def load_params(path):
    """`policy_params_fn` 이 저장한 체크포인트를 (normalizer, policy, value) 로 되읽는다.

    brax 의 `checkpoint.load` 를 쓰면 `ValueError: Expected list, got RestoreArgs(...)` 로
    죽는다 (2026-09-04, 잡 910947). brax 0.12.4 는 `PyTreeCheckpointer().metadata()` 가
    파라미터 트리를 돌려준다고 보고 그 위에 복원 인자를 매핑하는데, 같이 깔리는
    orbax 0.11+ 는 그 트리를 `StepMetadata` 로 한 겹 감싸서 돌려준다. 복원 인자가 트리가
    아니라 통째로 하나가 돼 버리는 것이다. 여기서는 감싼 걸 벗겨 트리를 꺼낸다.

    numpy 로 복원한다. 저장할 때의 GPU 배치 정보를 따르면 다른 노드·CPU 에서 못 읽는다.
    """
    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(path)
    tree = getattr(meta, "item_metadata", meta)
    tree = getattr(tree, "tree", tree)
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
    raw = list(ckptr.restore(path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None))
    raw[0] = running_statistics.RunningStatisticsState(**raw[0])
    return tuple(raw)


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,  # may not work
            output_path=onnx_export_path
        )

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        # SEED 를 주면 PPO 시드를 바꾼다 (안 주면 기존 설정 그대로 = 재현 가능).
        # 같은 설정을 시드만 바꿔 돌려야 "차이가 시드 노이즈인가" 를 가를 수 있다.
        self.ppo_training_params["seed"] = int(
            os.environ.get("SEED", self.ppo_training_params.get("seed", 0))
        )
        print(f"PPO params: {self.ppo_training_params}")

        # 이어받기는 brax 의 checkpoint.load 대신 load_params 로 읽어 restore_params 로 넘긴다
        # (위 load_params 설명). Adam 상태는 저장하지 않으므로 옵티마이저는 새로 시작한다.
        restore_params = None
        if self.restore_checkpoint_path:
            restore_params = load_params(os.path.abspath(self.restore_checkpoint_path))
            print(f"[restore] {self.restore_checkpoint_path} "
                  f"(관측 정규화 count {float(np.asarray(restore_params[0].count)):.4g})")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_params=restore_params,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
