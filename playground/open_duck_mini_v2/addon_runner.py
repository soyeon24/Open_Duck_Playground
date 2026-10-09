"""carry(머리에 물건) · skate(스케이트) · obstacle(벽·경사로) 학습 진입점.

runner.py 를 고치지 않으려고 따로 둔다 (standup_runner.py 와 같은 이유 — 서버의 걷기 잡이 그 파일을 쓴다).

    python playground/open_duck_mini_v2/addon_runner.py --env carry --output_dir checkpoints_carry
    python playground/open_duck_mini_v2/addon_runner.py --env skate --output_dir checkpoints_skate

과제 설정은 환경변수로 준다 (carry.py / skate.py 머리말). ONNX 는 걷기와 똑같이 저장할 때마다 나온다.
"""

import argparse
import functools

from playground.common import randomize
from playground.common.runner import BaseRunner


class AddonRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        # 고른 과제만 import 한다. carry / skate 는 따로 개발되는 브랜치라 한쪽만 있을 수 있다.
        if args.env == "carry":
            from playground.open_duck_mini_v2 import carry
            self.env_config = carry.default_config()
            self.env = carry.Carry(self.env_config)
            self.eval_env = carry.Carry(self.env_config)
            self.randomizer = functools.partial(
                carry.carry_randomize, obj_body=self.env._obj_body,
                obj_geom=self.env._obj_geom,
                mass_range=tuple(self.env_config.carry_mass_range))
        elif args.env == "skate":
            from playground.open_duck_mini_v2 import skate
            self.env_config = skate.default_config()
            self.env = skate.Skate(self.env_config)
            self.eval_env = skate.Skate(self.env_config)
            self.randomizer = randomize.domain_randomize
        elif args.env == "obstacle":
            from playground.open_duck_mini_v2 import obstacle
            self.env_config = obstacle.default_config()
            self.env = obstacle.Obstacle(self.env_config)
            self.eval_env = obstacle.Obstacle(self.env_config)
            self.randomizer = randomize.domain_randomize
        else:
            raise ValueError(f"Unknown env {args.env}")
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size} / privileged "
              f"{int(self.env.observation_size['privileged_state'][0])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini V2 carry / skate runner")
    parser.add_argument("--env", type=str, required=True, choices=["carry", "skate", "obstacle"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--task", type=str, default="ignored",
                        help="무시한다. 씬은 --env 가 정한다 (runner.py 와 인자 모양만 맞춘다)")
    parser.add_argument("--restore_checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"checkpoints_{args.env}"
    AddonRunner(args).train()


if __name__ == "__main__":
    main()
