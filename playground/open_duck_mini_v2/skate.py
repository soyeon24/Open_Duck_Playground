"""인라인 스케이트를 신고 걷기 / 타기 (skate).

씬은 make_addon_xml.py 의 scene_skate.xml — 양발 밑에 수동 바퀴 3개씩 (반지름 16 mm, 발 좌우축
힌지). 구동기는 그대로 14개라 정책 입출력 크기가 걷기와 같다.

joystick 과 다른 점은 둘뿐이다:
  1. 발 접지는 "그 발 바퀴 3개 중 하나라도 바닥에 닿았나" 로 잰다. 밑창은 바퀴 때문에 바닥에서
     36 mm 떠 있어서, 원래 접지(밑창 TPU)를 쓰면 늘 0 이다.
     (step() 안의 공중시간 계산은 joystick 을 고치지 않으려고 가운데 바퀴로 대신한다.
      그 값은 critic 관측에만 들어가고 보상에는 안 쓴다.)
  2. imitation 가중치를 SKATE_IMIT_W 로 따로 준다. imitation 은 **걷기** 레퍼런스라서,
     1.0 이면 스케이트 신고 걷는 법을, 0 이면 속도만 맞추는 아무 방법(밀고 미끄러지기 포함)을 배운다.
     어느 쪽이 되는지 모른다 — 첫 학습은 1.0 과 0 을 나란히 돌려 비교한다.

환경변수: SKATE_IMIT_W (1.0). 나머지 걷기 설정은 joystick 과 같은 이름 (LIN_VEL_X …).
"""

import os
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src.collision import geoms_colliding

from . import addons
from . import base as open_duck_mini_v2_base
from . import joystick

N_WHEELS = 3


def default_config():
    cfg = joystick.default_config()
    cfg.reward_config.scales.imitation = float(os.environ.get("SKATE_IMIT_W", "1.0"))
    return cfg


class Skate(joystick.Joystick):

    def __init__(self, config=None, config_overrides=None):
        config = default_config() if config is None else config
        open_duck_mini_v2_base.OpenDuckMiniV2Env.__init__(
            self, xml_path=addons.scene_xml("scene_skate.xml"), config=config,
            config_overrides=config_overrides)
        addons.exclude_extra_joints(self, lambda n: n.startswith("skate_wheel_"))
        self._post_init()

        m = self._mj_model
        self._wheel_geoms = [[m.geom(f"skate_wheel_{side}_{i}").id for i in range(N_WHEELS)]
                             for side in ("left", "right")]
        # joystick.step() 이 공중시간용으로 쓰는 발 geom. 가운데 바퀴로 바꿔 둔다.
        self._feet_geom_id = [w[N_WHEELS // 2] for w in self._wheel_geoms]
        print(f"[skate] imitation {config.reward_config.scales.imitation}"
              f" / 바퀴 geom {self._wheel_geoms}")

    def _skate_contact(self, data: mjx.Data) -> jax.Array:
        return jp.array([
            jp.any(jp.array([geoms_colliding(data, g, self._floor_geom_id) for g in wheels]))
            for wheels in self._wheel_geoms
        ])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array):
        return super()._get_obs(data, info, self._skate_contact(data))

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        return super()._get_reward(data, action, info, metrics, done, first_contact,
                                   self._skate_contact(data))
