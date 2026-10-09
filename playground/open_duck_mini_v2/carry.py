"""머리에 물건을 올리고 걷기 (carry).

joystick 과 같은 걷기 과제에 세 가지를 더한다:
  1. 머리 위 쟁반에 물건(기본 5 cm 상자)을 올리고 시작한다. 씬은 make_addon_xml.py 가 만든다.
  2. 물건이 떨어지면 에피소드가 끝난다 (넘어진 것과 같은 취급 → alive 를 잃는다).
  3. 물건이 쟁반 가운데 있을수록 보상 (`carry`).

머리 명령은 0 으로 고정하고 head_pos 가중치도 0 이다. 머리 4축은 정책이 **물건을 받치는 데**
마음대로 쓰게 둔다 (닭이 머리를 수평으로 들고 다니는 것처럼). 다만 머리 action_scale 은
걷기와 같은 0.25 rad 이라 쟁반을 ±14° 넘게는 못 기울인다.

정책 관측(state)은 걷기와 **같다** — 실물에 물건을 재는 센서가 없어서다. 물건이 어디 있는지는
critic(privileged_state)만 본다. 그래서 정책은 "물건이 덜 흔들리게 걷는 법" 을 배운다.
CARRY_ACTOR_OBS=1 이면 쟁반 기준 물건 위치 3개를 정책 관측에도 붙인다 (쟁반 밑에 FSR 4개를
깔면 실물에서도 무게중심으로 잴 수 있다).

환경변수 (안 주면 괄호 안 값):
  CARRY_OBJECT     box | ball (box). ball 은 정지 상태에서도 몸 기울기 1° 에 굴러떨어진다
  CARRY_W          가운데 유지 보상 가중치 (5.0). alive 는 joystick 과 같다 (ALIVE_W, 20)
  CARRY_SIGMA      가운데 유지 보상 폭, m (0.03)
  CARRY_PUSH_MAX   밀기 최대 세기 m/s (0.5). 걷기는 1.0 인데 그 세기면 물건은 못 지킨다
  CARRY_MASS_LO/HI 물건 질량 범위 kg (0.02 / 0.15)
  CARRY_ACTOR_OBS  1 이면 물건 위치를 정책 관측에 붙인다 (0)
그 밖의 걷기 설정(LIN_VEL_X, LIN_VEL_Y, FR_RAND_LO/HI, SEED …)은 joystick 과 같은 이름으로 먹는다.
"""

import os
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from playground.common import randomize
from . import addons
from . import base as open_duck_mini_v2_base
from . import joystick

TRAY_HALF = 0.06   # make_addon_xml.TRAY_HALF 와 같아야 한다
OBJ_HALF = 0.025   # make_addon_xml.OBJ_HALF


def default_config():
    cfg = joystick.default_config()
    s = cfg.reward_config.scales
    s.head_pos = 0.0                                   # 머리는 정책 마음대로
    s.carry = float(os.environ.get("CARRY_W", "5.0"))
    cfg.carry_sigma = float(os.environ.get("CARRY_SIGMA", "0.03"))
    cfg.carry_object = os.environ.get("CARRY_OBJECT", "box")
    cfg.carry_actor_obs = os.environ.get("CARRY_ACTOR_OBS", "0") == "1"
    cfg.carry_mass_range = [float(os.environ.get("CARRY_MASS_LO", "0.02")),
                            float(os.environ.get("CARRY_MASS_HI", "0.15"))]
    cfg.push_config.magnitude_range = [0.1, float(os.environ.get("CARRY_PUSH_MAX", "0.5"))]
    # 머리 명령은 0 으로 고정 (관측의 명령 7칸 구조는 그대로 둔다).
    cfg.neck_pitch_range = [0.0, 0.0]
    cfg.head_pitch_range = [0.0, 0.0]
    cfg.head_yaw_range = [0.0, 0.0]
    cfg.head_roll_range = [0.0, 0.0]
    return cfg


class Carry(joystick.Joystick):

    def __init__(self, config=None, config_overrides=None):
        config = default_config() if config is None else config
        xml = addons.scene_xml(f"scene_carry_{config.carry_object}.xml")
        open_duck_mini_v2_base.OpenDuckMiniV2Env.__init__(
            self, xml_path=xml, config=config, config_overrides=config_overrides)
        addons.exclude_extra_joints(self, lambda n: n == "carry_object")
        self._post_init()

        m = self._mj_model
        self._tray_body = m.body("carry_tray").id
        self._obj_body = m.body("carry_object").id
        self._obj_geom = m.geom("carry_object").id
        self._obj_qpos = int(m.joint("carry_object").qposadr[0])
        self._obj_qvel = int(m.joint("carry_object").dofadr[0])
        print(f"[carry] 물건 {config.carry_object} / carry_w {config.reward_config.scales.carry}"
              f" / sigma {config.carry_sigma} / push {list(config.push_config.magnitude_range)}"
              f" / 질량 {list(config.carry_mass_range)} / actor_obs {config.carry_actor_obs}")

    # ── 물건 위치 ─────────────────────────────────────────────────────────
    def _obj_rel(self, data: mjx.Data) -> jax.Array:
        """쟁반 프레임에서 본 물건 중심 (x 앞, y 왼쪽, z 위; 원점은 쟁반 윗면 중앙)."""
        R = data.xmat[self._tray_body]
        return R.T @ (data.xpos[self._obj_body] - data.xpos[self._tray_body])

    def _dropped(self, data: mjx.Data) -> jax.Array:
        rel = self._obj_rel(data)
        return (rel[2] < 0.0) | (jp.linalg.norm(rel[:2]) > TRAY_HALF + 0.02)

    # ── 리셋: 걷기 리셋 뒤 물건을 쟁반 위로 옮긴다 ───────────────────────
    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        d = state.data
        # 리셋이 몸통을 아무 방위로 돌려 놓으므로 키프레임의 물건 자리는 쓸 수 없다.
        p = d.xpos[self._tray_body] + d.xmat[self._tray_body][:, 2] * (OBJ_HALF + 0.001)
        q = d.xquat[self._tray_body]
        a = self._obj_qpos
        qpos = d.qpos.at[a:a + 3].set(p).at[a + 3:a + 7].set(q)
        qvel = d.qvel.at[self._obj_qvel:self._obj_qvel + 6].set(0.0)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=d.ctrl)
        contact = jp.array([geoms_colliding(data, g, self._floor_geom_id)
                            for g in self._feet_geom_id])
        obs = self._get_obs(data, state.info, contact)
        return state.replace(data=data, obs=obs)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        return super()._get_termination(data) | self._dropped(data)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array):
        obs = super()._get_obs(data, info, contact)
        rel = self._obj_rel(data)
        obj_vel = data.qvel[self._obj_qvel:self._obj_qvel + 6]
        state = obs["state"]
        if self._config.carry_actor_obs:
            state = jp.hstack([state, rel])
        return {
            "state": state,
            "privileged_state": jp.hstack([obs["privileged_state"], rel, obj_vel]),
        }

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        ret = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        rel = self._obj_rel(data)
        ret["carry"] = jp.exp(-jp.sum(jp.square(rel[:2])) / self._config.carry_sigma ** 2)
        return ret


def carry_randomize(model: mjx.Model, rng: jax.Array, obj_body: int, obj_geom: int,
                    mass_range):
    """걷기 도메인 랜덤화 + 물건 질량·마찰. 관성은 질량에 비례해 같이 바꾼다."""
    nominal_mass = model.body_mass[obj_body]        # 랜덤화 전 (배치 아님)
    nominal_inertia = model.body_inertia
    model, in_axes = randomize.domain_randomize(model, rng)

    @jax.vmap
    def rand_obj(key):
        k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
        mass = jax.random.uniform(k1, minval=mass_range[0], maxval=mass_range[1])
        fric = jax.random.uniform(k2, minval=0.4, maxval=1.0)
        return mass, fric

    mass, fric = rand_obj(rng)
    body_mass = model.body_mass.at[:, obj_body].set(mass)
    inertia = jp.broadcast_to(nominal_inertia, (mass.shape[0],) + nominal_inertia.shape)
    inertia = inertia.at[:, obj_body].set(
        nominal_inertia[obj_body][None, :] * (mass / nominal_mass)[:, None])
    geom_friction = model.geom_friction.at[:, obj_geom, 0].set(fric)
    model = model.tree_replace({"body_mass": body_mass, "body_inertia": inertia,
                               "geom_friction": geom_friction})
    in_axes = in_axes.tree_replace({"body_inertia": 0})
    return model, in_axes
