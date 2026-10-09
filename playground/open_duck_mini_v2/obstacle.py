"""벽에 부딪혀도, 경사로를 만나도 안 넘어지고 걷기 (obstacle).

씬은 make_obstacle_xml.py 의 scene_obst_train.xml — 걷기 로봇에 몸 충돌 상자 8개(벽하고만 부딪힘),
그리고 에피소드마다 옮기는 mocap 장애물 둘: 기울인 무한 평면(경사로)과 벽 상자.

joystick 과 다른 점:
  1. reset 에서 장애물 하나를 고른다 — 없음 / 벽 / 경사로. **걷기 명령이 가리키는 쪽**
     (몸 방위 + 명령 방향) 으로 d ~ U(OBST_DIST_LO, OBST_DIST_HI) 앞에 둔다. 명령이 0 이면 몸 정면.
       벽    : 벽면이 진행 방향과 β ~ U(-OBST_WALL_ANG, +OBST_WALL_ANG) 만큼 비스듬하다 (0 = 정면충돌)
       경사로 : 바닥과 만나는 선이 진행 방향에 비스듬하고(β 같은 범위 = 사선 경사), 그 너머로
                θ ~ U(OBST_SLOPE_LO, OBST_SLOPE_HI) 도로 오른다. 명령이 뒤집히면 내리막도 된다.
  2. 발 접지는 바닥 **또는 경사로** 를 밟았나로 잰다 (관측·보상). joystick.step 안의 공중시간 계산은
     바닥만 본다 (skate 와 같은 사정, joystick 을 안 고치려고).
  관측은 걷기와 **같다** (101 / 212). 벽·경사를 재는 센서가 없으니 정책은 몸으로 느껴서 버텨야 하고,
  걷기 체크포인트에서 그대로 이어받을 수 있다.

환경변수 (안 주면 괄호 안 값):
  OBST_P_WALL (0.35) / OBST_P_RAMP (0.35)  나머지 확률은 장애물 없음
  OBST_DIST_LO / HI (0.25 / 0.8) m
  OBST_WALL_ANG (70) 도      OBST_SLOPE_LO / HI (3 / 15) 도
나머지 걷기 설정(LIN_VEL_X, LIN_VEL_Y, FR_RAND_LO/HI, HEAD_POS_W …)은 joystick 과 같은 이름.
"""

import math
import os
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import addons
from . import base as open_duck_mini_v2_base
from . import joystick

HIDE_RAMP_Z = -1.0
HIDE_WALL_Z = -5.0
WALL_HALF_T = 0.05     # make_obstacle_xml.WALL_HALF[0]
WALL_HALF_H = 0.3


def default_config():
    cfg = joystick.default_config()
    e = os.environ.get
    cfg.obst_p_wall = float(e("OBST_P_WALL", "0.35"))
    cfg.obst_p_ramp = float(e("OBST_P_RAMP", "0.35"))
    cfg.obst_dist = [float(e("OBST_DIST_LO", "0.25")), float(e("OBST_DIST_HI", "0.8"))]
    cfg.obst_wall_ang = math.radians(float(e("OBST_WALL_ANG", "70")))
    cfg.obst_slope = [math.radians(float(e("OBST_SLOPE_LO", "3"))),
                      math.radians(float(e("OBST_SLOPE_HI", "15")))]
    # OBST_HOLD=1: 벽·경사 판에서는 명령을 "앞으로 OBST_VX_LO ~ lin_vel_x 최대, 옆·회전 0" 으로 두고
    # 에피소드 내내 다시 뽑지 않는다. 1차(10-09)는 명령이 바뀌어 돌아서 가 버리는 판이 많아
    # 경사를 실제로 밟는 판이 적었다 (경사 10° 이상 0/16 그대로, 학습 보상도 안 떨어짐).
    cfg.obst_hold = e("OBST_HOLD", "0") == "1"
    cfg.obst_vx_lo = float(e("OBST_VX_LO", "0.08"))
    return cfg


def _quat_z(a):
    return jp.array([jp.cos(a / 2), 0.0, 0.0, jp.sin(a / 2)])


def _quat_y(a):
    return jp.array([jp.cos(a / 2), 0.0, jp.sin(a / 2), 0.0])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Obstacle(joystick.Joystick):

    def __init__(self, config=None, config_overrides=None):
        config = default_config() if config is None else config
        open_duck_mini_v2_base.OpenDuckMiniV2Env.__init__(
            self, xml_path=addons.scene_xml("scene_obst_train.xml"), config=config,
            config_overrides=config_overrides)
        self._post_init()
        m = self._mj_model
        self._ramp_geom = m.geom("obst_ramp").id
        self._ramp_mocap = int(m.body_mocapid[m.body("obst_ramp").id])
        self._wall_mocap = int(m.body_mocapid[m.body("obst_wall").id])
        c = config
        print(f"[obstacle] p_wall {c.obst_p_wall} / p_ramp {c.obst_p_ramp} / dist {list(c.obst_dist)}"
              f" / wall_ang {math.degrees(c.obst_wall_ang):.0f} deg"
              f" / slope {[round(math.degrees(s), 1) for s in c.obst_slope]} deg")

    # ── 발 접지: 바닥 또는 경사로 ──────────────────────────────────────────
    def _contact(self, data: mjx.Data) -> jax.Array:
        return jp.array([
            geoms_colliding(data, g, self._floor_geom_id) | geoms_colliding(data, g, self._ramp_geom)
            for g in self._feet_geom_id])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array):
        return super()._get_obs(data, info, self._contact(data))

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        return super()._get_reward(data, action, info, metrics, done, first_contact,
                                   self._contact(data))

    # ── 리셋: 걷기 리셋 뒤 장애물을 명령 쪽에 둔다 ─────────────────────────
    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        info = dict(state.info)
        rng, k_kind, k_d, k_b, k_s, k_v = jax.random.split(info["rng"], 6)
        info["rng"] = rng
        c = self._config
        d = state.data

        u = jax.random.uniform(k_kind, ())
        is_wall = u < c.obst_p_wall
        is_ramp = (u >= c.obst_p_wall) & (u < c.obst_p_wall + c.obst_p_ramp)
        hold = (is_wall | is_ramp) & c.obst_hold
        vx = jax.random.uniform(k_v, (), minval=c.obst_vx_lo, maxval=c.lin_vel_x[1])
        cmd = jp.where(hold, info["command"].at[0].set(vx).at[1].set(0.0).at[2].set(0.0),
                       info["command"])
        info["command"] = cmd
        info["obst_hold"] = hold
        info["obst_cmd"] = cmd

        q = d.qpos[3:7]
        yaw = jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        moving = jp.hypot(cmd[0], cmd[1]) > 0.02
        phi = yaw + jp.where(moving, jp.arctan2(cmd[1], cmd[0]), 0.0)   # 진행 방향 (월드)
        dist = jax.random.uniform(k_d, (), minval=c.obst_dist[0], maxval=c.obst_dist[1])
        beta = jax.random.uniform(k_b, (), minval=-c.obst_wall_ang, maxval=c.obst_wall_ang)
        theta = jax.random.uniform(k_s, (), minval=c.obst_slope[0], maxval=c.obst_slope[1])

        base = d.qpos[0:2]
        ahead = base + dist * jp.array([jp.cos(phi), jp.sin(phi)])
        normal = phi + beta          # 벽 두께 방향 = 경사가 오르는 방향 (진행 방향에서 β 비스듬)
        # 벽: 가까운 면이 ahead 에 오게 두께만큼 더 민다.
        wall_xy = ahead + WALL_HALF_T * jp.array([jp.cos(normal), jp.sin(normal)])
        wall_pos = jp.where(is_wall, jp.array([wall_xy[0], wall_xy[1], WALL_HALF_H]),
                            jp.array([0.0, 0.0, HIDE_WALL_Z]))
        wall_quat = _quat_z(normal)
        # 경사로: ahead 를 지나는 선에서 바닥과 만나고 normal 방향으로 θ 만큼 오른다.
        #   R = Rz(normal) Ry(-θ)  ->  평면 법선 (-sinθ cos n, -sinθ sin n, cosθ)
        ramp_pos = jp.where(is_ramp, jp.array([ahead[0], ahead[1], 0.0]),
                            jp.array([0.0, 0.0, HIDE_RAMP_Z]))
        ramp_quat = jp.where(is_ramp, _qmul(_quat_z(normal), _quat_y(-theta)),
                             jp.array([1.0, 0.0, 0.0, 0.0]))

        mp = d.mocap_pos.at[self._wall_mocap].set(wall_pos).at[self._ramp_mocap].set(ramp_pos)
        mq = d.mocap_quat.at[self._wall_mocap].set(wall_quat).at[self._ramp_mocap].set(ramp_quat)
        data = d.replace(mocap_pos=mp, mocap_quat=mq)
        data = mjx.forward(self.mjx_model, data)
        metrics = dict(state.metrics)
        metrics["obst_wall"] = is_wall.astype(jp.float32)
        metrics["obst_ramp"] = is_ramp.astype(jp.float32)
        obs = self._get_obs(data, info, self._contact(data))
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # metrics 의 키는 reset 과 step 이 같아야 한다 (brax 가 트리 구조를 맞춘다).
        ow, orp = state.metrics["obst_wall"], state.metrics["obst_ramp"]
        state = super().step(state, action)
        metrics = dict(state.metrics)
        metrics["obst_wall"], metrics["obst_ramp"] = ow, orp
        # joystick.step 이 명령을 다시 뽑았어도 벽·경사 판이면 처음 명령으로 되돌린다
        # (다시 뽑는 건 관측을 만든 뒤라, 되돌린 값이 다음 스텝 관측에 그대로 들어간다).
        info = dict(state.info)
        info["command"] = jp.where(info["obst_hold"], info["obst_cmd"], info["command"])
        return state.replace(metrics=metrics, info=info)
