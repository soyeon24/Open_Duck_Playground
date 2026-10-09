"""감정 스타일 걷기 (emolo) — Kobayashi, "EmoLo: Emotion-Inspired Expressive Locomotion via Single-Policy
Reinforcement Learning on Low-Cost Bipedal Robots" (engrXiv 10.31224/6741) 를 이 저장소의 joystick 위에 옮긴 것.

논문과 같은 것:
  - 정책 하나, 스타일 s ∈ {Neutral, Happy, Sad} 를 one-hot 3칸으로 관측 끝에 붙인다 (101 → 104, 식 3, 표 I).
  - 보상 = 걷기 보상(joystick 그대로) + 스타일 보상 (식 6, 10):
        r_style = g · [ I_N·w_hn·φ(q_hp; μ_N) + I_H·(w_hu·φ(q_hp; μ_H) + w_ha·η(q̇_hp))
                                               + I_S·(w_hd·φ(q_hp; μ_S) − w_ha·η(q̇_hp)) ]
        φ(q; μ) = exp(−(q−μ)²/σ_hp²)   (식 8)     η(q̇) = tanh(|q̇|/β)   (식 9)
    표 II 값: w_hn = w_hu = w_hd = 2.0, w_ha = 0.6, μ_N/μ_H/μ_S = 0 / +0.3 / −0.3 rad, σ_hp = 0.05,
    β = 0.5 rad/s, 게이트 문턱 δ = 0.03 (명령 노름).
  - 스타일은 에피소드마다 뽑고 명령을 다시 뽑을 때 같이 다시 뽑는다 (III-E).
  - q_hp 는 head_pitch 관절 (neck_pitch 아님).
논문이 안 밝힌 것 — 여기서 정한 것:
  - 게이트 모양: g = clip((‖(v_x, v_y, ω_z)‖ − δ) / EMOLO_GATE_W, 0, 1), EMOLO_GATE_W = 0.02.
    "명령 노름이 δ 아래면 g ≈ 0" 만 적혀 있다.
  - 걷기 보상의 가중치는 이 저장소의 joystick (hp0dy2 설정) 을 그대로 쓴다. 논문 표 II 의 w_av 는 2.0 인데
    여기 joystick 은 tracking_ang_vel 6.0 이다 — 논문도 "OSS 베이스라인을 그대로" 라고 했으니 우리 베이스라인을 쓴다.

환경변수: EMOLO_SIGMA (0.05), EMOLO_BETA (0.5), EMOLO_MU (0.3 — Happy +, Sad −), EMOLO_W (2.0), EMOLO_WA (0.6),
EMOLO_DELTA (0.03), EMOLO_GATE_W (0.02), EMOLO_STYLE (−1 = 무작위, 0/1/2 고정 — 채점용).
머리 action 폭은 joystick 의 HEAD_ACTION_SCALE (기본 0.25 = 논문 식 4 의 단일 α). 0.25 면 head_pitch 목표가
±0.25 rad 를 못 넘어서 μ = ±0.3 에 φ 가 최대 e^{-1} 까지만 닿는다.
"""

import os
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import joystick

STYLE_NAMES = ("Neutral", "Happy", "Sad")


def default_config():
    cfg = joystick.default_config()
    e = os.environ.get
    cfg.reward_config.scales.style = 1.0       # r_style 는 안에서 이미 가중치를 곱한다
    cfg.emolo_sigma = float(e("EMOLO_SIGMA", "0.05"))
    cfg.emolo_beta = float(e("EMOLO_BETA", "0.5"))
    cfg.emolo_mu = float(e("EMOLO_MU", "0.3"))
    cfg.emolo_w = float(e("EMOLO_W", "2.0"))
    cfg.emolo_wa = float(e("EMOLO_WA", "0.6"))
    cfg.emolo_delta = float(e("EMOLO_DELTA", "0.03"))
    cfg.emolo_gate_w = float(e("EMOLO_GATE_W", "0.02"))
    cfg.emolo_style = int(e("EMOLO_STYLE", "-1"))
    return cfg


class Emolo(joystick.Joystick):

    def __init__(self, task: str = "flat_terrain_backlash", config=None, config_overrides=None):
        config = default_config() if config is None else config
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        m = self._mj_model
        self._hp_qpos = int(m.joint("head_pitch").qposadr[0])
        self._hp_qvel = int(m.joint("head_pitch").dofadr[0])
        c = config
        print(f"[emolo] mu ±{c.emolo_mu} sigma {c.emolo_sigma} beta {c.emolo_beta} w {c.emolo_w}"
              f" w_a {c.emolo_wa} delta {c.emolo_delta} gate_w {c.emolo_gate_w}"
              f" style {'random' if c.emolo_style < 0 else STYLE_NAMES[c.emolo_style]}")

    # ── 스타일 ────────────────────────────────────────────────────────────
    def _draw_style(self, rng):
        s = jax.random.randint(rng, (), 0, 3)
        return s if self._config.emolo_style < 0 else jp.asarray(self._config.emolo_style)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        info = dict(state.info)
        info["rng"], k = jax.random.split(info["rng"])
        info["style"] = self._draw_style(k)
        contact = jp.array([geoms_colliding(state.data, g, self._floor_geom_id) for g in self._feet_geom_id])
        obs = self._get_obs(state.data, info, contact)
        return state.replace(info=info, obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        info = dict(state.info)
        # joystick.step 은 step > 500 이면 명령을 다시 뽑고 step 을 0 으로 되돌린다 — 그때 스타일도 다시 뽑는다
        info["rng"], k = jax.random.split(info["rng"])
        info["style"] = jp.where(info["step"] == 0, self._draw_style(k), info["style"])
        return state.replace(info=info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array):
        obs = super()._get_obs(data, info, contact)
        onehot = jax.nn.one_hot(info.get("style", 0), 3)
        return {"state": jp.hstack([obs["state"], onehot]),
                "privileged_state": jp.hstack([obs["privileged_state"], onehot])}

    # ── 보상 (식 8~10) ────────────────────────────────────────────────────
    def _style_reward(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        c = self._config
        q = data.qpos[self._hp_qpos]
        qd = data.qvel[self._hp_qvel]
        phi = lambda mu: jp.exp(-jp.square(q - mu) / c.emolo_sigma ** 2)
        eta = jp.tanh(jp.abs(qd) / c.emolo_beta)
        s = info["style"]
        r = jp.where(s == 0, c.emolo_w * phi(0.0),
            jp.where(s == 1, c.emolo_w * phi(c.emolo_mu) + c.emolo_wa * eta,
                             c.emolo_w * phi(-c.emolo_mu) - c.emolo_wa * eta))
        n = jp.linalg.norm(info["command"][:3])
        g = jp.clip((n - c.emolo_delta) / c.emolo_gate_w, 0.0, 1.0)
        return g * r

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        ret = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        ret["style"] = self._style_reward(data, info)
        return ret
