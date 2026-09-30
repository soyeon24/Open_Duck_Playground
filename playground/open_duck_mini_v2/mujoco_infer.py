import mujoco
import os
import pickle
import sys
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True

# ── 기본 정책 (2026-09-28 부터 hp0dy2 시드 0, 잡 994830) ─────────────────────
# 정책 파일과 **학습 조건 셋(전진·게걸음 범위, 토크 상한)을 한 묶음으로** 둔다.
# 파일만 바꾸고 조건을 안 맞추면 학습 때와 다른 입력을 받아 걸음이 딴판이 된다 —
# fr186 을 3.23 에서 재서 "원을 그린다" 고 적은 게 그 예다 (SIM_NOTES 09-23).
# 정격 1.86 에서 전진 181 cm/12 s, 제자리 회전 380°, 빈 바닥 찾아가기 4/4 (9.6~18.9초).
# 제자리 게걸음은 못 한다 (0.2 cm). SIM_NOTES "994830 / 994831 결과".
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_POLICY = dict(
    onnx=os.path.join(_PROJECT_ROOT, "from_ubai",
                      "hp0dy2_2026_09_28_102045_300482560.onnx"),
    ref_range=True,      # 전진 [-0.148, 0.222]
    lin_vel_y=0.2,       # 게걸음은 원본 범위
    forcerange=1.86,     # 학습은 관절별 U(1.40, 1.90), 실물 정격이 1.86
)


# 물러날 때만 잠깐 쓰는 정책 (MjInfer.backoff). 기본 정책은 후진 명령에 0 cm 다.
# 08-31 은 실물 토크(1.86)에서 12초에 −76 cm 물러난다 — 가진 정책 중 가장 많이 (SIM_NOTES 09-29).
DEFAULT_BACKOFF_POLICY = os.path.join(_PROJECT_ROOT, "from_ubai",
                                      "2026_08_31_144425_300482560.onnx")


def resolve_policy(onnx=None, ref_range=False, lin_vel_y=None, forcerange=None):
    """정책 파일을 안 주면 기본 정책을 **학습 조건째로** 돌려준다. 주면 받은 그대로.

    뷰어·eval_goto·eval_follow·make_video 가 모두 이걸 거친다. 한 곳에서만 고치면
    나머지가 옛 정책으로 조용히 남는 걸 막으려는 것이다.
    """
    if onnx is not None:
        return onnx, ref_range, lin_vel_y, forcerange
    d = DEFAULT_POLICY
    return (d["onnx"], d["ref_range"],
            d["lin_vel_y"] if lin_vel_y is None else lin_vel_y,
            d["forcerange"] if forcerange is None else forcerange)


def _band_tracker():
    """band_tracker.py 를 불러온다.

    이 fork 안이 아니라 상위 프로젝트 루트에 있다 — 실기로 옮길 때 mujoco
    의존성 없이 그 파일 하나만 들고 가려고 일부러 밖에 뒀다.
    """
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import band_tracker
    return band_tracker

# ── 안무 (t 초 -> neck_pitch, head_pitch, head_yaw, head_roll) ────────────────
# 학습된 정책은 머리 명령을 무시한다 (joystick.py 에 머리 추종 보상이 등록돼 있지
# 않아서, head_yaw 를 ±1.5 로 줘도 실제 관절은 -0.23 에서 안 움직인다). 그래서
# 머리 4축은 정책을 우회해 명령으로 직접 구동한다 (DIRECT_HEAD).
# 아래 안무는 정지·전진·게걸음·제자리회전 4조건에서 12초씩 돌려 넘어지지 않는 것만
# 남긴 것이다 (최저 up ≥ 0.96). pitch 를 크게 쓰면 넘어지므로 진폭을 눌러 두었다.
def _bang(t, f=2.0):
    w = 2 * np.pi * f * t
    return (0.35 + 0.35 * np.sin(w), 0.35 * np.sin(w), 0.0, 0.0)

def _doriduri(t, f=1.5):
    w = 2 * np.pi * f * t
    return (0.0, 0.0, 1.3 * np.sin(w), 0.35 * np.sin(w))

def _groove(t, f=1.5):
    w = 2 * np.pi * f * t
    return (0.25 + 0.25 * np.sin(2 * w), 0.25 * np.sin(2 * w),
            1.2 * np.sin(w), 0.4 * np.sin(w))

def _curious(t, f=0.4):
    w = 2 * np.pi * f * t
    return (0.2 + 0.3 * np.sin(w * 0.7), 0.3 * np.sin(w * 1.3),
            1.4 * np.sin(w), 0.2 * np.sin(w * 2))

def _figure8(t, f=1.0):
    w = 2 * np.pi * f * t
    return (0.3 + 0.3 * np.sin(2 * w), 0.0, 1.2 * np.sin(w), 0.45 * np.sin(2 * w))

DANCES = [
    ("헤드뱅잉", _bang),
    ("도리도리", _doriduri),
    ("둠칫", _groove),
    ("두리번거리기", _curious),
    ("머리로 8자", _figure8),
]


class MjInfer(MJInferBase):
    def __init__(
        self,
        model_path: str,
        reference_data: str,
        onnx_model_path: str,
        standing: bool,
        ref_range: bool = False,
        lin_vel_y: float = None,
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        # 정책이 학습된 명령 범위와 뷰어가 보내는 범위는 반드시 같아야 한다.
        # 범위 밖 명령을 주면 정책이 겪어본 적 없는 입력이라 그냥 넘어진다.
        #   --ref_range 없음 : 원본 범위. 2026-09-04 이전 정책 (BEST_WALK_ONNX_2, head)
        #   --ref_range 있음 : 레퍼런스 모션에 정합시킨 범위. fast / rough 정책
        if ref_range:
            self.COMMANDS_RANGE_X = [-0.148, 0.222]
            self.COMMANDS_RANGE_Y = [-0.111, 0.111]
        else:
            self.COMMANDS_RANGE_X = [-0.15, 0.15]
            self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        # 게걸음 범위만 따로 준다. 2026-09-28 잡(hp0dy2)은 전진은 정합값 0.222,
        # 게걸음은 원본 0.2 로 학습해서 위 두 모드 어느 쪽과도 안 맞는다.
        if lin_vel_y is not None:
            self.COMMANDS_RANGE_Y = [-float(lin_vel_y), float(lin_vel_y)]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        # 머리 4축을 정책 대신 명령으로 직접 구동할지 (T 키로 토글).
        #
        # ⚠ 기본값이 True 였는데 **그것만으로 걸음이 왼쪽으로 휜다.** 머리를
        # 명령으로 home 에 붙들면 정책이 보는 머리 관절 상태가 제가 낸 것과
        # 달라지고, 그 어긋남이 요 드리프트로 쌓인다. 원본 정책 40초 전진에서
        # ±1.86 기준 True 면 +305.6°, False 면 −26.7° 다 (backlash 씬).
        # `eval_walk.py` 는 처음부터 False 로 쟀는데 뷰어만 True 로 떠 있어서,
        # 눈으로 본 것과 잰 숫자가 반대 방향으로 갈렸다.
        # 춤(1~5)과 추종(F/N)은 필요할 때 자기가 켠다. T 로 직접 켤 수도 있다.
        self.direct_head = False
        self.dance = None       # None 이면 춤 안 춤. 아니면 DANCES 의 인덱스
        self.dance_t = 0.0

        # ── 방위 유지 (K 키) ──────────────────────────────────────────────
        # 어떤 정책도 쏠림이 0 이 아니다. 원본조차 ±1.86 에서 1°/s 쯤 오른쪽으로
        # 밀린다 (SIM_NOTES "쏠림은 계통 편향인가"). 재학습으로 0 을 만드는 대신,
        # 사람 추종과 **같은 구조**로 정책 밖에서 잡는다 — 방위 오차를 요 명령에
        # 물릴 뿐 정책은 건드리지 않는다. Q/E 를 누르면 그 방위를 새 기준으로 삼는다.
        # 실기에서는 자이로 z 적분이라 기준이 서서히 흐르지만, 직진 구간 몇십 초를
        # 버티는 용도라 충분하다.
        self.heading_hold = False
        self.heading_target = None
        self.HEADING_KP = 0.02        # 도당 요 명령 (FOLLOW_KP 와 같은 눈금)
        self.HEADING_MAX = 0.5        # 이보다 세게는 안 민다
        self.user_ang = 0.0           # 키로 직접 준 요 명령
        self._hold_injected = 0.0     # 직전에 제가 써 넣은 요 명령 (소유권 판정용)
        # 머리 목표값은 따로 들고 있는다. prev_motor_targets 는 정책 목표로 이미
        # 덮인 뒤라, 그걸 기준으로 속도 제한을 걸면 머리가 사실상 안 움직인다.
        self.prev_head = np.array(self.default_actuator[5:9], dtype=float).copy()

        # ── 사람 자동 추종 (F 키) ─────────────────────────────────────────
        # 머리 카메라 영상에서 발목 형광밴드를 찾아 방위각을 뽑고, 그걸 그대로
        # 요 명령에 물린다. 정책은 건드리지 않는다 — 지각은 정책 밖 층이다.
        # 렌더러는 F 를 처음 누를 때 만든다. 추종을 안 쓰는 사람에게 오프스크린
        # GL 컨텍스트 비용을 지울 이유가 없다.
        self.follow = False
        self.follow_rend = None
        self.follow_cam_id = None
        self.follow_lost = 0
        self.follow_last_sign = 1.0   # 놓치면 훑을 쪽 (+1 왼쪽)
        # 훑을 쪽은 사람이 **움직이던 쪽**으로 고른다. 화면 어느 쪽에 있었는지(방위 부호)로
        # 고르면, 상자 뒤로 걸어 들어간 사람이 화면 가운데 조금 오른쪽에서 사라졌다고 오른쪽으로
        # 250° 를 돌았다 — 사람은 왼쪽으로 가고 있었다 (2026-09-30, 소파 씬). 보이는 동안의
        # 월드 방위를 들고 있다가 놓치는 순간 기울기를 본다. 거의 안 움직였으면 방위 부호.
        self.seen_world_deg = []          # 최근 SEEN_HIST_N 제어스텝의 월드 방위
        self.SEEN_HIST_N = 50             # 1초. 0.1 m/s 면 10 cm 라 위치 잡음보다 커야 한다
        self.MOVE_SIGN_DEG_S = 3.0        # 이보다 느리게 돌면 움직임으로 안 친다
        # 도당 요 명령. eval_follow.py 로 재본 값 (사람 2.4,+0.9 / 20초):
        #   0.012 -> 방위 32.1° 로 벌어진다. 못 따라간다
        #   0.020 -> 방위 11.1°  ← 이걸 쓴다
        #   0.035 -> 방위 14.9°. 세게 돌아 오버슈트한다
        self.FOLLOW_KP = 0.020
        self.FOLLOW_STOP_M = 0.55       # 이보다 가까우면 전진을 멈춘다
        self.FOLLOW_ALIGN_DEG = 40.0    # 이만큼 틀어져 있으면 전진 없이 제자리 선회
        # 표적을 놓쳤을 때 제자리에서 훑는 요 명령. 0.35 였는데 **그 값은 아무
        # 일도 안 한다.** scene_person.xml 에서 12초 제자리 회전을 재보면
        # (명령 -> 회전속도)
        #        0.35    0.40    0.50    0.55    0.70    0.80    1.00
        #   08-31  4°/s    5      8      11      19      24      33
        #   head   0°/s    1      4       9      23      29      39
        # 0.5 아래는 사실상 제자리다. 걸어가면서 주는 작은 요 명령(FOLLOW_KP)은
        # 이 사각지대와 별개다 — 전진 중에는 작은 값도 먹는다. 제자리 선회만
        # 문턱이 높다. 0.8 이면 두 정책 다 25~29°/s 로, 한 바퀴에 13초다.
        self.FOLLOW_SEARCH = 0.8

        # 사람이 너무 가까우면 물러난다. 기본 정책(hp0dy2)은 후진 명령에 0 cm 라,
        # 물러나는 동안만 후진이 되는 정책(DEFAULT_BACKOFF_POLICY, 08-31)으로 바꿔 쓴다.
        # 두 정책은 관측·행동 구조가 같다. 서 있을 때·전진 중·후진 도중에 바꿔 끼워도
        # 안 넘어졌다 (2026-09-29, 각 3판, 최저 up 0.991, 4초에 21~23 cm 후진).
        # 이게 없을 때는 사람이 다가와 25 cm 앞에 서면 그대로 서 있다가 밴드가 화각
        # 아래로 빠져 놓친 채 끝났다 (eval_follow_moving "다가오기" 0/3).
        self.backoff = True
        # 보이는 사람이 이보다 가까우면 물러나기 시작 (카메라 기준). 멈추는 건 FOLLOW_STOP_M
        # (0.55) 밖에서 다시 보일 때다. 사이를 둬서 떨지 않게 한다.
        # ⚠ 밴드는 카메라 0.46 m 에서 이미 화각 아래로 빠진다 (09-29 실측, 노트의 0.4 m 보다
        # 멀다). 처음에 0.45 로 뒀더니 보이는 동안 한 번도 안 걸려서 물러나기가 안 켜졌다.
        # 그래서 **놓쳤을 때**는 마지막으로 본 거리가 FOLLOW_STOP_M 안이면 물러난다 —
        # 멈출 거리 안에서 사라졌다면 가까이 와서 화각 아래로 빠진 것이다.
        self.BACKOFF_M = 0.50
        self.BACKOFF_VX = -0.15         # 08-31 이 학습한 후진 상한
        # 가까이서 놓친 뒤(밴드가 화각 아래로) 계속 물러날 시간. 3초로는 모자랐다 — 사람이
        # 계속 다가오는 동안 내내 안 보이고(0.1 m/s 면 7초 가까이), 멈춘 뒤에도 다시 보일
        # 거리(몸통 0.56 m)까지 5~6초를 더 물러나야 한다 (후진 약 5.5 cm/s).
        self.BACKOFF_LOST_S = 12.0
        self.backoff_onnx = DEFAULT_BACKOFF_POLICY
        self.back_policy = None         # 처음 물러날 때 읽는다
        self.backing = False
        self.backing_lost = 0           # 물러나는 중에 연달아 놓친 제어스텝

        # 장애물 회피 (V 키). 바닥 채도로 빈 곳을 찾아 표적 쪽에 가장 가까운
        # 빈 방향으로 간다. 이것도 정책 밖 층이라 재학습이 없다.
        self.avoid = True
        self.follow_blocked = False
        self.follow_free = float("nan")   # 정면 여유거리 (표시용)
        self.AVOID_LOOKAHEAD = 0.9        # 이보다 먼 것은 신경 쓰지 않는다
        # 1.2 로 두면 이 씬에서는 어디도 그만큼 비어 있지 않아 영영 막혔다고
        # 판정한다. 최소 가시거리(0.54 m)보다는 넉넉해야 반응할 시간이 있다.
        self.AVOID_TARGET_MARGIN = 0.30   # 표적보다 이만큼 가까운 것만 장애물로 친다
        self.AVOID_BLIND_MARGIN = 0.25    # 최소 가시거리 바깥에서 결정하기 위한 여유
        # 한 번 정한 우회 방향을 표적 쪽이 열릴 때까지 지킨다 (+1 왼쪽, -1 오른쪽).
        # 매 프레임 새로 고르면, 몸을 틀자마자 장애물이 정면에서 빠지고 다시
        # 표적 쪽으로 꺾여서 결국 장애물 옆구리로 박는다. 실제로 그렇게 됐다.
        self.avoid_side = 0
        # 본 장애물을 월드 좌표로 기억한다 (floor_scan.ObstacleMemory).
        # 카메라는 0.54 m 안쪽 바닥을 못 봐서, 18 cm 벽에 30 cm 까지 붙으면 벽 너머
        # 바닥을 보고 "비었다" 고 한다. 기억 없이는 거기서 우회를 풀고 벽으로 직진해
        # 벽 앞에 붙은 채 끝났다 (2026-09-28, eval_goto --avoid 0/4).
        self.obs_memory = True
        self.obs_mem = None               # 필요할 때 만든다. None 으로 두면 초기화
        # 우회를 방향(steer)이 아니라 월드 좌표 경유점으로 한다 (floor_scan.detour_waypoint).
        # 방향만 주면 정책이 덜 꺾은 만큼 오차가 쌓여 모서리를 못 비켰다.
        # 경유점은 기억(obs_memory)을 쓰므로 기억이 켜져 있어야 동작한다.
        self.waypoint_detour = True
        self.detour_wp = None             # 지금 향하는 경유점 (표시·채점용)
        # 경유점을 모서리 하나로 고르지 않고 A* 경로 위에서 고른다 (floor_scan.plan_detour).
        # 모서리 방식은 벽을 돌자마자 턱·기둥에 막혀 다시 판단하다가, 벽–턱 사이
        # 22 cm 틈(몸통 24 cm)에 경유점을 찍고 20초를 서 있었다 (2026-09-28).
        # 끄면 예전 모서리 방식(detour_waypoint)으로 돌아간다.
        self.path_plan = True
        self.plan_path = None             # 마지막으로 찾은 경로 (표시·채점용)
        # 따라가는 사람의 발은 장애물로 기억하지 않는다. 걸어간 사람의 발자국이 장애물로
        # 남아 빈 바닥에서 우회하다 사람을 놓쳤다 — 걷는 사람 따라가기 회피 켬 2/15,
        # 끔 6/15 (2026-09-29, eval_follow_moving). 기억은 "보이는데 비었을 때" 만 지우는데
        # 뒤따라가는 거리(0.55~0.7 m)에서는 사각(0.54 m)과 사람 발 사이에 지울 틈이 없다.
        # 밴드가 가리키는 바닥 위치 둘레 TARGET_IGNORE_R_M 안은 넣지 않고, 있던 점도 지운다.
        # 끄면 예전처럼 사람 발도 기억한다.
        self.target_not_obstacle = True
        self.TARGET_IGNORE_R_M = 0.30
        # 밴드 무게중심의 화면 높이로 낸 사람(발목) 월드 위치 (band_tracker.ground_xy).
        # 밴드 폭으로 낸 거리(`follow_dist`)는 옆에서 보면 2배 넘게 부풀지만 이건 0.12 m
        # 안이다 (09-29 실측, 참 0.72 m 를 폭은 1.74, 높이는 0.59). 안 보이면 None.
        self.person_xy = None
        # 경유점으로 갈 때의 명령. 기본 정책 실측 (2026-09-28, 정격 1.86):
        #   전진 0.07 은 회전을 얼마를 얹든 거의 제자리 — **0.1 미만은 사각지대**다.
        #   전진 0.15 는 회전을 얹어도 12 cm/s 를 유지하고, 회전 0.5 -> 31°/s,
        #   1.0 -> 58°/s 로 명령에 비례한다 (최소 회전반경 약 12 cm).
        # 그래서 전진은 0.15 로 고정하고 방향은 회전으로만 잡는다.
        self.WP_VX = 0.15
        self.WP_KP = 0.025                # 경유점 방위 1° 당 요 명령
        # 코앞에 장애물이 있으면 전진하지 않고 제자리에서 경유점 쪽으로 먼저 돈다.
        # 이 정책은 **뒤로 못 걷고** (후진 명령 -0.148 에 0 cm/s), 벽에 몸이 닿으면
        # 제자리 회전도 걸려서 25초에 26° 밖에 못 돈다. 그러니 닿기 전에 서야 한다.
        # 몸통 중심에서 앞쪽 이 거리 안, 좌우 반폭 안에 기억된 점이 있으면 선다.
        self.WP_GUARD_M = 0.22
        self.WP_GUARD_HALF_W = 0.14

        # 머리로 표적 붙들기 (G 키). 몸이 우회해도 사람을 계속 본다.
        # 화각 절반이 약 31° 이므로 그보다 작게 묶어야 몸통 정면이 시야에 남는다.
        # 25° 면 머리 ±25° + 화각 ±31° 로 실질 시야가 ±56° 로 넓어지면서도
        # 몸이 가는 쪽(카메라 기준 -head_yaw)이 항상 화면 안에 있다.
        self.head_track = True
        self.HEAD_TRACK_MAX_DEG = 25.0
        # 머리를 사람(1.0)과 갈 방향(0.0) 사이 어디에 두는지. 0.5 가 가운데.
        #
        # eval_follow.py 30 초, 두 배치에서 훑은 값 (놓친 제어스텝 비율 / 최종 거리):
        #
        #   조준   게걸음 | 벽 정면(1.9,0.15)  | 턱 옆(2.4,0.9)
        #   -------------+--------------------+-------------------
        #   1.0    O      | 35%  넘어짐        | 1.57 m  17%
        #   1.0    X      | 52%  넘어짐        | 1.59 m   0%
        #   0.5    O      | 81%  넘어짐        | 1.24 m  51% 넘어짐
        #   0.5    X      | 32%  넘어짐        | 1.60 m   0%   <- 기본값
        #   0.0    X      | 61%  넘어짐        | 2.42 m  94%
        #   머리끔 X      | 77%  넘어짐        | 0.90 m   0%
        #
        # 0.5 + 게걸음 X 가 머리를 켠 조합 중 두 배치 모두에서 가장 낫다.
        # 머리를 아예 끄면 턱 배치는 더 멀리 가지만(0.90 m) 벽 배치에서 표적을
        # 77% 놓친다. 뷰어 기본 배치가 벽 쪽이라 이쪽을 기본으로 둔다. G 키로 뒤집을 것.
        #
        # ⚠ 벽이 표적 정면을 막는 배치는 **어떤 조합도 아직 통과하지 못한다.**
        self.HEAD_AIM_BLEND = 0.5
        # 몸통을 표적 쪽으로 향한 채 두고, 우회는 게걸음으로만 할지.
        # 앞서 게걸음을 켰을 때도 ang 을 go_b(갈 방향)에 물려서 몸이 같이 돌았고,
        # 그게 사람을 화각 밖으로 밀어내는 원인이었다. 몸을 표적에 고정하면
        # 사람이 시야에서 빠질 일이 없다. 대신 우회 속도가 게걸음 한계(0.111)에
        # 묶여 전진(0.222)의 절반이다.
        self.face_target = True

        # 표적을 놓쳤을 때 쓸 월드 기준 방위 기억. 몸통 요를 적분해 들고 있으면
        # 안 보이는 동안에도 사람이 어느 쪽인지 안다. 막 훑는 것보다 훨씬 낫다.
        # 실기에서는 자이로 z 적분이라 드리프트가 쌓이지만, 표적을 다시 잡을
        # 때까지 몇 초만 버티면 되는 용도라 충분하다.
        self.target_world_deg = None
        # 방위만 기억하면 모자랐다. 벽 너머를 가로지르는 사람을 쫓다 놓치자, 오리는 기억한
        # 방위로 돌아서 섰고 사람은 그 사이 더 걸어가 영영 못 찾았다 (2026-09-30,
        # eval_follow_moving --scene wall). 그래서 놓치면 **마지막으로 본 월드 위치**까지
        # 걸어간다 (회피가 켜져 있으면 A* 로). 거기서도 안 보이면 사라진 쪽으로 훑는다.
        # 실기에서 위치는 오도메트리 기준이라 몇 초 동안만 믿을 만하다 — 그래서 시간을 묶는다.
        #
        # 다만 사라진 자리 그 자체가 아니라 **그 사람이 가던 대로 계속 갔을 자리**를 쫓는다
        # (사라진 자리 + 속도 × 놓친 시간, CHASE_PREDICT_M 까지). 소파 씬에서 잰 것 (2026-09-30):
        #   사라진 자리로      숨기 2/3 · 돌아 나오기 1/3 (그 모서리에서는 소파가 반대편을 가린다)
        #   가던 대로 간 자리  숨기 1/3 · 돌아 나오기 3/3  <- 이것. 벽 씬도 6/9 -> 7/9
        #   둘을 차례로        2/6 (두 구간을 걷는 사이 사람이 이미 반대편으로 나가 있다)
        # 남은 약점: 숨자마자 선 사람은 1 m 지나쳐 간다 (숨기 0.2 · 0.3 m/s).
        self.chase_last_seen = True
        self.last_seen_xy = None          # 마지막으로 본 사람(발목) 월드 xy
        self.seen_xy = []                 # 최근 SEEN_HIST_N 제어스텝의 사람 위치 (속도용)
        self.lost_v = np.zeros(2)         # 놓치는 순간의 사람 속도 [m/s]
        self.CHASE_PREDICT_M = 1.0        # 가던 길로 이만큼까지만 내다본다
        self.MOVE_MIN_V = 0.05            # 이보다 느리면 선 것으로 본다 (위치 추정 잡음 3~12 cm)
        self.chase_steps = 0              # 그 위치로 가는 데 쓴 제어스텝
        self.CHASE_ARRIVE_M = 0.45        # 몸통이 이만큼 다가가도 안 보이면 사람이 떠난 것
        self.CHASE_MAX_S = 20.0           # 이보다 오래 가도 못 닿으면 포기하고 훑는다
        # 갈 곳이 이보다 옆이면 걷지 않고 제자리에서 먼저 돈다. 걸으면서 돌면(전진 0.15 + 회전
        # 1.0) 호를 그리는 사이 사람이 더 멀어져서, 화각 가장자리로 빠진 사람을 제자리 선회보다
        # 늦게 다시 잡았다 (빈 바닥 "둘레 반 바퀴" 0.3 m/s 가 7.5초 놓침, 2026-09-30).
        self.CHASE_ALIGN_DEG = 45.0
        # 기억한 방위로 돌 때 이만큼 안에 들어왔는데도 안 보이면 방위 기억을 버리고 훑는다.
        # 전에는 비례 명령이 0.5 아래(제자리 회전 사각지대)로 떨어져 오차 19° 를 남기고
        # 30초를 서 있었다.
        self.TURN_DONE_DEG = 8.0

        # 게걸음으로 비켜 갈지. 켜면 벽 배치에서 81% 로 나빠진다 (위 표).
        # 회피 방향과 머리 조준이 서로 물려 돌아 표적을 놓치기 때문으로 보인다.
        self.use_sidestep = False

        # ── 지정한 지점까지 혼자 가기 (N 키) ──────────────────────────────
        # 출발점과 도착점이 정해져 있을 때, 방향키 없이 도착점까지 간다.
        # 순서는 셋이다:
        #   scan    제자리에서 돌며 사람을 찾는다. 출발 방향을 모른 채 걸어
        #           나가면 엉뚱한 데로 갔다가 되돌아와야 한다. 서서 도는 건
        #           싸고, 도는 동안 넘어질 일도 없다.
        #   go      찾았으면 그 뒤로는 추종과 **같은 코드**(follow_step)다.
        #   arrived FOLLOW_STOP_M 까지 붙으면 명령을 0 으로 두고 선다.
        #
        # 도착점 좌표(self.goal)는 어느 쪽으로 돌지 정하는 데만 쓴다. 좌표를
        # 따라 걷지 않는다 — 걸어가는 동안 좌표를 믿으려면 위치추정이 필요한데,
        # 사람을 눈으로 보고 가면 그게 필요 없다. 좌표는 "왼쪽으로 돌까
        # 오른쪽으로 돌까" 한 번 고르는 데만 있으면 된다.
        self.goto = False
        self.goto_phase = "scan"
        self.goal = None          # (x, y) 월드. None 이면 아무 쪽으로나 훑는다.
        self.scan_sign = 1.0
        self.scan_steps = 0
        self.follow_dist = float("nan")   # 마지막으로 본 표적 거리 (도착 판정용)
        # 한 프레임만 보고 출발하면 반사광 같은 걸 사람으로 오인한 채 나간다.
        self.GOTO_LOCK_N = 3      # 연속 이만큼 잡히면 진짜로 본 것으로 친다
        self.goto_seen = 0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return obs

    def full_reset(self):
        """'home' 키프레임 자세 + 정책 내부 상태까지 완전 초기화.

        MuJoCo 뷰어 기본 Backspace 는 mj_resetData 를 불러 qpos0(다리를 편 자세)로
        되돌리는데, 이 정책은 'home' 키프레임(무릎 굽힌 자세) 기준으로 학습되어
        곧바로 무너진다. 게다가 이전 액션/보행 위상 같은 내부 상태가 남아 있어
        리셋 직후 이상 동작이 나온다. 그래서 별도 리셋을 둔다.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.data.ctrl[:] = self.default_actuator
        self.obs_mem = None               # 이전 판에서 본 장애물은 잊는다
        self.detour_wp = None
        self.plan_path = None
        self.person_xy = None
        self.last_seen_xy = None
        self.seen_world_deg = []
        self.seen_xy = []
        self.chase_steps = 0
        self.backing = False
        self.backing_lost = 0
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = np.array(self.default_actuator).copy()
        self.prev_motor_targets = np.array(self.default_actuator).copy()
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []
        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)
        self.phase_frequency_factor = 1.0
        self.head_control_mode = False
        self.dance = None
        self.dance_t = 0.0
        self.prev_head = np.array(self.default_actuator[5:9], dtype=float).copy()
        self.goto = False
        self.goto_phase = "scan"
        self.goto_seen = 0
        self.follow_dist = float("nan")
        self.heading_target = None
        self.user_ang = 0.0
        self._hold_injected = 0.0
        self.track_reset()
        print(">>> RESET : home keyframe + policy state cleared")

    def heading_step(self):
        """방위 유지 (K). 안 시켰는데 도는 만큼을 요 명령으로 되민다.

        정책은 건드리지 않는다 — 사람 추종이 카메라 방위각을 요 명령에 물리는 것과
        같은 층이다. 끄면(기본) 아무 일도 안 한다.

        **요 명령의 소유권을 추적한다.** `commands[2]` 가 직전에 제가 써 넣은 값
        그대로면 아무도 안 건드린 것이니 계속 밀고, 값이 다르면 바깥(키 입력이든
        eval 스크립트든)이 돌라고 시킨 것이니 손을 뗀다. 안 그러면 제자리 회전
        명령을 제가 지워버린다 (실제로 469.7° 짜리 회전이 -1.6° 가 됐다).

        돌라고 시킨 동안에는 기준을 따라 옮겨서, 손을 떼면 그 자리에서 새 방위를
        지킨다. 제자리(전후·좌우 명령이 0)에서는 안 민다 — 제자리 선회는 문턱이
        높아서(SIM_NOTES) 작은 명령이 아무 일도 못 한다.
        """
        if not self.heading_hold:
            return
        commanded = self.commands[2] != self._hold_injected or self.user_ang != 0
        if commanded:
            # 바깥이 요를 쥐고 있다. 기준만 따라 옮기고 명령은 그대로 둔다.
            self.heading_target = self._yaw_now()
            self._hold_injected = 0.0
            return
        if abs(self.commands[0]) < 1e-6 and abs(self.commands[1]) < 1e-6:
            self.heading_target = self._yaw_now()
            self.commands[2] = 0.0
            self._hold_injected = 0.0
            return
        if self.heading_target is None:
            self.heading_target = self._yaw_now()
        err = np.degrees(
            (self.heading_target - self._yaw_now() + np.pi) % (2 * np.pi) - np.pi
        )
        self.commands[2] = float(
            np.clip(self.HEADING_KP * err, -self.HEADING_MAX, self.HEADING_MAX)
        )
        self._hold_injected = self.commands[2]

    def track_reset(self):
        """리셋 지점 기준의 이동/회전 누적을 다시 0 으로."""
        base = self.get_floating_base_qpos(self.data.qpos)
        self._t_p0 = base[:3].copy()
        self._t_y0 = self._yaw_now()
        self._t_yaw_prev = self._t_y0
        self._t_yaw_acc = 0.0
        self._t_steps = 0

    def _yaw_now(self):
        w, x, y, z = self.get_floating_base_qpos(self.data.qpos)[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def track_step(self, every=50):
        """제어 스텝마다 불러 누적을 갱신하고, 1초에 한 줄 찍는다.

        뷰어로 보면 "왼쪽으로 좀 도는 것 같다" 까지밖에 안 나온다. 눈으로 본 것과
        `eval_walk.py` 가 낸 숫자가 어긋나면 어느 쪽이 틀렸는지 가릴 방법이 없어서,
        보는 사람이 같은 숫자를 읽을 수 있게 뷰어가 직접 찍는다.

        요는 매 스텝 차분을 누적한다. 시작과 끝 자세만 빼면 180도를 넘는 회전이
        반대 부호로 접힌다 (+ 가 왼쪽, - 가 오른쪽).
        """
        yn = self._yaw_now()
        self._t_yaw_acc += (yn - self._t_yaw_prev + np.pi) % (2 * np.pi) - np.pi
        self._t_yaw_prev = yn
        self._t_steps += 1
        if self._t_steps % every:
            return
        d = self.get_floating_base_qpos(self.data.qpos)[:3] - self._t_p0
        fwd = d[0] * np.cos(self._t_y0) + d[1] * np.sin(self._t_y0)
        lat = -d[0] * np.sin(self._t_y0) + d[1] * np.cos(self._t_y0)
        deg = np.rad2deg(self._t_yaw_acc)
        side = "왼쪽" if deg > 0 else "오른쪽"
        print("[{:5.1f}s] 명령 x{:+.2f} y{:+.2f} th{:+.2f} | 전진 {:+6.1f}cm "
              "횡 {:+6.1f}cm | 누적 요 {:+7.1f}° ({})".format(
                  self._t_steps / 50.0, self.commands[0], self.commands[1],
                  self.commands[2], fwd * 100, lat * 100, deg, side))

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 82:  # R : 완전 리셋 (Backspace 대신 이걸 쓸 것)
            self.full_reset()
            return
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode
        if keycode == 84:  # t : 머리 직접 구동 on/off
            self.direct_head = not self.direct_head
            print(f">>> 머리 직접 구동 {'ON' if self.direct_head else 'OFF'} "
                  f"({'ON 이면 머리가 명령을 따르지만 걸음이 휜다' if self.direct_head else 'OFF 가 기본. 2026-09-04 이전 정책은 머리 명령을 무시한다'})")
            return
        if 49 <= keycode <= 53:  # 1~5 : 안무 선택
            self.dance = keycode - 49
            self.dance_t = 0.0
            self.direct_head = True
            print(f">>> 춤: {DANCES[self.dance][0]}  (0 키로 정지)")
            return
        if keycode == 48:  # 0 : 춤 정지
            self.dance = None
            self.commands[3:] = [0.0, 0.0, 0.0, 0.0]
            print(">>> 춤 정지")
            return
        if keycode == 75:  # k : 방위 유지 on/off
            self.heading_hold = not self.heading_hold
            self.heading_target = self._yaw_now() if self.heading_hold else None
            print(">>> 방위 유지 {} {}".format(
                "ON" if self.heading_hold else "OFF",
                "(지금 보는 쪽을 기준으로 잡는다. Q/E 로 기준을 바꾼다)"
                if self.heading_hold else ""))
            return
        if keycode == 71:  # g : 머리로 표적 붙들기 on/off
            self.head_track = not self.head_track
            if not self.head_track:
                self.commands[5] = 0.0
            print(f">>> 머리 표적추종 {'ON' if self.head_track else 'OFF'}")
            return
        if keycode == 86:  # v : 장애물 회피 on/off (추종 중에만 의미가 있다)
            self.avoid = not self.avoid
            print(f">>> 장애물 회피 {'ON' if self.avoid else 'OFF'}"
                  f"{'  (추종을 켜야 동작한다)' if self.avoid and not self.follow else ''}")
            return
        if keycode == 78:  # n : 자율 이동 on/off (돌며 찾고 -> 가고 -> 선다)
            if self.goto:
                self.goto = False
                self.commands[0:3] = [0.0, 0.0, 0.0]
                self.commands[5] = 0.0
                print(">>> 자율 이동 OFF")
            else:
                self.start_goto()
            return
        if keycode == 70:  # f : 사람 자동 추종 on/off
            self.follow = not self.follow
            self.follow_lost = 0
            self.backing = False
            self.avoid_side = 0
            self.obs_mem = None
            self.target_world_deg = None
            self.last_seen_xy = None
            self.seen_world_deg = []
            self.seen_xy = []
            if not self.follow:
                self.commands[0:3] = [0.0, 0.0, 0.0]
                self.commands[5] = 0.0
            print(f">>> 자동 추종 {'ON' if self.follow else 'OFF'}"
                  f"{'  (방향키를 누르면 꺼진다)' if self.follow else ''}")
            return
        # 방향키 등 수동 입력이 들어오면 추종을 끈다. 사람이 몰기 시작했는데
        # 정책이 계속 자기 명령을 덮어쓰면 조종이 안 되는 것처럼 보인다.
        if self.follow or self.goto:
            self.follow = False
            self.goto = False
            print(">>> 수동 입력 : 자동 추종/자율 이동 OFF")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # a
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 80:  # p
                self.phase_frequency_factor += 0.1
            if keycode == 59:  # m
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
            head_roll = 0
            if keycode == 265:  # arrow up
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode == 81:  # a
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode == 69:  # e
                head_roll = self.HEAD_ROLL_RANGE[0]

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

        self.user_ang = ang_vel
        if self.heading_hold and ang_vel != 0:
            # 사람이 직접 돌리는 동안은 기준을 따라 움직인다. 안 그러면 손을
            # 떼는 순간 원래 방위로 되돌아가 버린다.
            self.heading_target = self._yaw_now()
        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def render_head(self):
        """머리 카메라를 한 장 찍어 돌려준다. 씬에 카메라가 없으면 None.

        **한 장만 찍는 게 요점이다.** 색추종과 바닥스캔이 같은 프레임을 봐야
        한다 — 따로 찍으면 두 판단이 서로 다른 순간의 장면을 근거로 삼는다.
        렌더러는 처음 쓸 때 만든다. 추종을 안 쓰는 사람에게 오프스크린 GL
        컨텍스트 비용을 지울 이유가 없다.
        """
        if self.follow_rend is None:
            try:
                self.follow_cam_id = self.model.camera("head_cam").id
            except KeyError:
                print(">>> 이 씬에는 head_cam 이 없다. "
                      "--model_path 를 scene_person.xml 로 줄 것.")
                self.follow = False
                self.goto = False
                return None
            self.follow_rend = mujoco.Renderer(self.model, height=240, width=320)
            print(">>> 추종용 렌더러 생성 (320x240)")
        self.follow_rend.update_scene(self.data, camera="head_cam")
        return self.follow_rend.render()

    def body_yaw_deg(self):
        """월드 기준 몸통 방위(도). 실기에서는 자이로 z 적분으로 얻는다."""
        q = self.get_floating_base_qpos(self.data.qpos)[3:7]
        return float(np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                           1 - 2 * (q[2] ** 2 + q[3] ** 2))))

    def place(self, start=None, start_yaw_deg=None, person=None):
        """출발점·출발 방위·사람(=도착점)을 세팅한다. full_reset 뒤에 부를 것.

        출발 방위를 돌려놓을 수 있어야 한다. 사람을 항상 정면에 두고 시작하면
        그냥 직진해도 도착해서, 스스로 찾아간 건지 알 수가 없다.
        """
        if start is not None or start_yaw_deg is not None:
            qpos = self.data.qpos
            base = self.get_floating_base_qpos(qpos).copy()
            if start is not None:
                base[0], base[1] = float(start[0]), float(start[1])
            if start_yaw_deg is not None:
                h = np.radians(float(start_yaw_deg)) / 2.0
                base[3:7] = [np.cos(h), 0.0, 0.0, np.sin(h)]
            self.set_floating_base_qpos(base, qpos)
        if person is not None:
            self.data.mocap_pos[0] = [float(person[0]), float(person[1]), 0.0]
            self.goal = (float(person[0]), float(person[1]))
        mujoco.mj_forward(self.model, self.data)
        # 몸통을 옮겨 놨으므로 누적의 기준점도 여기로 옮긴다.
        self.track_reset()

    def start_goto(self, goal=None):
        """지금 자리에서 출발해 도착점까지 간다. goal 은 어느 쪽으로 돌지에만 쓴다."""
        if goal is not None:
            self.goal = (float(goal[0]), float(goal[1]))
        self.goto = True
        self.follow = False
        self.goto_phase = "scan"
        self.goto_seen = 0
        self.scan_steps = 0
        self.follow_dist = float("nan")
        self.follow_lost = 0
        self.avoid_side = 0
        self.target_world_deg = None
        self.obs_mem = None
        self.commands[0:3] = [0.0, 0.0, 0.0]

        # 어느 쪽으로 돌지 한 번만 고른다. 도착점을 알면 가까운 쪽으로 돌고,
        # 모르면 왼쪽으로 돈다. 반대로 돌면 사람을 찾는 데 최대 두 배 걸린다.
        self.scan_sign = 1.0
        if self.goal is not None:
            base = self.get_floating_base_qpos(self.data.qpos)
            gb = np.degrees(np.arctan2(self.goal[1] - base[1],
                                       self.goal[0] - base[0])) - self.body_yaw_deg()
            gb = (gb + 180.0) % 360.0 - 180.0
            self.scan_sign = 1.0 if gb >= 0 else -1.0
            print(f">>> 목표 ({self.goal[0]:.2f}, {self.goal[1]:.2f}) / "
                  f"몸통 기준 {gb:+.0f}도, {'왼' if gb >= 0 else '오른'}쪽으로 훑는다")
        print(">>> 자율 이동 ON : 제자리에서 돌며 사람을 찾는 중")

    def goto_step(self):
        """제자리 선회로 사람을 찾고, 찾으면 추종에 넘긴다.

        사람을 찾은 뒤로는 `follow_step` 을 **그대로** 쓴다. 여기서 주행 코드를
        새로 쓰면 뷰어의 F(추종)와 N(자율 이동)이 서로 다르게 굴러서, 한쪽에서
        고친 게 다른 쪽에 안 붙는다. 이 파일에서 이미 한 번 데인 실수다.
        """
        img = self.render_head()
        if img is None:
            return

        if self.goto_phase == "scan":
            res = _band_tracker().track(
                img, float(self.model.cam_fovy[self.follow_cam_id]))
            self.scan_steps += 1
            # 도는 동안 장애물도 훑어 둔다. 출발 전에 주변 지도를 채우는 셈이다.
            # 사람이 화면에 들어왔으면 그 발은 빼고 넣는다.
            if self.avoid and self.obs_memory:
                self.see_obstacles(img, None if res is None else self._person_xy(res, img))
            if res is None:
                self.goto_seen = 0
            else:
                self.goto_seen += 1
            if self.goto_seen < self.GOTO_LOCK_N:
                # 아직. 서서 돈다. 머리는 정면에 둔다 — 머리까지 같이 돌면
                # 몸통 기준 어느 쪽에서 찾았는지가 흐려진다.
                self.commands[0] = 0.0
                self.commands[1] = 0.0
                self.commands[2] = self.FOLLOW_SEARCH * self.scan_sign
                if self.head_track:
                    self.commands[5] = 0.0
                return
            self.goto_phase = "go"
            print(f">>> 사람 발견 (방위 {res['bearing_deg']:+.0f}도, "
                  f"{res['distance_m']:.2f} m, 선회 "
                  f"{self.scan_steps * self.sim_dt * self.decimation:.1f}초) : 출발")

        if self.goto_phase == "arrived":
            self.commands[0:3] = [0.0, 0.0, 0.0]
            return

        # 찍어 둔 프레임을 그대로 넘긴다. 여기서 다시 찍으면 한 제어스텝에 두 번
        # 렌더링하는 셈이라 그냥 느려진다.
        self.follow_step(img=img)

        # ── 언제 섰다고 할 것인가 ────────────────────────────────────────
        # 카메라 거리(`follow_dist`)만 보고 서면 안 된다. 그 값은 **양쪽 밴드를
        # 합친 덩어리의 가로폭**으로 낸 것이라, 비스듬히 보면 두 발목이 겹쳐
        # 폭이 줄고 거리가 부풀려진다. 실측: 참값 1.92 m 일 때 2.18 m,
        # 화면 가장자리에 걸리면 8.48 m 까지 튄다.
        #
        # 가까이서 더 나쁘다. 카메라는 0.375 m 높이에 10° 숙여 있고 밴드는
        # 0.09 m 라, 0.4 m 안쪽이면 밴드가 화각 아래로 빠져 아예 안 보인다.
        # 그래서 추정거리가 0.55 에 닿기 전에 표적을 잃고, 오리는 사람을
        # 지나쳐 계속 걸어갔다 (실제로 목표 (-1.2, 1.5) 에서 0.36 m 까지
        # 파고들고도 도착 판정이 안 났다).
        #
        # 도착점 좌표를 알면 그걸로 선다. 방향은 눈으로 잡고, 정지는 좌표로
        # 한다 — 각자 잘하는 것만 시킨다.
        # (2026-09-29 부터 `follow_dist` 는 밴드 폭이 아니라 밴드 높이로 낸 위치에서 잰다.
        #  옆에서 봐도 0.12 m 안이라 위의 부풀림은 폭 거리로 되돌아갈 때만 해당된다.)
        # 둘 중 **먼저 닿는 쪽**으로 선다. 하나만 보면 둘 다 막힌다:
        #   카메라만  -> 밴드가 화각 아래로 빠지면 영영 0.55 에 안 닿아 지나친다.
        #   좌표만    -> 카메라가 먼저 "다 왔다"며 다리를 세워 버리는데
        #                (follow_step 의 FOLLOW_STOP_M) 좌표로는 0.60 m 라
        #                판정이 안 나고 그 자리에 굳는다. 실제로 그랬다.
        base = self.get_floating_base_qpos(self.data.qpos)
        d, why = self.follow_dist, "카메라"
        if self.goal is not None:
            dg = float(np.hypot(self.goal[0] - base[0], self.goal[1] - base[1]))
            if not (d == d) or dg < d:
                d, why = dg, "좌표"
        if d == d and d < self.FOLLOW_STOP_M:
            self.goto_phase = "arrived"
            self.commands[0:3] = [0.0, 0.0, 0.0]
            print(f">>> 도착 : {why} 기준 {d:.2f} m "
                  f"(오리 {base[0]:.2f}, {base[1]:.2f})")

    def _wp_turn(self, bearing_deg, vx):
        """경유점 방위로 요 명령을 낸다. 제자리(vx=0)면 사각지대를 건너뛴다.

        제자리 회전은 0.5 미만이 사각지대다 (FOLLOW_SEARCH 주석). 전진 중에는 작은
        값도 먹지만, 서서 0.3 을 주면 그대로 굳는다 — 실제로 경유점을 12° 남기고
        30초를 서 있었다 (2026-09-28).
        """
        ang = float(np.clip(self.WP_KP * bearing_deg,
                            self.COMMANDS_RANGE_THETA[0], self.COMMANDS_RANGE_THETA[1]))
        if vx == 0.0 and abs(ang) < 0.8:
            ang = 0.8 if bearing_deg >= 0 else -0.8
        return ang

    def _plan_detour(self, robot_xy, tgt):
        """경유점(detour_wp)을 새로 잡는다. None 이면 표적까지 직선이 비었다."""
        import floor_scan
        if self.path_plan:
            self.detour_wp, self.plan_path, blocked = floor_scan.plan_detour(
                self.obs_mem.pts, robot_xy, tgt)
            if self.detour_wp is not None or not blocked:
                return
            # 막혔는데 격자로는 길이 없다 (번진 기억이 틈을 메웠을 때). 모서리 방식으로.
        self.detour_wp, self.avoid_side = floor_scan.detour_waypoint(
            self.obs_mem.pts, robot_xy, tgt, side=self.avoid_side)

    def _front_blocked(self):
        """기억된 장애물이 몸통 바로 앞(WP_GUARD_M 안, 좌우 반폭 안)에 있는가."""
        if self.obs_mem is None or len(self.obs_mem.pts) == 0:
            return False
        base = self.get_floating_base_qpos(self.data.qpos)
        a = np.radians(self.body_yaw_deg())
        rel = self.obs_mem.pts - base[:2]
        fwd = rel[:, 0] * np.cos(a) + rel[:, 1] * np.sin(a)
        lat = -rel[:, 0] * np.sin(a) + rel[:, 1] * np.cos(a)
        return bool(np.any((fwd > 0.0) & (fwd < self.WP_GUARD_M)
                           & (np.abs(lat) < self.WP_GUARD_HALF_W)))

    def _rear_blocked(self):
        """기억된 장애물이 몸통 바로 뒤에 있는가. 카메라는 앞만 보므로 뒤는 기억뿐이다."""
        if self.obs_mem is None or len(self.obs_mem.pts) == 0:
            return False
        base = self.get_floating_base_qpos(self.data.qpos)
        a = np.radians(self.body_yaw_deg())
        rel = self.obs_mem.pts - base[:2]
        fwd = rel[:, 0] * np.cos(a) + rel[:, 1] * np.sin(a)
        lat = -rel[:, 0] * np.sin(a) + rel[:, 1] * np.cos(a)
        return bool(np.any((fwd < 0.0) & (fwd > -self.WP_GUARD_M)
                           & (np.abs(lat) < self.WP_GUARD_HALF_W)))

    def _back_off(self, target_b):
        """물러난다. target_b 는 몸통 기준 사람 방위(도), 안 보이면 None.

        True 를 돌려주면 이번 스텝 명령을 채운 것이다. 후진 정책을 못 읽으면 물러나기를
        끄고 False — 호출한 쪽이 예전처럼 선다.
        """
        if self.back_policy is None:
            if not os.path.exists(self.backoff_onnx):
                print(f">>> 물러나기 정책이 없다 ({self.backoff_onnx}). 물러나기 끔")
                self.backoff = self.backing = False
                return False
            self.back_policy = OnnxInfer(self.backoff_onnx, awd=True)
            print(f">>> 물러나기 정책 읽음: {os.path.basename(self.backoff_onnx)}")
        self.commands[0] = 0.0 if self._rear_blocked() else self.BACKOFF_VX
        self.commands[1] = 0.0
        # 보이면 사람을 정면에 둔 채 물러난다. 안 보이면 기억한 월드 방위 쪽으로 튼다 —
        # 가까이서 옆으로 빠진 사람도 다시 화각에 들어오게.
        if target_b is None and self.target_world_deg is not None:
            target_b = (self.target_world_deg - self.body_yaw_deg() + 180.0) % 360.0 - 180.0
        self.commands[2] = (0.0 if target_b is None else float(np.clip(
            self.FOLLOW_KP * target_b,
            self.COMMANDS_RANGE_THETA[0], self.COMMANDS_RANGE_THETA[1])))
        return True

    def _chase_step(self, body_yaw, img):
        """놓친 사람을 마지막으로 본 자리로 간다. True 면 이번 스텝 명령을 채운 것이다.

        닿았거나(CHASE_ARRIVE_M) 너무 오래 걸리면(CHASE_MAX_S) 그 자리를 잊고, 방위 기억도
        버려서 사라진 쪽(follow_last_sign)으로 훑게 한 뒤 False.
        """
        base = self.get_floating_base_qpos(self.data.qpos)
        self.chase_steps += 1
        t_lost = self.chase_steps * self.sim_dt * self.decimation
        # 가던 대로 계속 갔을 자리. CHASE_PREDICT_M 에 닿으면 거기서 멈춘다.
        step = self.lost_v * t_lost
        n = float(np.hypot(*step))
        if n > self.CHASE_PREDICT_M:
            step = step * (self.CHASE_PREDICT_M / n)
        tgt = self.last_seen_xy + step
        far = float(np.hypot(*(tgt - base[:2]))) > self.CHASE_ARRIVE_M
        if not far or t_lost > self.CHASE_MAX_S:
            self.last_seen_xy = None
            self.detour_wp = None
            self.target_world_deg = None
            return False
        if self.avoid and self.obs_memory:
            self.see_obstacles(img)       # 가는 동안에도 바닥을 봐서 기억을 채운다
        wp = tgt
        if self.avoid and self.path_plan and self.obs_mem is not None:
            self._plan_detour(base[:2], tgt)
            if self.detour_wp is not None:
                wp = self.detour_wp
        wb = (np.degrees(np.arctan2(wp[1] - base[1], wp[0] - base[0])) - body_yaw + 180.0) % 360.0 - 180.0
        self.commands[0] = (0.0 if abs(wb) > self.CHASE_ALIGN_DEG or self._front_blocked()
                            else self.WP_VX)
        self.commands[1] = 0.0
        self.commands[2] = self._wp_turn(wb, self.commands[0])
        if self.head_track:
            # 보일 때와 같게, 머리는 사람이 있던 곳과 갈 방향 사이를 본다.
            tb = (np.degrees(np.arctan2(tgt[1] - base[1], tgt[0] - base[0])) - body_yaw + 180.0) % 360.0 - 180.0
            self.direct_head = True
            self.commands[5] = float(np.radians(np.clip(
                self.HEAD_AIM_BLEND * tb + (1.0 - self.HEAD_AIM_BLEND) * wb,
                -self.HEAD_TRACK_MAX_DEG, self.HEAD_TRACK_MAX_DEG)))
        return True

    def _person_xy(self, res, img):
        """밴드 검출 결과 -> 사람(발목)의 월드 xy. 실기에서는 카메라 자세를 IMU·관절각으로."""
        band_tracker = _band_tracker()
        cid = self.follow_cam_id
        return band_tracker.ground_xy(
            res, img.shape, float(self.model.cam_fovy[cid]),
            self.data.cam_xpos[cid], self.data.cam_xmat[cid].reshape(3, 3))

    def see_obstacles(self, img, person_xy=None):
        """머리 카메라 한 장으로 바닥 여유거리를 재고, 켜져 있으면 장애물 기억에 넣는다.

        반환: (방위[몸통 프레임, 도], 여유거리[m], 최소 가시거리[m]). 여유거리는
        기억을 합친 값이다. 추종(follow_step)과 사람 찾기 선회(goto_step 의 scan)가
        같이 쓴다 — 선회하는 동안 사방을 한 번 훑으므로 그때 기억을 채워 두면,
        걷다가 몸을 틀어 화면 밖으로 나간 장애물도 이미 알고 있다.
        person_xy 를 주면 그 둘레는 기억에 넣지 않는다 (target_not_obstacle).
        """
        import floor_scan
        R = self.data.cam_xmat[self.follow_cam_id].reshape(3, 3)
        f = -R[:, 2]
        # 실기에서는 이 두 값을 IMU 에서 읽는다. 시뮬이라 카메라 행렬에서 뽑는다.
        pitch_down = -np.degrees(np.arcsin(np.clip(f[2], -1, 1)))
        roll = np.degrees(np.arctan2(R[2, 0], R[2, 1]))
        cam_h = float(self.data.cam_xpos[self.follow_cam_id][2])
        fovy = float(self.model.cam_fovy[self.follow_cam_id])

        bearings, free = floor_scan.free_space(img, fovy, cam_h, pitch_down, roll)
        # 카메라 프레임 -> 몸통 프레임. 실기에서 head_yaw 는 서보 엔코더에서 읽는다.
        head_yaw = np.degrees(self.data.qpos[self.model.joint("head_yaw").qposadr[0]])
        bearings = bearings + head_yaw
        blind = floor_scan.min_visible_range(fovy, cam_h, pitch_down)
        if self.obs_memory:
            if self.obs_mem is None:
                self.obs_mem = floor_scan.ObstacleMemory()
            # bearings 는 몸통 프레임이므로 기준 방위는 몸통 요다.
            # 실기에서 카메라 위치는 오도메트리로 얻는다.
            free = self.obs_mem.update(
                self.data.cam_xpos[self.follow_cam_id][:2], self.body_yaw_deg(),
                bearings, free, blind,
                ignore_xy=person_xy if self.target_not_obstacle else None,
                ignore_r_m=self.TARGET_IGNORE_R_M)
        return bearings, free, blind

    def follow_step(self, img=None):
        """머리 카메라로 사람을 찾아 속도 명령을 만든다. 정책은 그대로 둔다.

        지각은 정책 밖의 층이라 재학습이 필요 없다. 정책은 자기가 받는 3개 숫자가
        사람이 누른 방향키에서 왔는지 카메라에서 왔는지 구분하지 못한다.
        """
        band_tracker = _band_tracker()
        if img is None:
            img = self.render_head()
            if img is None:
                return
        res = band_tracker.track(img, float(self.model.cam_fovy[self.follow_cam_id]))
        self.person_xy = None if res is None else self._person_xy(res, img)

        body_yaw = self.body_yaw_deg()

        if res is None:
            if self.follow_lost == 0 and len(self.seen_world_deg) >= 5:
                # 막 놓쳤다. 보이던 동안 사람이 어느 쪽으로 움직였는지로 훑을 쪽을 정한다.
                w = np.unwrap(np.radians(self.seen_world_deg))
                rate = np.degrees(w[-1] - w[0]) / ((len(w) - 1) * self.sim_dt * self.decimation)
                if abs(rate) > self.MOVE_SIGN_DEG_S:
                    self.follow_last_sign = 1.0 if rate > 0 else -1.0
            if self.follow_lost == 0:
                self.lost_v = np.zeros(2)
                if len(self.seen_xy) >= 10:
                    v = (self.seen_xy[-1] - self.seen_xy[0]) / (
                        (len(self.seen_xy) - 1) * self.sim_dt * self.decimation)
                    if np.hypot(*v) > self.MOVE_MIN_V:
                        self.lost_v = v
            self.seen_world_deg = []
            self.seen_xy = []
            # 놓쳤다. 월드 기준으로 어디 있었는지 기억하고 있으면 그쪽으로 돈다.
            # 기억이 없을 때만 마지막으로 본 쪽으로 훑는다.
            # 놓치기 직전의 부호를 쓰는 게 핵심이다 — 고정 방향으로 훑으면
            # 표적이 오른쪽으로 사라졌는데 왼쪽으로 도는 일이 생긴다.
            self.follow_lost += 1
            if self.backoff and (self.backing or self.follow_dist < self.FOLLOW_STOP_M):
                # 가까이서 놓쳤다 = 밴드가 화각 아래로 빠졌다 (카메라 0.46 m 안). 물러나던
                # 중이었거나 마지막으로 본 거리가 멈출 거리 안이면 정해 둔 시간은 계속 물러난다.
                self.backing_lost += 1
                if self.backing_lost * self.sim_dt * self.decimation <= self.BACKOFF_LOST_S:
                    self.backing = True
                    if self._back_off(None):
                        return
                self.backing = False
            if self.chase_last_seen and not self.goto and self.last_seen_xy is not None:
                if self._chase_step(body_yaw, img):
                    return
            # 우회 중이면 경유점까지는 간다. 경유점은 월드 좌표라 사람이 안 보여도
            # 유효하다. 우회하느라 몸을 틀면 사람이 화각 밖으로 빠지기 쉬운데, 그때마다
            # 서서 사람 쪽으로 돌아버리면 우회가 영영 안 끝난다. 닿으면 놓고 찾는다.
            if (self.avoid and self.path_plan and self.goto and self.goal is not None
                    and self.obs_mem is not None and self.detour_wp is not None):
                # 도착점 좌표를 알면 사람이 안 보여도 경로는 다시 풀 수 있다. 옛 점을
                # 끝까지 쫓으면 그 사이 새로 본 장애물을 무시하게 된다.
                base = self.get_floating_base_qpos(self.data.qpos)
                self._plan_detour(base[:2], np.array(self.goal, dtype=float))
            if self.avoid and self.detour_wp is not None:
                base = self.get_floating_base_qpos(self.data.qpos)
                dx, dy = self.detour_wp - base[:2]
                if np.hypot(dx, dy) > 0.08:
                    wb = (np.degrees(np.arctan2(dy, dx)) - body_yaw + 180.0) % 360.0 - 180.0
                    self.commands[0] = (0.0 if abs(wb) > 90.0 or self._front_blocked()
                                        else self.WP_VX)
                    self.commands[1] = 0.0
                    self.commands[2] = self._wp_turn(wb, self.commands[0])
                    return
                self.detour_wp = None
            self.commands[0] = 0.0
            self.commands[1] = 0.0
            if self.target_world_deg is not None:
                # 기억한 월드 방위를 지금 몸통 기준으로 되돌린다.
                err = (self.target_world_deg - body_yaw + 180.0) % 360.0 - 180.0
                if abs(err) < self.TURN_DONE_DEG:
                    # 다 돌았는데 안 보인다 — 사람은 그 사이 옮겨 갔다. 사라진 쪽으로 훑는다.
                    self.target_world_deg = None
                else:
                    # 제자리 회전이라 사각지대(0.5 미만)를 건너뛴다.
                    self.commands[2] = self._wp_turn(err, 0.0)
                    return
            self.commands[2] = self.FOLLOW_SEARCH * self.follow_last_sign
            return
        self.follow_lost = 0
        self.chase_steps = 0
        if self.person_xy is not None:
            self.last_seen_xy = np.array(self.person_xy, dtype=float)
            self.seen_xy = (self.seen_xy + [self.last_seen_xy])[-self.SEEN_HIST_N:]
        # 거리는 밴드 높이로 낸 위치에서 잰다 (카메라 바닥 투영에서 수평거리). 밴드 폭으로
        # 낸 `distance_m` 은 옆에서 보면 두 발목이 겹쳐 2배 넘게 부풀어, 옆에서 멈춘 사람에게
        # 0.55 m 에서 서지 않고 0.34~0.39 m 까지 파고들었다 — 그러면 밴드가 화각 아래로
        # 빠져 놓친 채 끝난다 (2026-09-29, eval_follow_moving). 위치를 못 구할 때만 폭 거리.
        dist = float(res["distance_m"])
        if self.person_xy is not None:
            dist = float(np.hypot(*(np.asarray(self.person_xy)
                                    - self.data.cam_xpos[self.follow_cam_id][:2])))
        self.follow_dist = dist
        self.follow_last_sign = 1.0 if res["bearing_deg"] >= 0 else -1.0

        # ── 좌표계 ────────────────────────────────────────────────────────
        # 색추종도 바닥스캔도 **카메라 프레임** 값을 낸다. 머리가 돌아가 있으면
        # 몸통이 가야 할 방향과 어긋나므로, 여기서 한 번에 몸통 프레임으로 옮기고
        # 그 뒤로는 계속 몸통 프레임에서만 다룬다. 프레임을 섞는 게 이 코드에서
        # 가장 틀리기 쉬운 곳이다.
        # 실기에서는 head_yaw 를 서보 엔코더에서 읽는다.
        head_yaw = np.degrees(self.data.qpos[self.model.joint("head_yaw").qposadr[0]])
        target_b = res["bearing_deg"] + head_yaw
        self.target_world_deg = body_yaw + target_b   # 안 보일 때 쓸 기억
        self.seen_world_deg = (self.seen_world_deg + [self.target_world_deg])[-self.SEEN_HIST_N:]
        go_b, blocked = target_b, False

        # ── 너무 가까우면 물러난다 ────────────────────────────────────────
        # BACKOFF_M 안이면 시작, FOLLOW_STOP_M 밖에서 보이면 멈춘다 (사이를 둬서 안 떤다).
        # 물러나는 동안은 회피·전진 계산을 건너뛴다 — 뒤는 기억으로만 막힘을 본다.
        self.backing_lost = 0
        if self.backoff:
            if dist < self.BACKOFF_M:
                self.backing = True
            elif self.backing and dist >= self.FOLLOW_STOP_M:
                self.backing = False
            if self.backing:
                if self.head_track:
                    self.direct_head = True
                    self.commands[5] = float(np.radians(np.clip(
                        target_b, -self.HEAD_TRACK_MAX_DEG, self.HEAD_TRACK_MAX_DEG)))
                if self._back_off(target_b):
                    return

        if self.avoid:
            import floor_scan
            bearings, free, blind = self.see_obstacles(img, self.person_xy)

            # 어차피 갈 표적을 장애물로 보고 피하면 영영 못 간다. 표적보다 가까운
            # 것만 장애물로 친다. 동시에 최소 가시거리보다는 넉넉해야 한다 —
            # 그 안쪽은 카메라가 못 보므로 거기 닿기 전에 결정해야 한다.
            clearance = min(dist - self.AVOID_TARGET_MARGIN, self.AVOID_LOOKAHEAD)
            clearance = max(clearance, blind + self.AVOID_BLIND_MARGIN)
            if self.waypoint_detour and self.obs_mem is not None:
                # 표적 위치: 자율 이동이면 도착점 좌표, 추종이면 카메라 추정.
                # 카메라 추정은 밴드 높이로 낸 위치를 먼저 쓴다. 밴드 폭으로 낸 거리는
                # 옆에서 보면 2배 넘게 부풀어, A* 의 도착 영역이 사람 너머에 찍혔다.
                base = self.get_floating_base_qpos(self.data.qpos)
                if self.goto and self.goal is not None:
                    tgt = np.array(self.goal, dtype=float)
                elif self.person_xy is not None:
                    tgt = np.array(self.person_xy, dtype=float)
                else:
                    a = np.radians(self.target_world_deg)
                    tgt = base[:2] + dist * np.array([np.cos(a), np.sin(a)])
                self._plan_detour(base[:2], tgt)
                if self.detour_wp is not None:
                    dx, dy = self.detour_wp - base[:2]
                    go_b = (np.degrees(np.arctan2(dy, dx)) - body_yaw + 180.0) % 360.0 - 180.0
            else:
                self.detour_wp = None
                go_b, blocked, self.avoid_side = floor_scan.steer(
                    bearings, free, target_b, clearance, prefer_side=self.avoid_side)
            self.follow_free = float(free[np.argmin(np.abs(bearings))])  # 몸통 정면 여유

        # ── 머리로 표적을 붙든다 ──────────────────────────────────────────
        # 몸이 우회하려고 돌면 사람이 화각(±31°) 밖으로 나가버린다. 실제로 벽을
        # 피하는 동안 표적을 80% 놓치고, 방향을 잃어 제자리에서 돌다 넘어졌다.
        # 머리를 표적 쪽으로 돌려 두면 몸은 딴 데로 가면서도 계속 볼 수 있다.
        #
        # 다만 머리를 마음껏 돌리면 **몸이 가는 쪽을 아무도 안 보게 된다** —
        # 바닥스캔도 머리가 보는 곳만 훑기 때문이다. 그래서 몸통 정면이 항상
        # 화각 안에 남도록 화각 절반(약 31°)보다 작게 묶는다.
        if self.head_track:
            # 정책은 머리 명령을 무시하므로 직접 구동으로 돌린다.
            self.direct_head = True
            # 머리는 **사람과 갈 방향 사이**를 본다. 카메라 하나로 두 가지를
            # 동시에 해야 하기 때문이다:
            #   사람을 봐야 어디로 갈지 알고 (색추종),
            #   갈 곳을 봐야 거기 뭐가 있는지 안다 (바닥스캔).
            # 사람만 보면 스캔도 사람 쪽만 훑어서 몸이 가는 쪽이 사각이 된다.
            # 실제로 그렇게 두었더니 턱 시나리오가 0.90 m 에서 1.57 m 로 후퇴했다.
            # 가운데를 보면 화각 ±31° 안에 둘 다 들어온다 (둘이 62° 이내일 때).
            #
            # HEAD_AIM_BLEND: 1.0 이면 사람만, 0.0 이면 갈 방향만, 0.5 가 가운데.
            #
            # target_b 와 go_b 는 이미 몸통 프레임이므로 머리 관절각을 그대로
            # 지정한다. 현재 각에 더하는 증분 방식은 오차가 쌓인다.
            want = (self.HEAD_AIM_BLEND * target_b
                    + (1.0 - self.HEAD_AIM_BLEND) * go_b)
            want = float(np.clip(want, -self.HEAD_TRACK_MAX_DEG,
                                 self.HEAD_TRACK_MAX_DEG))
            self.commands[5] = float(np.radians(want))

        # ── 속도를 방향 벡터로 분해한다 ───────────────────────────────────
        # 예전에는 많이 틀어져 있으면 전진을 0 으로 두고 제자리에서 돌게 했다.
        # 그 상태가 이 정책에서 넘어지는 원인이었다 — 크게 돌려면 오래 도는데,
        # 도는 동안 표적이 화각을 벗어나 탐색으로 빠지고 계속 돌다 자빠진다.
        #
        # 오리는 게걸음(lin_vel_y)을 할 줄 안다. 가고 싶은 방향으로 속도 벡터를
        # 쪼개서 주면, 몸이 다 돌기를 기다리지 않고 곧바로 옆으로 비켜 간다.
        # 몸은 몸대로 go_b 쪽으로 돌고, 다 돌고 나면 go_b 가 0 이 되어 자연히
        # 순수 전진이 된다. 제자리 선회라는 상태 자체가 사라진다.
        # face_target 이면 몸통은 표적을 향한 채로 두고 우회는 게걸음이 맡는다.
        turn_to = target_b if self.face_target else go_b
        ang = float(np.clip(self.FOLLOW_KP * turn_to,
                            self.COMMANDS_RANGE_THETA[0], self.COMMANDS_RANGE_THETA[1]))

        # 몸통을 표적에 고정했으면 go_b 는 그대로 몸통 기준 진행방향이다.
        sidestep = self.use_sidestep or self.face_target
        rad = np.radians(np.clip(go_b, -90.0, 90.0)) if sidestep else 0.0
        cx, sy = np.cos(rad), np.sin(rad)
        # 전진과 게걸음의 허용치가 다르므로(0.222 대 0.111), 둘 다 범위에 들어가는
        # 가장 빠른 속도를 고른다. 한쪽만 포화시키면 실제 진행 방향이 틀어진다.
        v = self.COMMANDS_RANGE_X[1]
        if abs(cx) > 1e-6:
            v = min(v, self.COMMANDS_RANGE_X[1] / abs(cx))
        if abs(sy) > 1e-6:
            v = min(v, self.COMMANDS_RANGE_Y[1] / abs(sy))

        if self.avoid and self.detour_wp is not None:
            # 경유점으로 간다. 몸통을 경유점 쪽으로 틀며 걷는다 (전진+회전이 이 정책이
            # 가장 잘하는 동작이다). 게걸음·face_target 은 여기서 안 쓴다.
            # 경유점이 등 뒤면 먼저 제자리에서 돈다.
            vy = 0.0
            vx = 0.0 if abs(go_b) > 90.0 or self._front_blocked() else self.WP_VX
            ang = self._wp_turn(go_b, vx)
        elif blocked or dist < self.FOLLOW_STOP_M:
            vx = vy = 0.0                 # 막혔거나 다 왔으면 선다
        elif not sidestep:
            # 옛 방식: 많이 틀어져 있으면 전진 없이 제자리 선회
            vy = 0.0
            vx = (0.0 if abs(go_b) > self.FOLLOW_ALIGN_DEG else
                  self.COMMANDS_RANGE_X[1] * (1.0 - abs(go_b) / self.FOLLOW_ALIGN_DEG))
        else:
            vx = float(np.clip(v * cx,
                               self.COMMANDS_RANGE_X[0], self.COMMANDS_RANGE_X[1]))
            vy = float(np.clip(v * sy,
                               self.COMMANDS_RANGE_Y[0], self.COMMANDS_RANGE_Y[1]))

        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = ang
        self.follow_blocked = blocked

    def control_step(self):
        """제어 한 스텝 (50Hz). 위상 -> 관측 -> 정책 -> 모터목표 -> ctrl.

        뷰어의 run() 과 채점 스크립트가 **이 하나를 같이 쓴다.** 예전에는 각
        스크립트가 이 루프를 복제했는데, 그러다 DIRECT_HEAD 블록을 빠뜨려
        머리 명령이 모터에 실리지 않았고 머리 추종이 통째로 무효가 됐다.
        복제본을 두면 뷰어에서 보는 것과 측정값이 조용히 갈린다.

        mj_step 과 화면 갱신은 부르는 쪽 몫이다.
        """
        self.heading_step()
        if not self.standing:
            self.imitation_i += 1.0 * self.phase_frequency_factor
            self.imitation_i = (
                self.imitation_i % self.PRM.nb_steps_in_period
            )
            # print(self.PRM.nb_steps_in_period)
            # exit()
            self.imitation_phase = np.array(
                [
                    np.cos(
                        self.imitation_i
                        / self.PRM.nb_steps_in_period
                        * 2
                        * np.pi
                    ),
                    np.sin(
                        self.imitation_i
                        / self.PRM.nb_steps_in_period
                        * 2
                        * np.pi
                    ),
                ]
            )
        if self.dance is not None:
            self.dance_t += self.sim_dt * self.decimation
            self.commands[3:] = list(
                DANCES[self.dance][1](self.dance_t)
            )

        if self.goto:
            self.goto_step()
        elif self.follow:
            self.follow_step()

        obs = self.get_obs(
            self.data,
            self.commands,
        )
        self.saved_obs.append(obs)
        # 물러나는 동안만 후진이 되는 정책으로 추론한다 (backoff). 나머지는 기본 정책.
        policy = (self.back_policy if self.backing and self.back_policy is not None
                  else self.policy)
        action = policy.infer(obs)

        # self.action_filter.push(action)
        # action = self.action_filter.get_filtered_action()

        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        self.motor_targets = (
            self.default_actuator + action * self.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            self.motor_targets = np.clip(
                self.motor_targets,
                self.prev_motor_targets
                - self.max_motor_velocity
                * (self.sim_dt * self.decimation),
                self.prev_motor_targets
                + self.max_motor_velocity
                * (self.sim_dt * self.decimation),
            )

            self.prev_motor_targets = self.motor_targets.copy()

        if self.direct_head:
            # 머리는 정책이 아니라 명령으로 구동한다. 서보 속도
            # 제한은 다른 관절과 똑같이 걸어야 한다 — 목표값을
            # 순간이동시키면 실제 서보보다 가혹해져서, 시뮬에서만
            # 넘어지는 가짜 실패가 나온다.
            lim = self.max_motor_velocity * (self.sim_dt * self.decimation)
            head = np.clip(
                np.asarray(self.commands[3:], dtype=float),
                self.prev_head - lim,
                self.prev_head + lim,
            )
            self.motor_targets[5:9] = head
            self.prev_motor_targets[5:9] = head
            self.prev_head = head.copy()

        self.data.ctrl = self.motor_targets.copy()

    def run(self):
        # 오프스크린 렌더러를 **뷰어보다 먼저** 만든다.
        #
        # 뷰어가 떠 있는 상태에서 mujoco.Renderer 를 만들면 세그폴트(139)로
        # 죽는다. 자율 이동을 켜고 띄운 창이 실제로 그렇게 날아갔다 — 출력이
        # 버퍼에 남은 채라 로그도 안 남는다. 순서만 뒤집으면 F/N 을 도중에
        # 눌러도 새로 만들 게 없다.
        try:
            self.model.camera("head_cam")
        except KeyError:
            pass            # 카메라 없는 씬은 추종 자체가 안 되므로 그냥 둔다
        else:
            self.render_head()

        # full_reset 없이 바로 run() 으로 들어오는 경로가 있어서 여기서 한 번 잡는다.
        self.track_reset()
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        self.control_step()
                        self.track_step()

                        # 화면 갱신은 제어 주기(50Hz)에 맞춘다. 원래는 물리
                        # 스텝마다, 즉 초당 500번 sync 했는데 화면은 60Hz면
                        # 충분하고 sync 는 싸지 않다. CPU 점유의 큰 몫이었다.
                        viewer.sync()

                    # 관측 기록이 무한히 쌓여 장시간 실행 시 메모리를 먹는다.
                    # Ctrl+C 시 덤프하는 용도라 최근 것만 있으면 된다.
                    if len(self.saved_obs) > 20000:
                        del self.saved_obs[:10000]

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, default=None,
                        help="안 주면 기본 정책(DEFAULT_POLICY)을 학습 조건째로 쓴다")
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        # 기본값이 백래시 없는 씬이었다. from_ubai 의 정책은 **전부**
        # scene_flat_terrain_backlash.xml 로 학습됐고(슬럼 로그의 xml: 줄),
        # 백래시 없는 모델로 굴리면 걸음이 달라진다 — 원본 정책 40초 전진에서
        # 쏠림이 -26.7° 대 -102.8° 로 갈린다.
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument(
        "--ref_range",
        action="store_true",
        default=False,
        help="레퍼런스 정합 명령 범위를 쓴다 (fast / rough 정책용)",
    )
    parser.add_argument("--lin_vel_y", type=float, default=None,
                        help="게걸음 범위만 덮어쓴다. hp0dy2 정책은 --ref_range --lin_vel_y 0.2")
    parser.add_argument("--start", type=float, nargs=2, default=None,
                        help="출발점 x y")
    parser.add_argument("--start_yaw", type=str, default=None,
                        help="출발 방위(도). 사람을 등지고 시작시키려면 180. "
                             "'random' 이면 매 실행 아무 방향이나 보고 선다")
    parser.add_argument("--seed", type=int, default=None,
                        help="--start_yaw random 을 재현하고 싶을 때")
    parser.add_argument("--person", type=float, nargs=2, default=None,
                        help="사람(=도착점) x y")
    parser.add_argument("--goto", action="store_true", default=False,
                        help="켜고 시작한다 (뷰어에서 N 키를 누른 것과 같다)")
    parser.add_argument("--no_heading_hold", action="store_true", default=False,
                        help="방위 유지를 끈 채로 시작한다. 뷰어는 켜고 시작하는데, "
                             "정책 자체의 쏠림을 보고 싶으면 꺼야 한다 "
                             "(eval_* 스크립트는 항상 꺼져 있다)")
    parser.add_argument("--direct_head", action="store_true", default=False,
                        help="머리를 정책 대신 명령으로 직접 구동한 채 시작한다 "
                             "(T 키와 같다). 머리는 움직이지만 걸음이 왼쪽으로 휜다")
    parser.add_argument("--forcerange", type=float, default=None,
                        help="토크 상한[N·m]. 씬 XML 은 ±3.23 으로 고정돼 있는데 "
                             "fr186 계열은 ±1.86 으로 학습됐다. 학습값과 다르게 "
                             "굴리면 걸음이 딴판이 된다 (fr186 은 3.23 에서 왼쪽으로 원을 그린다)")

    args = parser.parse_args()
    (args.onnx_model_path, args.ref_range, args.lin_vel_y,
     args.forcerange) = resolve_policy(args.onnx_model_path, args.ref_range,
                                       args.lin_vel_y, args.forcerange)
    print(">>> 정책 {}  (ref_range={}, dy ±{})".format(
        os.path.basename(args.onnx_model_path), args.ref_range, args.lin_vel_y))

    mjinfer = MjInfer(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.standing,
        args.ref_range,
        args.lin_vel_y,
    )
    if args.forcerange is not None:
        mjinfer.model.actuator_forcerange[:] = np.array(
            [-args.forcerange, args.forcerange]
        )
        print(f">>> forcerange ±{args.forcerange} N·m")
    mjinfer.direct_head = args.direct_head
    # 라이브러리 기본값은 꺼짐이고, **뷰어에서만** 켜고 시작한다. eval_* 은
    # MjInfer 를 직접 만들어 쓰므로 정책 맨몸을 그대로 잰다.
    mjinfer.heading_hold = not args.no_heading_hold
    # 조건을 매번 찍는다. 토크·머리모드·씬 셋 다 조용히 어긋나서 "왼쪽으로 돈다"
    # 를 만든 적이 있다. 보이면 안 어긋난다.
    print(">>> 씬 {}".format(os.path.basename(args.model_path)))
    print(">>> 머리 직접 구동 {} (T 로 토글)".format(
        "ON — 걸음이 휜다" if args.direct_head else "OFF"))
    print(">>> 방위 유지 {} (K 로 토글)".format(
        "OFF — 정책 쏠림이 그대로 보인다" if args.no_heading_hold else "ON"))
    yaw = args.start_yaw
    if yaw is not None:
        if yaw.lower() == "random":
            # 아무 방향이나 보고 시작한다. 이게 실제 조건에 가깝다 — 켰을 때
            # 오리가 사람 쪽을 보고 있을 이유가 없다.
            rng = np.random.default_rng(args.seed)
            yaw = float(rng.uniform(-180.0, 180.0))
            print(f">>> 출발 방위 무작위: {yaw:+.0f}도")
        else:
            yaw = float(yaw)
    mjinfer.place(args.start, yaw, args.person)
    if args.goto:
        mjinfer.start_goto()
    mjinfer.run()
