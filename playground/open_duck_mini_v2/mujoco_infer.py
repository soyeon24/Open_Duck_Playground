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

        # 머리 4축을 정책 대신 명령으로 직접 구동할지 (T 키로 토글)
        self.direct_head = True
        self.dance = None       # None 이면 춤 안 춤. 아니면 DANCES 의 인덱스
        self.dance_t = 0.0
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
        self.follow_last_sign = 1.0   # 놓치기 직전 표적이 있던 쪽 (+1 왼쪽)
        # 도당 요 명령. eval_follow.py 로 재본 값 (사람 2.4,+0.9 / 20초):
        #   0.012 -> 방위 32.1° 로 벌어진다. 못 따라간다
        #   0.020 -> 방위 11.1°  ← 이걸 쓴다
        #   0.035 -> 방위 14.9°. 세게 돌아 오버슈트한다
        self.FOLLOW_KP = 0.020
        self.FOLLOW_STOP_M = 0.55       # 이보다 가까우면 전진을 멈춘다
        self.FOLLOW_ALIGN_DEG = 40.0    # 이만큼 틀어져 있으면 전진 없이 제자리 선회
        self.FOLLOW_SEARCH = 0.35       # 표적을 놓쳤을 때 훑는 요 명령

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
        # 게걸음으로 비켜 갈지. 켜면 벽 배치에서 81% 로 나빠진다 (위 표).
        # 회피 방향과 머리 조준이 서로 물려 돌아 표적을 놓치기 때문으로 보인다.
        self.use_sidestep = False

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
        print(">>> RESET : home keyframe + policy state cleared")

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
                  f"(OFF 면 정책에 맡기는데, 지금 정책은 머리 명령을 무시한다)")
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
        if keycode == 70:  # f : 사람 자동 추종 on/off
            self.follow = not self.follow
            self.follow_lost = 0
            self.avoid_side = 0
            if not self.follow:
                self.commands[0:3] = [0.0, 0.0, 0.0]
                self.commands[5] = 0.0
            print(f">>> 자동 추종 {'ON' if self.follow else 'OFF'}"
                  f"{'  (방향키를 누르면 꺼진다)' if self.follow else ''}")
            return
        # 방향키 등 수동 입력이 들어오면 추종을 끈다. 사람이 몰기 시작했는데
        # 정책이 계속 자기 명령을 덮어쓰면 조종이 안 되는 것처럼 보인다.
        if self.follow:
            self.follow = False
            print(">>> 수동 입력 — 자동 추종 OFF")
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

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def follow_step(self):
        """머리 카메라로 사람을 찾아 속도 명령을 만든다. 정책은 그대로 둔다.

        지각은 정책 밖의 층이라 재학습이 필요 없다. 정책은 자기가 받는 3개 숫자가
        사람이 누른 방향키에서 왔는지 카메라에서 왔는지 구분하지 못한다.
        """
        # band_tracker.py 는 이 fork 가 아니라 상위 프로젝트 루트에 있다 (실기로
        # 옮길 때 mujoco 의존성 없이 그 파일만 들고 가려고 밖에 뒀다).
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        import band_tracker

        if self.follow_rend is None:
            try:
                self.follow_cam_id = self.model.camera("head_cam").id
            except KeyError:
                print(">>> 이 씬에는 head_cam 이 없다. "
                      "--model_path 를 scene_obstacles.xml 로 줄 것.")
                self.follow = False
                return
            self.follow_rend = mujoco.Renderer(self.model, height=240, width=320)
            print(">>> 추종용 렌더러 생성 (320x240)")

        # 영상은 한 번만 찍어 색추종과 바닥스캔이 **같은 프레임**을 본다.
        # 따로 찍으면 두 판단이 다른 순간의 장면을 근거로 삼게 된다.
        self.follow_rend.update_scene(self.data, camera="head_cam")
        img = self.follow_rend.render()
        res = band_tracker.track(img, float(self.model.cam_fovy[self.follow_cam_id]))

        if res is None:
            # 놓쳤다. 전진을 멈추고 마지막으로 본 쪽으로 천천히 훑는다.
            # 놓치기 직전의 부호를 쓰는 게 핵심이다 — 고정 방향으로 훑으면
            # 표적이 오른쪽으로 사라졌는데 왼쪽으로 도는 일이 생긴다.
            self.follow_lost += 1
            self.commands[0] = 0.0
            self.commands[1] = 0.0
            self.commands[2] = self.FOLLOW_SEARCH * self.follow_last_sign
            return
        self.follow_lost = 0
        self.follow_last_sign = 1.0 if res["bearing_deg"] >= 0 else -1.0

        # ── 좌표계 ────────────────────────────────────────────────────────
        # 색추종도 바닥스캔도 **카메라 프레임** 값을 낸다. 머리가 돌아가 있으면
        # 몸통이 가야 할 방향과 어긋나므로, 여기서 한 번에 몸통 프레임으로 옮기고
        # 그 뒤로는 계속 몸통 프레임에서만 다룬다. 프레임을 섞는 게 이 코드에서
        # 가장 틀리기 쉬운 곳이다.
        # 실기에서는 head_yaw 를 서보 엔코더에서 읽는다.
        head_yaw = np.degrees(self.data.qpos[self.model.joint("head_yaw").qposadr[0]])
        target_b = res["bearing_deg"] + head_yaw
        go_b, blocked = target_b, False

        if self.avoid:
            import floor_scan
            R = self.data.cam_xmat[self.follow_cam_id].reshape(3, 3)
            f = -R[:, 2]
            # 실기에서는 이 두 값을 IMU 에서 읽는다. 시뮬이라 카메라 행렬에서 뽑는다.
            pitch_down = -np.degrees(np.arcsin(np.clip(f[2], -1, 1)))
            roll = np.degrees(np.arctan2(R[2, 0], R[2, 1]))
            cam_h = float(self.data.cam_xpos[self.follow_cam_id][2])

            bearings, free = floor_scan.free_space(
                img, float(self.model.cam_fovy[self.follow_cam_id]),
                cam_h, pitch_down, roll)
            bearings = bearings + head_yaw      # 카메라 프레임 -> 몸통 프레임

            # 어차피 갈 표적을 장애물로 보고 피하면 영영 못 간다. 표적보다 가까운
            # 것만 장애물로 친다. 동시에 최소 가시거리보다는 넉넉해야 한다 —
            # 그 안쪽은 카메라가 못 보므로 거기 닿기 전에 결정해야 한다.
            blind = floor_scan.min_visible_range(
                float(self.model.cam_fovy[self.follow_cam_id]), cam_h, pitch_down)
            clearance = min(res["distance_m"] - self.AVOID_TARGET_MARGIN,
                            self.AVOID_LOOKAHEAD)
            clearance = max(clearance, blind + self.AVOID_BLIND_MARGIN)
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
        ang = float(np.clip(self.FOLLOW_KP * go_b,
                            self.COMMANDS_RANGE_THETA[0], self.COMMANDS_RANGE_THETA[1]))

        rad = np.radians(np.clip(go_b, -90.0, 90.0)) if self.use_sidestep else 0.0
        cx, sy = np.cos(rad), np.sin(rad)
        # 전진과 게걸음의 허용치가 다르므로(0.222 대 0.111), 둘 다 범위에 들어가는
        # 가장 빠른 속도를 고른다. 한쪽만 포화시키면 실제 진행 방향이 틀어진다.
        v = self.COMMANDS_RANGE_X[1]
        if abs(cx) > 1e-6:
            v = min(v, self.COMMANDS_RANGE_X[1] / abs(cx))
        if abs(sy) > 1e-6:
            v = min(v, self.COMMANDS_RANGE_Y[1] / abs(sy))

        if blocked or res["distance_m"] < self.FOLLOW_STOP_M:
            vx = vy = 0.0                 # 막혔거나 다 왔으면 선다
        elif not self.use_sidestep:
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

        if self.follow:
            self.follow_step()

        obs = self.get_obs(
            self.data,
            self.commands,
        )
        self.saved_obs.append(obs)
        action = self.policy.infer(obs)

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
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument(
        "--ref_range",
        action="store_true",
        default=False,
        help="레퍼런스 정합 명령 범위를 쓴다 (fast / rough 정책용)",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.standing,
        args.ref_range,
    )
    mjinfer.run()
