import mujoco
import pickle
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
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool
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

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

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

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    )
    mjinfer.run()
