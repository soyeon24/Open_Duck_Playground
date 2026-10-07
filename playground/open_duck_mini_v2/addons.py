"""carry · skate 처럼 로봇에 관절을 더 붙인 씬을 joystick 환경에 올리기 위한 공용 부품.

base.OpenDuckMiniV2Env 는 "구동기도 아니고 바닥 자유관절도 아닌 관절은 전부 백래시 관절"
로 센다. 물건의 자유관절이나 스케이트 바퀴 힌지를 그대로 두면 백래시로 잡혀서 관측
(joint_angles + joint_backlash)의 길이가 어긋나 죽는다. 여기서 그 목록만 다시 만든다.
joystick.py / base.py 는 고치지 않는다 — 서버의 걷기 잡이 같은 파일을 쓴다.
"""

from . import constants


def scene_xml(name: str) -> str:
    return (constants.ROOT_PATH / "xmls" / name).as_posix()


def exclude_extra_joints(env, is_extra) -> list:
    """is_extra(name) 가 참인 관절을 백래시 목록에서 뺀다. 뺀 이름을 돌려준다."""
    extra = [n for n in env.backlash_joint_names if is_extra(n)]
    env.backlash_joint_names = [n for n in env.backlash_joint_names if not is_extra(n)]
    env.backlash_joint_ids = [env.get_joint_id_from_name(n) for n in env.backlash_joint_names]
    env.backlash_joint_qpos_addr = [env.get_joint_addr_from_name(n)
                                    for n in env.backlash_joint_names]
    # 구동기 수 + (백래시 없는 머리 4축) 이 맞는지 본다. 어긋나면 관측 길이가 틀린다.
    n_back = len(env.backlash_joint_names) + len(env.backlash_idx_to_add)
    assert n_back == env.mj_model.nu, (
        f"백래시 {len(env.backlash_joint_names)} + 빈칸 {len(env.backlash_idx_to_add)}"
        f" != 구동기 {env.mj_model.nu}")
    print(f"[addons] 백래시에서 뺀 관절: {extra}")
    return extra
