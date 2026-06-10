from guinsoo_mujoco.operators.control import default_joint_target

KP = 20.0
KD = 2.0


def home_target(dof: int) -> list[float]:
    return default_joint_target(dof)
