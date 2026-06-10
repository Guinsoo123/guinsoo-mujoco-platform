from guinsoo_mujoco.operators.control import default_joint_target

KP = 12.0
KD = 1.5


def home_target(dof: int) -> list[float]:
    return default_joint_target(dof)
