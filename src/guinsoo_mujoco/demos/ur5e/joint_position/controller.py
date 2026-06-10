from __future__ import annotations

from guinsoo_mujoco.demos.ur5e.joint_position.config import KD, KP, home_target
from guinsoo_mujoco.operators.control import JointPositionController
from guinsoo_mujoco.runtime import MuJoCoRuntime


def create_controller(runtime: MuJoCoRuntime) -> JointPositionController:
    return JointPositionController(
        name="joint_position",
        target=home_target(runtime.model.nq),
        kp=KP,
        kd=KD,
    )
