from __future__ import annotations

from guinsoo_mujoco.demos.ur5e.ik_reach.config import (
    AMPLITUDE,
    FREQUENCY,
    JOINT_INDEX,
    KD,
    KP,
    home_target,
)
from guinsoo_mujoco.operators.control import ReachMotionController
from guinsoo_mujoco.runtime import MuJoCoRuntime


def create_controller(runtime: MuJoCoRuntime) -> ReachMotionController:
    return ReachMotionController(
        name="ik_reach",
        home_target=home_target(runtime.model.nq),
        amplitude=AMPLITUDE,
        frequency=FREQUENCY,
        joint_index=JOINT_INDEX,
        kp=KP,
        kd=KD,
    )
