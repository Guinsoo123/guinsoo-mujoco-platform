import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.ik_reach.controller import create_controller as create_ik_reach
from guinsoo_mujoco.demos.ur5e.joint_position.controller import create_controller as create_joint_position
from guinsoo_mujoco.operators.control import (
    JointPositionController,
    ReachMotionController,
)


class FakeRuntime:
    def __init__(self, dof: int = 3):
        self.qpos = np.array([0.0, 0.5, -0.25] if dof == 3 else [0.0] * dof)
        self.qvel = np.zeros(dof)
        self.ctrl = None
        self.model = self
        self.nq = dof

    def read_joint_state(self):
        return self.qpos.copy(), self.qvel.copy()

    def set_control(self, control):
        self.ctrl = np.asarray(control, dtype=float)


def test_joint_position_controller_writes_pd_control_to_runtime():
    runtime = FakeRuntime()
    controller = JointPositionController(
        name="hold",
        target=[0.25, 0.25, -0.25],
        kp=10.0,
        kd=1.0,
    )

    sample = controller.step(runtime, t=0.0, dt=0.01)

    np.testing.assert_allclose(runtime.ctrl, [2.5, -2.5, 0.0])
    assert sample["controller"] == "hold"
    np.testing.assert_allclose(sample["target"], [0.25, 0.25, -0.25])
    np.testing.assert_allclose(sample["control"], runtime.ctrl)


def test_demo_factories_create_expected_controllers():
    runtime = FakeRuntime(dof=6)
    joint = create_joint_position(runtime)
    reach = create_ik_reach(runtime)

    assert isinstance(joint, JointPositionController)
    assert isinstance(reach, ReachMotionController)


def test_reach_motion_controller_modulates_target_over_time():
    runtime = FakeRuntime(dof=6)
    runtime.qpos = np.zeros(6)
    runtime.qvel = np.zeros(6)
    controller = ReachMotionController(
        name="ik_reach",
        home_target=[0.0] * 6,
        amplitude=0.5,
        frequency=1.0,
        joint_index=2,
    )

    sample = controller.step(runtime, t=0.25, dt=0.01)

    assert sample["target"][2] == pytest.approx(0.5)
