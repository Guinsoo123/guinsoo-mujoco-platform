import numpy as np
import pytest

from guinsoo_mujoco.controllers import (
    JointPositionController,
    ReachMotionController,
    create_demo_controller,
)


class FakeRuntime:
    def __init__(self):
        self.qpos = np.array([0.0, 0.5, -0.25])
        self.qvel = np.array([0.0, 0.0, 0.0])
        self.ctrl = None

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


def test_create_demo_controller_supports_joint_position_and_ik_reach():
    joint = create_demo_controller("joint_position", 6)
    reach = create_demo_controller("ik_reach", 6)

    assert isinstance(joint, JointPositionController)
    assert isinstance(reach, ReachMotionController)


def test_reach_motion_controller_modulates_target_over_time():
    runtime = FakeRuntime()
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
