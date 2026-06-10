import numpy as np

from guinsoo_mujoco.controllers import JointPositionController


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
