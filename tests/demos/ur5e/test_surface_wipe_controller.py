import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import PRE_CONTACT_OFFSET
from guinsoo_mujoco.demos.ur5e.surface_wipe.controller import SurfaceWipeController
from guinsoo_mujoco.demos.ur5e.surface_wipe.workflow import Phase


class FakeRuntime:
    def __init__(self, dof: int = 6):
        self.qpos = np.zeros(dof)
        self.qvel = np.zeros(dof)
        self.ctrl = np.zeros(dof)
        self.data = self
        self.model = type("Model", (), {"nu": dof})()

    def read_joint_state(self):
        return self.qpos.copy(), self.qvel.copy()

    def set_control(self, control):
        self.ctrl = np.asarray(control, dtype=float)

    def joint_limits(self):
        return np.full(self.qpos.size, -2.0 * np.pi), np.full(
            self.qpos.size, 2.0 * np.pi
        )

    def read_site_wrench(self, force_sensor, torque_sensor):
        del force_sensor, torque_sensor
        return np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=float)

    def site_pose(self, site_name):
        del site_name
        return np.zeros(3), np.eye(3)


def test_follow_phase_advances_path_parameter(monkeypatch):
    runtime = FakeRuntime()
    controller = SurfaceWipeController(runtime)
    controller.phase = Phase.FOLLOW
    controller.path_s = 0.0
    monkeypatch.setattr(
        controller,
        "_solve_pose",
        lambda _runtime, _pose, _options, **kwargs: np.ones(6),
    )

    sample = controller.step(runtime, t=0.0, dt=0.02)

    assert controller.path_s > 0.0
    assert sample["phase"] == Phase.FOLLOW.value


def test_descend_transitions_to_follow(monkeypatch):
    runtime = FakeRuntime()
    controller = SurfaceWipeController(runtime)
    controller.phase = Phase.DESCEND
    controller.descend_offset = PRE_CONTACT_OFFSET
    controller.descend_elapsed = 1.0
    monkeypatch.setattr(
        controller,
        "_read_normal_force",
        lambda _runtime, *, s: 5.0,
    )
    monkeypatch.setattr(
        controller,
        "_tool_contacts_wave",
        lambda _runtime: True,
    )
    monkeypatch.setattr(
        controller,
        "_solve_pose",
        lambda _runtime, _pose, _options, **kwargs: np.ones(6),
    )

    controller.step(runtime, t=0.0, dt=0.01)

    assert controller.phase == Phase.FOLLOW


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_scene_loads():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession
    from guinsoo_mujoco.demos.ur5e.surface_wipe.workflow import Phase

    scene = (
        Path.home()
        / ".guinsoo_mujoco/assets/mujoco_menagerie/universal_robots_ur5e/guinsoo_surface_wipe_scene.xml"
    )
    if not scene.exists():
        pytest.skip("UR5e surface_wipe scene assets not cached")
    session = SimSession.load("ur5e", "surface_wipe")
    sample = session.step(0.002)
    assert sample["target"] is not None
    assert session.controller.name == "surface_wipe"


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_reaches_done():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession
    from guinsoo_mujoco.demos.ur5e.surface_wipe.config import WIPE_LENGTH
    from guinsoo_mujoco.demos.ur5e.surface_wipe.workflow import Phase

    scene = (
        Path.home()
        / ".guinsoo_mujoco/assets/mujoco_menagerie/universal_robots_ur5e/guinsoo_surface_wipe_scene.xml"
    )
    if not scene.exists():
        pytest.skip("UR5e surface_wipe scene assets not cached")
    session = SimSession.load("ur5e", "surface_wipe")
    saw_follow = False
    for _ in range(35000):
        sample = session.step(0.002)
        if sample["phase"] == Phase.FOLLOW.value:
            saw_follow = True
        if session.controller.phase == Phase.DONE:
            break
    assert saw_follow
    assert float(sample["path_s"]) >= WIPE_LENGTH - 0.01
