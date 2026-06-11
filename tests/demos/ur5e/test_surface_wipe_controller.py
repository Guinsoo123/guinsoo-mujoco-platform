import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import (
    ADMITTANCE,
    PRE_CONTACT_OFFSET,
    WIPE_LENGTH,
)
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


def test_approach_blends_without_instant_phase_skip(monkeypatch):
    runtime = FakeRuntime()
    controller = SurfaceWipeController(runtime)
    controller.approach_q_goal = np.array([0.2, -0.5, 1.2, -0.8, -0.7, -0.2])
    controller.approach_q_start = runtime.qpos.copy()

    sample = controller.step(runtime, t=0.0, dt=0.02)

    assert controller.phase == Phase.APPROACH
    assert 0.0 < sample["approach_alpha"] < 1.0


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
        "_signed_standoff",
        lambda _runtime, *, s: 0.004,
    )
    monkeypatch.setattr(
        controller,
        "_solve_pose",
        lambda _runtime, _pose, _options, **kwargs: np.ones(6),
    )

    controller.step(runtime, t=0.0, dt=0.01)

    assert controller.phase == Phase.FOLLOW


def test_read_normal_force_returns_raw_for_admittance():
    class WrenchRuntime(FakeRuntime):
        def read_site_wrench(self, force_sensor, torque_sensor):
            del force_sensor, torque_sensor
            return np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0], dtype=float)

    runtime = WrenchRuntime()
    controller = SurfaceWipeController(runtime)
    raw = controller._read_normal_force(runtime, s=0.0)
    assert raw > 40.0
    assert controller.last_force_normal_raw == raw
    assert controller.last_force_normal <= 80.0


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_scene_loads():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession

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
    assert "{{HFIELD_ELEVATION}}" not in scene.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_reaches_done():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession

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


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_quality_metrics():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession

    scene = (
        Path.home()
        / ".guinsoo_mujoco/assets/mujoco_menagerie/universal_robots_ur5e/guinsoo_surface_wipe_scene.xml"
    )
    if not scene.exists():
        pytest.skip("UR5e surface_wipe scene assets not cached")

    session = SimSession.load("ur5e", "surface_wipe")
    rt = session.runtime
    prev_q = None
    max_dq = 0.0
    descend_steps = 0
    follow_forces: list[float] = []
    in_descend = False

    for _ in range(35000):
        sample = session.step(0.002)
        q, _ = rt.read_joint_state()
        if prev_q is not None:
            max_dq = max(max_dq, float(np.max(np.abs(q - prev_q))))
        prev_q = q.copy()

        if sample["phase"] == Phase.DESCEND.value:
            in_descend = True
            descend_steps += 1
        elif in_descend and sample["phase"] == Phase.FOLLOW.value:
            in_descend = False

        if sample["phase"] == Phase.FOLLOW.value:
            follow_forces.append(float(sample["force_normal_raw"]))

        if session.controller.phase == Phase.DONE:
            break

    assert descend_steps * rt.model.opt.timestep < 8.0
    assert max_dq < 0.35
    assert follow_forces
    follow_arr = np.asarray(follow_forces, dtype=float)
    assert float(np.median(follow_arr)) > 5.0
    assert float(np.std(follow_arr)) > 1.0
    assert float(np.max(sample["path_s"])) >= WIPE_LENGTH - 0.01
