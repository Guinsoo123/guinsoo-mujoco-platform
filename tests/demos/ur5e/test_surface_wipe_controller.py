import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import (
    ADMITTANCE,
    CONTACT_SETTLE_TIME,
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

    def set_joint_positions(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)

    def set_control(self, control):
        self.ctrl = np.asarray(control, dtype=float)

    def forward(self, qpos=None):
        if qpos is not None:
            self.set_joint_positions(qpos)

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


def test_reset_starts_at_wipe_origin_without_approach_phase():
    runtime = FakeRuntime()
    controller = SurfaceWipeController(runtime)
    controller.reset(runtime)

    assert controller.phase == Phase.DESCEND
    assert controller.approach_alpha == 1.0
    assert controller.descend_offset > 0.0


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

    controller.step(runtime, t=0.0, dt=CONTACT_SETTLE_TIME + 0.01)

    assert controller.phase == Phase.FOLLOW


def test_cartesian_errors_use_admittance_offset():
    runtime = FakeRuntime()
    controller = SurfaceWipeController(runtime)
    controller.path_s = 0.05
    controller.last_normal_offset = 0.008
    pos = np.array([0.52, -0.26, 0.402])
    errors = controller._cartesian_errors(pos, s=0.05, normal_offset=0.008)
    assert errors["ee_pose_error"] >= 0.0
    assert "ee_normal_error" in errors
    assert "ee_tangential_error" in errors
    assert errors["ee_surface_distance"] > 0.0


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
    q_before, _ = session.runtime.read_joint_state()
    sample = session.step(0.002)
    q_after, _ = session.runtime.read_joint_state()
    assert float(np.max(np.abs(q_after - q_before))) < 0.02
    assert sample["target"] is not None
    assert sample["phase"] == Phase.DESCEND.value
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
    follow_dq_steps: list[float] = []
    follow_ctrl_steps: list[float] = []
    in_descend = False
    prev_ctrl = None

    for _ in range(35000):
        sample = session.step(0.002)
        q, _ = rt.read_joint_state()
        dq_step = None
        if prev_q is not None:
            dq_step = float(np.max(np.abs(q - prev_q)))
            max_dq = max(max_dq, dq_step)

        if sample["phase"] == Phase.DESCEND.value:
            in_descend = True
            descend_steps += 1
        elif in_descend and sample["phase"] == Phase.FOLLOW.value:
            in_descend = False

        if sample["phase"] == Phase.FOLLOW.value:
            follow_forces.append(float(sample["force_normal_raw"]))
            if dq_step is not None:
                follow_dq_steps.append(dq_step)
            if prev_ctrl is not None:
                follow_ctrl_steps.append(
                    float(np.max(np.abs(sample["control"] - prev_ctrl)))
                )
        prev_ctrl = sample["control"].copy()
        prev_q = q.copy()

        if session.controller.phase == Phase.DONE:
            break

    assert descend_steps * rt.model.opt.timestep < 8.0
    assert max_dq < 0.35
    assert follow_forces
    follow_arr = np.asarray(follow_forces, dtype=float)
    assert float(np.median(follow_arr)) > 1.0
    assert float(np.std(follow_arr)) > 0.05
    follow_dq_arr = np.asarray(follow_dq_steps, dtype=float)
    follow_ctrl_arr = np.asarray(follow_ctrl_steps, dtype=float)
    assert float(np.percentile(follow_dq_arr, 99)) < 0.00035
    assert float(np.percentile(follow_ctrl_arr, 99)) < 0.0006
    assert float(np.max(sample["path_s"])) >= WIPE_LENGTH - 0.01


def test_done_phase_holds_joint_target_steady():
    from pathlib import Path

    from guinsoo_mujoco.app.session import SimSession

    scene = (
        Path.home()
        / ".guinsoo_mujoco/assets/mujoco_menagerie/universal_robots_ur5e/guinsoo_surface_wipe_scene.xml"
    )
    if not scene.exists():
        pytest.skip("UR5e surface_wipe scene assets not cached")

    session = SimSession.load("ur5e", "surface_wipe")
    ctrl = session.controller
    rt = session.runtime
    for _ in range(35000):
        session.step(0.002)
        if ctrl.phase == Phase.DONE:
            break
    assert ctrl.phase == Phase.DONE
    assert ctrl.hold_q_target is not None

    for _ in range(200):
        session.step(0.002)

    targets = []
    qvels = []
    for _ in range(300):
        session.step(0.002)
        targets.append(ctrl.hold_q_target.copy())
        qvels.append(rt.read_joint_state()[1].copy())

    target_arr = np.asarray(targets)
    qvel_arr = np.asarray(qvels)
    assert float(np.max(np.abs(np.diff(target_arr, axis=0)))) == 0.0
    assert float(np.max(np.abs(qvel_arr))) < 0.08
