import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import HOME_QPOS
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller import EEPoseAvoidController
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.workflow import Phase


class FakeRuntime:
    def __init__(self, dof: int = 6):
        self.qpos = np.zeros(dof)
        self.qvel = np.zeros(dof)
        self.ctrl = None
        self.data = self

    def read_joint_state(self):
        return self.qpos.copy(), self.qvel.copy()

    def set_control(self, control):
        self.ctrl = np.asarray(control, dtype=float)

    def joint_limits(self):
        return np.full(self.qpos.size, -2.0 * np.pi), np.full(
            self.qpos.size, 2.0 * np.pi
        )


def test_hold_phase_transitions_to_plan_without_none_target(monkeypatch):
    runtime = FakeRuntime()
    controller = EEPoseAvoidController(runtime)
    controller.phase = Phase.HOLD
    controller.hold_target = np.zeros(6)
    controller.hold_elapsed = 1.0
    controller.waypoint_index = 0
    monkeypatch.setattr(
        controller,
        "_step_plan",
        lambda _runtime, _dt: controller._sample(_runtime, np.ones(6)),
    )

    sample = controller.step(runtime, t=0.0, dt=0.01)

    assert sample["target"] is not None
    np.testing.assert_allclose(sample["target"], np.ones(6))


def test_controller_transitions_from_track_to_hold(monkeypatch):
    runtime = FakeRuntime()
    controller = EEPoseAvoidController(runtime)
    controller.phase = Phase.TRACK
    controller.tracker.set_path([np.zeros(6), np.zeros(6)])
    controller.tracker.progress = controller.tracker.total_length

    sample = controller.step(runtime, t=0.0, dt=0.01)

    assert controller.phase == Phase.HOLD
    assert runtime.ctrl is not None


def test_control_target_stays_near_current_qpos_branch():
    runtime = FakeRuntime()
    runtime.qpos = np.array(HOME_QPOS, dtype=float)
    controller = EEPoseAvoidController(runtime)

    target = controller._control_target(runtime, HOME_QPOS)

    assert np.linalg.norm(target - runtime.qpos) < 0.2


def test_control_target_fixes_wrist_branch_near_two_pi():
    runtime = FakeRuntime()
    runtime.qpos = np.array([3.895, 5.353, 0.592, 6.283, -1.554, -2.852])
    controller = EEPoseAvoidController(runtime)
    place_ref = np.array([3.895, 5.335, 0.585, 1.274, -1.554, -2.852])

    target = controller._control_target(runtime, place_ref)

    assert target[3] == pytest.approx(1.274, abs=1e-3)


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_hold_at_home_does_not_jump_ctrl():
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from pathlib import Path

    from guinsoo_mujoco.runtime import MuJoCoRuntime

    scene = (
        Path.home()
        / ".guinsoo_mujoco/assets/mujoco_menagerie/universal_robots_ur5e/guinsoo_ee_pose_avoid_scene.xml"
    )
    if not scene.exists():
        pytest.skip("UR5e scene assets not cached")
    runtime = MuJoCoRuntime(scene)
    controller = EEPoseAvoidController(runtime)
    controller.reset(runtime)
    controller.phase = Phase.PLAN
    controller.waypoint_index = 0
    controller._step_plan(runtime, 0.0)

    qpos = runtime.data.qpos[:6].copy()
    ctrl = runtime.data.ctrl[:6].copy()
    assert np.linalg.norm(ctrl - qpos) < 0.2


def test_controller_plan_phase_uses_rrt_path(monkeypatch):
    runtime = FakeRuntime()
    controller = EEPoseAvoidController(runtime)
    controller.phase = Phase.PLAN
    planned_path = [np.zeros(6), np.array([0.2, -0.5, 1.0, -0.5, -1.0, 0.1])]

    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller.EEPoseAvoidController._is_goal_free",
        lambda _self, _runtime, _q_goal: False,
    )
    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller.solve_ik_multi_seed",
        lambda *_args, **_kwargs: planned_path[-1],
    )
    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller.densify_path",
        lambda *_args, **_kwargs: planned_path,
    )
    controller.planner = type(
        "Planner",
        (),
        {"plan": lambda self, q_start, q_goal: planned_path},
    )()

    sample = controller.step(runtime, t=0.0, dt=0.0)

    assert controller.phase == Phase.TRACK
    assert sample["plan_success"] is True
    np.testing.assert_allclose(runtime.ctrl, sample["target"])
