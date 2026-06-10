import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller import EEPoseAvoidController, Phase
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.path_tracker import JointPathTracker


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


def test_joint_path_tracker_interpolates_and_completes():
    tracker = JointPathTracker(speed=1.0, arrival_tol=0.05)
    path = [np.zeros(3), np.ones(3)]
    tracker.set_path(path)

    tracker.advance(0.5)
    target, _ = tracker.control(np.zeros(3), np.zeros(3))
    expected = 0.5 / np.sqrt(3.0)
    assert target[0] == pytest.approx(expected)

    tracker.advance(2.0)
    assert tracker.is_complete(np.ones(3) * 0.98)


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
