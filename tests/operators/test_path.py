import numpy as np
import pytest

from guinsoo_mujoco.operators.path import JointPathTracker, snap_path_start

LOW = np.full(6, -2.0 * np.pi)
HIGH = np.full(6, 2.0 * np.pi)


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


def test_tracker_actuator_anchor_keeps_first_target_near_qpos():
    tracker = JointPathTracker()
    q_anchor = np.array([3.895, 5.353, 0.592, 6.283, -1.554, -2.852])
    home = np.array([3.895, 5.335, 0.585, 5.261, 4.729, 3.431])
    tracker.set_path([q_anchor, home], q_anchor=q_anchor, joint_limits=(LOW, HIGH))
    target = tracker.target_at_progress()
    assert np.linalg.norm(target - q_anchor) < 0.1


def test_snap_path_start_replaces_first_node():
    path = [np.ones(6), np.full(6, 2.0)]
    snapped = snap_path_start(path, np.zeros(6))
    np.testing.assert_allclose(snapped[0], np.zeros(6))
    np.testing.assert_allclose(snapped[1], np.full(6, 2.0))
