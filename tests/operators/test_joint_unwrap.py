import numpy as np
import pytest

from guinsoo_mujoco.operators.path import (
    actuator_joint_target,
    anchor_path_for_actuator,
    canonicalize_joint_q,
    interpolate_joints,
    shortest_joint_delta,
    unwrap_joint_target,
    unwrap_path,
)


def test_shortest_joint_delta_avoids_full_rotation():
    q_from = np.array([0.0, 0.0])
    q_to = np.array([0.1, 6.0])
    delta = shortest_joint_delta(q_from, q_to)
    assert delta[1] == pytest.approx(-0.283185, abs=1e-4)
    assert np.linalg.norm(delta) < 1.0


def test_unwrap_joint_target_follows_shortest_motion():
    q_from = np.array([0.0, 0.0])
    q_to = np.array([0.1, 6.0])
    unwrapped = unwrap_joint_target(q_from, q_to)
    assert unwrapped[1] == pytest.approx(-0.283185, abs=1e-4)


def test_interpolate_joints_uses_shortest_path():
    q_from = np.zeros(2)
    q_to = np.array([0.0, 2.0 * np.pi - 0.2])
    mid = interpolate_joints(q_from, q_to, 0.5)
    assert mid[1] == pytest.approx(-0.1, abs=1e-6)


def test_unwrap_path_rewrites_consecutive_nodes():
    path = [
        np.zeros(2),
        np.array([0.5, 6.0]),
        np.array([1.0, 6.5]),
    ]
    unwrapped = unwrap_path(path, q_start=np.zeros(2))
    assert unwrapped[0][1] == pytest.approx(0.0)
    assert unwrapped[1][1] == pytest.approx(-0.283185, abs=1e-4)


def test_actuator_joint_target_avoids_wrong_2pi_branch_at_limit():
    """Regression: wrist_1 at 2*pi must ctrl to 1.274, not 7.557."""
    low = np.array([-2.0 * np.pi] * 6)
    high = np.array([2.0 * np.pi] * 6)
    qpos = np.array([3.895, 5.353, 0.592, 6.283, -1.554, -2.852])
    q_ref = np.array([3.895, 5.335, 0.585, 1.274, -1.554, -2.852])
    target = actuator_joint_target(qpos, q_ref, low, high)
    assert target[3] == pytest.approx(1.274, abs=1e-3)
    assert target[3] != pytest.approx(7.557, abs=1e-2)


def test_anchor_path_for_actuator_avoids_segment_start_discontinuity():
    low = np.array([-2.0 * np.pi] * 6)
    high = np.array([2.0 * np.pi] * 6)
    q_anchor = np.array([3.895, 5.353, 0.592, 6.283, -1.554, -2.852])
    home_raw = np.array([3.895, 5.335, 0.585, 5.261, 4.729, 3.431])
    anchored = anchor_path_for_actuator([q_anchor, home_raw], q_anchor, low, high)
    assert abs(anchored[0][3] - q_anchor[3]) < 0.05
    assert abs(anchored[1][3] - anchored[0][3]) < 6.0
    assert anchored[1][3] == pytest.approx(5.261, abs=0.2)


def test_actuator_joint_target_stays_near_qpos_for_home_pose():
    low = np.array([-2.0 * np.pi] * 6)
    high = np.array([2.0 * np.pi] * 6)
    home = np.array([6.027, 4.91, -0.872, 5.261, -5.202, -4.127])
    target = actuator_joint_target(home, home, low, high)
    assert np.linalg.norm(target - home) < 0.2


def test_canonicalize_joint_q_maps_into_limits():
    low = np.array([-2.0 * np.pi] * 2)
    high = np.array([2.0 * np.pi] * 2)
    q = np.array([0.0, 7.5])
    canonical = canonicalize_joint_q(q, low, high)
    assert canonical[1] == pytest.approx(7.5 - 2.0 * np.pi, abs=1e-6)
    assert low[1] <= canonical[1] <= high[1]
