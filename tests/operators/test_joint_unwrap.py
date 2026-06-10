import numpy as np
import pytest

from guinsoo_mujoco.operators.path import (
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


def test_canonicalize_joint_q_maps_into_limits():
    low = np.array([-2.0 * np.pi] * 2)
    high = np.array([2.0 * np.pi] * 2)
    q = np.array([0.0, 7.5])
    canonical = canonicalize_joint_q(q, low, high)
    assert canonical[1] == pytest.approx(7.5 - 2.0 * np.pi, abs=1e-6)
    assert low[1] <= canonical[1] <= high[1]
