from __future__ import annotations

import numpy as np


def shortest_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Per-joint shortest angular delta from q_from to q_to."""
    delta = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_joint_target(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Return equivalent q_to reachable from q_from via shortest joint motion."""
    return np.asarray(q_from, dtype=float) + shortest_joint_delta(q_from, q_to)


def interpolate_joints(
    q_from: np.ndarray,
    q_to: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear interpolation along shortest joint-space path."""
    return np.asarray(q_from, dtype=float) + alpha * shortest_joint_delta(q_from, q_to)


def canonicalize_joint_q(
    q: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map continuous joint trajectory samples into actuator/joint limit range."""
    value = np.asarray(q, dtype=float).copy()
    wrapped = np.arctan2(np.sin(value), np.cos(value))
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    size = min(value.size, low.size, high.size)
    for index in range(size):
        if high[index] - low[index] >= 2.0 * np.pi - 1e-6:
            candidate = wrapped[index]
            while candidate > high[index]:
                candidate -= 2.0 * np.pi
            while candidate < low[index]:
                candidate += 2.0 * np.pi
            value[index] = candidate
        else:
            value[index] = np.clip(wrapped[index], low[index], high[index])
    return value


def unwrap_path(
    path: list[np.ndarray],
    q_start: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Rewrites path nodes so consecutive segments use shortest joint motion."""
    if not path:
        return []
    out: list[np.ndarray] = []
    current = (
        np.asarray(q_start, dtype=float).copy()
        if q_start is not None
        else np.asarray(path[0], dtype=float).copy()
    )
    out.append(current.copy())
    start_index = 0 if q_start is None else 1
    for index in range(start_index, len(path)):
        current = unwrap_joint_target(current, path[index])
        out.append(current.copy())
    return out
