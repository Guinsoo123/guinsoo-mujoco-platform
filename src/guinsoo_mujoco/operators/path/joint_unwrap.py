from __future__ import annotations

import numpy as np


def shortest_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Per-joint shortest angular delta from q_from to q_to."""
    delta = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_joint_target(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Return equivalent q_to reachable from q_from via shortest joint motion."""
    return np.asarray(q_from, dtype=float) + shortest_joint_delta(q_from, q_to)


def _actuator_target_scalar(
    qpos: float,
    q_ref: float,
    low: float,
    high: float,
) -> float:
    """Pick q_ref + n*2pi within [low, high] that minimizes |ctrl - qpos|."""
    span = high - low
    if span < 2.0 * np.pi - 1e-6:
        return float(np.clip(q_ref, low, high))

    base = float(np.arctan2(np.sin(q_ref), np.cos(q_ref)))
    n_min = int(np.ceil((low - base) / (2.0 * np.pi)))
    n_max = int(np.floor((high - base) / (2.0 * np.pi)))
    best = base
    best_distance = abs(best - qpos)
    for shift in range(n_min, n_max + 1):
        candidate = base + shift * (2.0 * np.pi)
        distance = abs(candidate - qpos)
        if distance < best_distance:
            best_distance = distance
            best = candidate
    return float(best)


def anchor_path_for_actuator(
    path: list[np.ndarray],
    q_anchor: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> list[np.ndarray]:
    """Rewrite path nodes into a continuous actuator ctrl branch from q_anchor."""
    if not path:
        return []
    raw = [np.asarray(node, dtype=float).copy() for node in path]
    anchored = [actuator_joint_target(q_anchor, raw[0], low, high)]
    for index in range(1, len(raw)):
        anchored.append(actuator_joint_target(anchored[-1], raw[index], low, high))
    return anchored


def actuator_joint_target(
    qpos: np.ndarray,
    q_ref: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map a joint reference for MuJoCo position actuators (linear ctrl - q error)."""
    qpos = np.asarray(qpos, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    size = min(qpos.size, q_ref.size, low.size, high.size)
    target = q_ref[:size].copy()
    for index in range(size):
        target[index] = _actuator_target_scalar(
            float(qpos[index]),
            float(q_ref[index]),
            float(low[index]),
            float(high[index]),
        )
    return target


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
