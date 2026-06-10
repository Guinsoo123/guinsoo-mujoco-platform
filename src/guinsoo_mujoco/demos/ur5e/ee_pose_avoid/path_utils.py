from __future__ import annotations

import numpy as np

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.collision import (
    is_configuration_colliding,
    is_edge_colliding,
)
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import COLLISION_MARGIN, OBSTACLE_GEOMS
from guinsoo_mujoco.runtime import MuJoCoRuntime


def densify_path(
    runtime: MuJoCoRuntime,
    path: list[np.ndarray],
    *,
    max_joint_step: float = 0.04,
    margin: float = COLLISION_MARGIN,
) -> list[np.ndarray] | None:
    if not path:
        return []
    if len(path) == 1:
        return [np.asarray(path[0], dtype=float).copy()]

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float).copy()]
    for index in range(len(path) - 1):
        q_from = np.asarray(path[index], dtype=float)
        q_to = np.asarray(path[index + 1], dtype=float)
        segment_length = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(segment_length / max_joint_step)))
        for step in range(1, steps + 1):
            alpha = step / steps
            q_mid = q_from + alpha * (q_to - q_from)
            if is_configuration_colliding(runtime, q_mid, OBSTACLE_GEOMS, margin=margin):
                return None
            dense.append(q_mid.copy())
    return dense


def snap_path_start(path: list[np.ndarray], q_start: np.ndarray) -> list[np.ndarray]:
    if not path:
        return [np.asarray(q_start, dtype=float).copy()]
    snapped = [np.asarray(node, dtype=float).copy() for node in path]
    snapped[0] = np.asarray(q_start, dtype=float).copy()
    return snapped
