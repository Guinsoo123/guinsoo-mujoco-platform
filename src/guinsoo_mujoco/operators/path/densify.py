from __future__ import annotations

import numpy as np

from guinsoo_mujoco.operators.collision import CollisionModel, is_configuration_colliding
from guinsoo_mujoco.operators.path.joint_unwrap import interpolate_joints, unwrap_joint_target
from guinsoo_mujoco.runtime import MuJoCoRuntime


def densify_path(
    runtime: MuJoCoRuntime,
    path: list[np.ndarray],
    collision_model: CollisionModel,
    *,
    max_joint_step: float = 0.05,
) -> list[np.ndarray] | None:
    if not path:
        return []
    if len(path) == 1:
        return [np.asarray(path[0], dtype=float).copy()]

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float).copy()]
    for index in range(len(path) - 1):
        q_from = np.asarray(path[index], dtype=float)
        q_to = unwrap_joint_target(q_from, path[index + 1])
        delta = q_to - q_from
        segment_length = float(np.linalg.norm(delta))
        steps = max(1, int(np.ceil(segment_length / max_joint_step)))
        for step in range(1, steps + 1):
            alpha = step / steps
            q_mid = interpolate_joints(q_from, q_to, alpha)
            if is_configuration_colliding(runtime, q_mid, collision_model):
                return None
            dense.append(q_mid.copy())
    return dense


def snap_path_start(path: list[np.ndarray], q_start: np.ndarray) -> list[np.ndarray]:
    if not path:
        return [np.asarray(q_start, dtype=float).copy()]
    snapped = [np.asarray(node, dtype=float).copy() for node in path]
    snapped[0] = np.asarray(q_start, dtype=float).copy()
    return snapped
