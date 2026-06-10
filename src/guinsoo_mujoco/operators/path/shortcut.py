from __future__ import annotations

import numpy as np

from guinsoo_mujoco.operators.collision import CollisionModel, is_edge_colliding
from guinsoo_mujoco.operators.path.joint_unwrap import unwrap_joint_target
from guinsoo_mujoco.runtime import MuJoCoRuntime


def shortcut_path(
    runtime: MuJoCoRuntime,
    path: list[np.ndarray],
    collision_model: CollisionModel,
    *,
    edge_collision_samples: int = 10,
) -> list[np.ndarray]:
    """Greedy shortcutting: skip intermediate nodes when a direct edge is free."""
    if len(path) < 3:
        return [np.asarray(node, dtype=float).copy() for node in path]

    nodes = [np.asarray(node, dtype=float).copy() for node in path]
    shortened: list[np.ndarray] = [nodes[0].copy()]
    index = 0
    while index < len(nodes) - 1:
        best = index + 1
        for candidate in range(len(nodes) - 1, index, -1):
            target = unwrap_joint_target(shortened[-1], nodes[candidate])
            if not is_edge_colliding(
                runtime,
                shortened[-1],
                target,
                collision_model,
                samples=edge_collision_samples,
            ):
                best = candidate
                break
        shortened.append(unwrap_joint_target(shortened[-1], nodes[best]))
        index = best
    return shortened
