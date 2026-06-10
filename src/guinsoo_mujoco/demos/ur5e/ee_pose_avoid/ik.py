from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.collision import is_configuration_colliding
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import (
    COLLISION_MARGIN,
    EE_SITE,
    IK_DAMPING,
    IK_MAX_ITERATIONS,
    IK_ORIENTATION_TOL,
    IK_POSITION_TOL,
    OBSTACLE_GEOMS,
)

if TYPE_CHECKING:
    from guinsoo_mujoco.runtime import MuJoCoRuntime


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation_error = target @ current.T
    trace = np.trace(rotation_error)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    if abs(cos_angle - 1.0) < 1e-8:
        return np.zeros(3)
    angle = np.arccos(cos_angle)
    axis = np.array(
        [
            rotation_error[2, 1] - rotation_error[1, 2],
            rotation_error[0, 2] - rotation_error[2, 0],
            rotation_error[1, 0] - rotation_error[0, 1],
        ]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros(3)
    return axis / norm * angle


def _clip_to_limits(q: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(q, low, high)


def solve_ik(
    runtime: MuJoCoRuntime,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iterations: int = IK_MAX_ITERATIONS,
    position_tol: float = IK_POSITION_TOL,
    orientation_tol: float = IK_ORIENTATION_TOL,
    damping: float = IK_DAMPING,
    check_collision: bool = True,
) -> tuple[np.ndarray | None, float]:
    low, high = runtime.joint_limits()
    q = _clip_to_limits(np.asarray(q_init, dtype=float).copy(), low, high)
    best_q = None
    best_error = float("inf")

    for _ in range(max_iterations):
        runtime.forward(q)
        pos, rot = runtime.site_pose(EE_SITE)
        pos_err = target_pos - pos
        ori_err = _orientation_error(rot, target_rot)
        pos_norm = float(np.linalg.norm(pos_err))
        ori_norm = float(np.linalg.norm(ori_err))
        total_error = pos_norm + ori_norm
        if total_error < best_error:
            best_error = total_error
            best_q = q.copy()
        if pos_norm < position_tol and ori_norm < orientation_tol:
            if not check_collision or not is_configuration_colliding(
                runtime, q, OBSTACLE_GEOMS, margin=COLLISION_MARGIN
            ):
                return q.copy(), total_error
            break

        jac = runtime.site_jacobian(EE_SITE)
        error = np.concatenate([pos_err, ori_err])
        jj_t = jac @ jac.T
        damped = jj_t + (damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(damped, error)
        q = _clip_to_limits(q + dq, low, high)

    if best_q is None:
        return None, best_error
    if check_collision and is_configuration_colliding(
        runtime, best_q, OBSTACLE_GEOMS, margin=COLLISION_MARGIN
    ):
        return None, best_error
    return best_q, best_error


def solve_ik_multi_seed(
    runtime: MuJoCoRuntime,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    seeds: tuple[np.ndarray, ...],
) -> np.ndarray | None:
    best_solution = None
    best_error = float("inf")
    for seed in seeds:
        solution, error = solve_ik(runtime, target_pos, target_rot, seed)
        if solution is None:
            continue
        if error < best_error:
            best_error = error
            best_solution = solution
    return best_solution
