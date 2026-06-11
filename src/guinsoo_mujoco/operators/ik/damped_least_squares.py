from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from guinsoo_mujoco.operators.collision import CollisionModel, is_configuration_colliding
from guinsoo_mujoco.operators.path.joint_unwrap import shortest_joint_delta

if TYPE_CHECKING:
    from guinsoo_mujoco.runtime import MuJoCoRuntime


@dataclass(frozen=True)
class IkOptions:
    site_name: str
    max_iterations: int = 80
    position_tol: float = 0.01
    orientation_tol: float = 0.05
    damping: float = 0.05
    collision_model: CollisionModel | None = None
    position_only: bool = False


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


def _is_colliding(
    runtime: MuJoCoRuntime,
    q: np.ndarray,
    collision_model: CollisionModel | None,
) -> bool:
    if collision_model is None:
        return False
    return is_configuration_colliding(runtime, q, collision_model)


def solve_ik(
    runtime: MuJoCoRuntime,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_init: np.ndarray,
    options: IkOptions,
    *,
    check_collision: bool = True,
) -> tuple[np.ndarray | None, float]:
    low, high = runtime.joint_limits()
    q = _clip_to_limits(np.asarray(q_init, dtype=float).copy(), low, high)
    best_q = None
    best_error = float("inf")
    collision_model = options.collision_model if check_collision else None

    for _ in range(options.max_iterations):
        runtime.forward(q)
        pos, rot = runtime.site_pose(options.site_name)
        pos_err = target_pos - pos
        ori_err = _orientation_error(rot, target_rot)
        pos_norm = float(np.linalg.norm(pos_err))
        ori_norm = float(np.linalg.norm(ori_err))
        total_error = pos_norm if options.position_only else pos_norm + ori_norm
        if total_error < best_error:
            best_error = total_error
            best_q = q.copy()
        orientation_ok = options.position_only or ori_norm < options.orientation_tol
        if pos_norm < options.position_tol and orientation_ok:
            if not _is_colliding(runtime, q, collision_model):
                return q.copy(), total_error
            break

        jac = runtime.site_jacobian(options.site_name)
        if options.position_only:
            error = pos_err
            jac = jac[:3, :]
        else:
            error = np.concatenate([pos_err, ori_err])
        jj_t = jac @ jac.T
        damped = jj_t + (options.damping**2) * np.eye(jj_t.shape[0])
        dq = jac.T @ np.linalg.solve(damped, error)
        q = _clip_to_limits(q + dq, low, high)

    if best_q is None:
        return None, best_error
    if _is_colliding(runtime, best_q, collision_model):
        return None, best_error
    return best_q, best_error


def _achieved_pose_error(
    runtime: MuJoCoRuntime,
    q: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    options: IkOptions,
    *,
    site_name: str,
) -> tuple[float, float]:
    runtime.forward(q)
    pos, rot = runtime.site_pose(site_name)
    pos_err = float(np.linalg.norm(pos - target_pos))
    if options.position_only:
        return pos_err, 0.0
    orient_err = float(
        np.arccos(np.clip(float(np.dot(rot[:, 2], target_rot[:, 2])), -1.0, 1.0))
    )
    return pos_err, orient_err


def solve_ik_nearest(
    runtime: MuJoCoRuntime,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_current: np.ndarray,
    options: IkOptions,
    *,
    fallback_seeds: tuple[np.ndarray, ...] = (),
) -> np.ndarray | None:
    """Prefer IK solutions close to q_current to avoid 2pi branch jumps."""
    candidates: list[np.ndarray] = []
    for seed in (q_current, *fallback_seeds):
        solution, _error = solve_ik(runtime, target_pos, target_rot, seed, options)
        if solution is None:
            continue
        pos_err, orient_err = _achieved_pose_error(
            runtime,
            solution,
            target_pos,
            target_rot,
            options,
            site_name=options.site_name,
        )
        orientation_ok = options.position_only or orient_err <= options.orientation_tol * 2.0
        if pos_err <= options.position_tol * 2.5 and orientation_ok:
            candidates.append(solution)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda q: float(np.linalg.norm(shortest_joint_delta(q_current, q))),
    )


def solve_ik_multi_seed(
    runtime: MuJoCoRuntime,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    seeds: tuple[np.ndarray, ...],
    options: IkOptions,
) -> np.ndarray | None:
    best_solution = None
    best_error = float("inf")
    for seed in seeds:
        solution, error = solve_ik(runtime, target_pos, target_rot, seed, options)
        if solution is None:
            continue
        if error < best_error:
            best_error = error
            best_solution = solution
    return best_solution
