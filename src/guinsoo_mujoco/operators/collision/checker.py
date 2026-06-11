from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from guinsoo_mujoco.operators.collision.models import CollisionModel

if TYPE_CHECKING:
    from guinsoo_mujoco.runtime import MuJoCoRuntime


def _body_name(runtime: MuJoCoRuntime, body_id: int) -> str:
    name = runtime.mujoco.mj_id2name(
        runtime.model, runtime.mujoco.mjtObj.mjOBJ_BODY, int(body_id)
    )
    return name or f"body_{body_id}"


def robot_collision_geom_ids(
    runtime: MuJoCoRuntime,
    model: CollisionModel,
) -> tuple[int, ...]:
    ids: list[int] = []
    for geom_index in range(runtime.model.ngeom):
        if int(runtime.model.geom_contype[geom_index]) == 0:
            continue
        geom_name = runtime.mujoco.mj_id2name(
            runtime.model, runtime.mujoco.mjtObj.mjOBJ_GEOM, geom_index
        )
        if geom_name in model.ignore_robot_geom_names:
            continue
        body_id = int(runtime.model.geom_bodyid[geom_index])
        body_name = _body_name(runtime, body_id)
        if model.collision_body_names is not None:
            if body_name not in model.collision_body_names:
                continue
        elif body_name not in model.robot_body_names:
            continue
        ids.append(geom_index)
    return tuple(ids)


def obstacle_collision_geom_ids(
    runtime: MuJoCoRuntime,
    model: CollisionModel,
) -> tuple[int, ...]:
    ids: list[int] = []
    for name in model.obstacle_geom_names:
        try:
            ids.append(runtime.geom_id(name))
        except ValueError:
            continue
    for geom_index in range(runtime.model.ngeom):
        if int(runtime.model.geom_contype[geom_index]) == 0:
            continue
        body_id = int(runtime.model.geom_bodyid[geom_index])
        body_name = _body_name(runtime, body_id)
        if body_name in model.robot_body_names or body_name in model.ignore_body_names:
            continue
        if body_name == "world":
            continue
        if geom_index not in ids:
            ids.append(geom_index)
    return tuple(ids)


def is_configuration_colliding(
    runtime: MuJoCoRuntime,
    qpos: np.ndarray,
    model: CollisionModel,
) -> bool:
    saved_qpos = runtime.data.qpos.copy()
    try:
        runtime.set_joint_positions(qpos)
        runtime.forward()

        robot_ids = robot_collision_geom_ids(runtime, model)
        obstacle_ids = obstacle_collision_geom_ids(runtime, model)
        ignore_ids = set()
        for name in model.ignore_geom_names:
            try:
                ignore_ids.add(runtime.geom_id(name))
            except ValueError:
                continue

        for robot_id in robot_ids:
            for obstacle_id in obstacle_ids:
                if obstacle_id in ignore_ids:
                    continue
                distance = runtime.geom_distance_ids(
                    robot_id, obstacle_id, distmax=1.0
                )
                if distance < model.margin:
                    return True

        for contact_index in range(int(runtime.data.ncon)):
            contact = runtime.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in ignore_ids or geom2 in ignore_ids:
                continue
            if geom1 in robot_ids and geom2 in obstacle_ids:
                if float(contact.dist) < model.margin:
                    return True
            if geom2 in robot_ids and geom1 in obstacle_ids:
                if float(contact.dist) < model.margin:
                    return True
        return False
    finally:
        runtime.data.qpos[:] = saved_qpos
        runtime.forward()


def is_edge_colliding(
    runtime: MuJoCoRuntime,
    q_from: np.ndarray,
    q_to: np.ndarray,
    model: CollisionModel,
    *,
    samples: int = 8,
) -> bool:
    if samples < 2:
        samples = 2
    for alpha in np.linspace(0.0, 1.0, samples):
        q = (1.0 - alpha) * q_from + alpha * q_to
        if is_configuration_colliding(runtime, q, model):
            return True
    return False
