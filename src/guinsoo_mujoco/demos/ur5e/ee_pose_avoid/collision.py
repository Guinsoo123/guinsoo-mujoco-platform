from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from guinsoo_mujoco.runtime import MuJoCoRuntime

ROBOT_BODY_NAMES: frozenset[str] = frozenset(
    {
        "base",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    }
)

IGNORE_BODY_NAMES: frozenset[str] = frozenset({"world", "target_home", "target_approach", "target_over", "target_place"})


def _geom_name(runtime: MuJoCoRuntime, geom_id: int) -> str:
    name = runtime.mujoco.mj_id2name(
        runtime.model, runtime.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
    )
    return name or f"geom_{geom_id}"


def _body_name(runtime: MuJoCoRuntime, body_id: int) -> str:
    name = runtime.mujoco.mj_id2name(
        runtime.model, runtime.mujoco.mjtObj.mjOBJ_BODY, int(body_id)
    )
    return name or f"body_{body_id}"


def robot_collision_geom_ids(runtime: MuJoCoRuntime) -> tuple[int, ...]:
    ids: list[int] = []
    for geom_index in range(runtime.model.ngeom):
        if int(runtime.model.geom_contype[geom_index]) == 0:
            continue
        body_id = int(runtime.model.geom_bodyid[geom_index])
        if _body_name(runtime, body_id) in ROBOT_BODY_NAMES:
            ids.append(geom_index)
    return tuple(ids)


def obstacle_collision_geom_ids(
    runtime: MuJoCoRuntime,
    obstacle_geom_names: tuple[str, ...],
) -> tuple[int, ...]:
    ids: list[int] = []
    for name in obstacle_geom_names:
        try:
            ids.append(runtime.geom_id(name))
        except ValueError:
            continue
    for geom_index in range(runtime.model.ngeom):
        if int(runtime.model.geom_contype[geom_index]) == 0:
            continue
        body_id = int(runtime.model.geom_bodyid[geom_index])
        body_name = _body_name(runtime, body_id)
        if body_name in ROBOT_BODY_NAMES or body_name in IGNORE_BODY_NAMES:
            continue
        if body_name == "world":
            continue
        if geom_index not in ids:
            ids.append(geom_index)
    return tuple(ids)


def is_configuration_colliding(
    runtime: MuJoCoRuntime,
    qpos: np.ndarray,
    obstacle_geom_names: tuple[str, ...],
    *,
    margin: float = 0.03,
    ignore_geom_names: tuple[str, ...] = ("floor",),
) -> bool:
    saved_qpos = runtime.data.qpos.copy()
    try:
        runtime.set_joint_positions(qpos)
        runtime.forward()

        robot_ids = robot_collision_geom_ids(runtime)
        obstacle_ids = obstacle_collision_geom_ids(runtime, obstacle_geom_names)
        ignore_ids = set()
        for name in ignore_geom_names:
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
                if distance < margin:
                    return True

        for contact_index in range(int(runtime.data.ncon)):
            contact = runtime.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in ignore_ids or geom2 in ignore_ids:
                continue
            if geom1 in robot_ids and geom2 in obstacle_ids:
                if float(contact.dist) < margin:
                    return True
            if geom2 in robot_ids and geom1 in obstacle_ids:
                if float(contact.dist) < margin:
                    return True
        return False
    finally:
        runtime.data.qpos[:] = saved_qpos
        runtime.forward()


def is_edge_colliding(
    runtime: MuJoCoRuntime,
    q_from: np.ndarray,
    q_to: np.ndarray,
    obstacle_geom_names: tuple[str, ...],
    *,
    margin: float = 0.03,
    samples: int = 8,
) -> bool:
    if samples < 2:
        samples = 2
    for alpha in np.linspace(0.0, 1.0, samples):
        q = (1.0 - alpha) * q_from + alpha * q_to
        if is_configuration_colliding(
            runtime, q, obstacle_geom_names, margin=margin
        ):
            return True
    return False
