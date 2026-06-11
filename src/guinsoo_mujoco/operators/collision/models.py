from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionModel:
    """Scene-specific collision configuration injected by each demo."""

    robot_body_names: frozenset[str]
    ignore_body_names: frozenset[str]
    obstacle_geom_names: tuple[str, ...]
    margin: float = 0.03
    ignore_geom_names: tuple[str, ...] = ("floor",)
    ignore_robot_geom_names: tuple[str, ...] = ()
    collision_body_names: frozenset[str] | None = None
