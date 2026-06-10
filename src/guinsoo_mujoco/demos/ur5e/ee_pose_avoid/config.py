from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    name: str
    mocap_body: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    joint_goal: tuple[float, float, float, float, float, float]


WAYPOINTS: tuple[Waypoint, ...] = (
    Waypoint(
        name="home",
        mocap_body="target_home",
        position=(0.207, -0.2414, 0.9741),
        quaternion_wxyz=(0.6627, 0.4436, 0.5988, -0.0742),
        joint_goal=(6.027, 4.91, -0.872, 5.261, -5.202, -4.127),
    ),
    Waypoint(
        name="approach",
        mocap_body="target_approach",
        position=(0.4251, -0.452, 0.8209),
        quaternion_wxyz=(-0.3127, 0.328, -0.0397, 0.8905),
        joint_goal=(5.605, 4.292, -0.503, -2.681, 4.204, 2.802),
    ),
    Waypoint(
        name="over_obstacle",
        mocap_body="target_over",
        position=(0.417, 0.4729, 0.6578),
        quaternion_wxyz=(0.5561, -0.6627, 0.1994, 0.4603),
        joint_goal=(3.61, 5.165, 0.601, -0.03, -6.202, 4.671),
    ),
    Waypoint(
        name="place",
        mocap_body="target_place",
        position=(0.2529, 0.4232, 0.6652),
        quaternion_wxyz=(0.2494, -0.177, -0.2715, 0.9125),
        joint_goal=(3.895, 5.335, 0.585, 1.274, 4.729, 3.431),
    ),
)

EE_SITE = "attachment_site"

OBSTACLE_GEOMS: tuple[str, ...] = (
    "table",
    "obstacle_a",
    "obstacle_b",
    "obstacle_c",
)

HOME_QPOS = np.array(WAYPOINTS[0].joint_goal, dtype=float)

RRT_STEP_SIZE = 0.08
RRT_GOAL_BIAS = 0.25
RRT_MAX_ITERATIONS = 5000
COLLISION_MARGIN = 0.05
PATH_DENSIFY_STEP = 0.05
EDGE_COLLISION_SAMPLES = 10

IK_MAX_ITERATIONS = 80
IK_POSITION_TOL = 0.01
IK_ORIENTATION_TOL = 0.05
IK_DAMPING = 0.05

TRACK_KP = 20.0
TRACK_KD = 2.0
PATH_SPEED = 1.5
JOINT_ARRIVAL_TOL = 0.05
HOLD_DURATION = 1.0
