from __future__ import annotations

import numpy as np

from guinsoo_mujoco.operators.collision import CollisionModel
from guinsoo_mujoco.operators.ik import IkOptions
from guinsoo_mujoco.operators.admittance import NormalAdmittance
from guinsoo_mujoco.operators.surface import SineSheetSurface

# Menagerie home: clears the lowered wave surface at reset.
SAFE_HOME_QPOS = np.array(
    [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    dtype=float,
)

# Collision-free intermediate posture between safe home and wipe workspace.
PREP_QPOS = np.array(
    [-1.0082, -1.4619, 2.9043, -1.7507, -1.7056, -0.6057],
    dtype=float,
)

# Wipe-workspace IK fallback (reachable top-down poses; arm links excluded from IK collision).
WIPE_IK_SEED = np.array(
    [5.8375, 4.9302, -2.0453, 4.3525, -1.8403, -1.2114],
    dtype=float,
)

APPROACH_REACH_TOL = 0.02
APPROACH_DURATION = 3.0
MIN_NORMAL_OFFSET = 0.003
MAX_JOINT_STEP = 0.025

EE_SITE = "attachment_site"
WAVE_GEOM = "wave"
TOOL_BODY_NAMES: frozenset[str] = frozenset(
    {"wrist_1_link", "wrist_2_link", "wrist_3_link"}
)
FORCE_SENSOR = "tool_force"
TORQUE_SENSOR = "tool_torque"

UR5E_ROBOT_BODY_NAMES: frozenset[str] = frozenset(
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

ARM_LINK_BODY_NAMES: frozenset[str] = frozenset(
    {"shoulder_link", "upper_arm_link", "forearm_link"}
)

IGNORE_BODY_NAMES: frozenset[str] = frozenset(
    {"world", "wipe_start", "wipe_end", "wave_stand", "wave_surface"}
)

# 沿 Y 负侧擦拭，曲面降低以避免初始姿态穿模。
SURFACE = SineSheetSurface(
    x0=0.30,
    y0=-0.28,
    z0=0.36,
    amplitude=0.015,
    wavelength=0.12,
)

WIPE_LENGTH = 0.25
TANGENTIAL_SPEED = 0.025

PRE_CONTACT_OFFSET = 0.04
CONTACT_STANDOFF = 0.005
CONTACT_FORCE_THRESHOLD = 2.0
MAX_CONTACT_FORCE = 80.0
DESCEND_SPEED = 0.02
RETRACT_SPEED = 0.02
RETRACT_DISTANCE = 0.05
MIN_DESCEND_TIME = 0.2
DESCEND_STANDOFF_TOL = 0.001

ADMITTANCE = NormalAdmittance(
    mass=2.0,
    damping=200.0,
    stiffness=0.0,
    force_des=8.0,
    d_n_limit=0.025,
    force_lpf_alpha=0.08,
)

COLLISION_MODEL = CollisionModel(
    robot_body_names=UR5E_ROBOT_BODY_NAMES,
    collision_body_names=ARM_LINK_BODY_NAMES,
    ignore_body_names=IGNORE_BODY_NAMES,
    obstacle_geom_names=(WAVE_GEOM,),
    margin=0.008,
    ignore_geom_names=("floor", "stand_plate", "stand_leg_a", "stand_leg_b", "stand_leg_c"),
)

IK_OPTIONS = IkOptions(
    site_name=EE_SITE,
    max_iterations=120,
    position_tol=0.004,
    orientation_tol=0.1,
    damping=0.06,
    collision_model=None,
    position_only=False,
)

APPROACH_IK_TOLERANCE = IkOptions(
    site_name=EE_SITE,
    max_iterations=120,
    position_tol=0.005,
    orientation_tol=0.12,
    damping=0.06,
    collision_model=COLLISION_MODEL,
    position_only=False,
)

# Top-down wipe pose is far from PREP_QPOS in joint space; seed from wipe workspace.
APPROACH_IK_FALLBACK = IkOptions(
    site_name=EE_SITE,
    max_iterations=120,
    position_tol=0.005,
    orientation_tol=0.12,
    damping=0.06,
    collision_model=None,
    position_only=False,
)
