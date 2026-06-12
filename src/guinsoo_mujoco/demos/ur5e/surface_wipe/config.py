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

# Wipe-start posture: approach pose above path origin (PRE_CONTACT_OFFSET along +n).
START_QPOS = np.array(
    [6.0678, 3.8673, -0.5038, 1.2819, -1.5561, -1.7857],
    dtype=float,
)

# IK seed near the wipe workspace (arm links excluded from IK collision).
WIPE_IK_SEED = START_QPOS.copy()

MIN_NORMAL_OFFSET = 0.003
MAX_JOINT_STEP = 0.025
FOLLOW_MAX_JOINT_STEP = 0.006
JOINT_TARGET_SMOOTHING = 0.07
FOLLOW_IK_DECIMATION = 8

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

# 沿机械臂 -Y 可行域布置缓波曲面；从远端向基座方向擦拭（direction=-1）。
SURFACE = SineSheetSurface(
    x0=0.55,
    y0=-0.26,
    z0=0.40,
    amplitude=0.004,
    wavelength=0.36,
    direction=-1.0,
)

WIPE_LENGTH = 0.22
TANGENTIAL_SPEED = 0.005

PRE_CONTACT_OFFSET = 0.035
CONTACT_STANDOFF = 0.005
CONTACT_FORCE_THRESHOLD = 2.0
MAX_CONTACT_FORCE = 80.0
CONTACT_SETTLE_TIME = 0.45
CONTACT_SETTLE_QVEL = 0.075
DESCEND_SPEED = 0.02
RETRACT_SPEED = 0.02
RETRACT_DISTANCE = 0.05
MIN_DESCEND_TIME = 0.2
DESCEND_STANDOFF_TOL = 0.001

ADMITTANCE = NormalAdmittance(
    mass=3.0,
    damping=500.0,
    stiffness=80.0,
    force_des=2.0,
    d_n_limit=0.012,
    force_lpf_alpha=0.05,
    d_n_rate_limit=0.0015,
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
