from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import (
    COLLISION_MODEL,
    EDGE_COLLISION_SAMPLES,
    HOLD_DURATION,
    HOME_QPOS,
    IK_OPTIONS,
    JOINT_ARRIVAL_TOL,
    PATH_DENSIFY_STEP,
    RRT_GOAL_BIAS,
    RRT_MAX_ITERATIONS,
    RRT_STEP_SIZE,
    TRACK_KD,
    TRACK_KP,
    PATH_SPEED,
    WAYPOINTS,
    Waypoint,
)
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.workflow import Phase
from guinsoo_mujoco.logging_config import get_logger
from guinsoo_mujoco.operators.collision import is_configuration_colliding
from guinsoo_mujoco.operators.ik import solve_ik_multi_seed
from guinsoo_mujoco.operators.path import (
    JointPathTracker,
    actuator_joint_target,
    densify_path,
    shortest_joint_delta,
    shortcut_path,
    snap_path_start,
    unwrap_path,
)
from guinsoo_mujoco.operators.rrt import RRTConnectPlanner
from guinsoo_mujoco.runtime import MuJoCoRuntime

controller_logger = get_logger("controller")
planner_logger = get_logger("planner")


def _quat_wxyz_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


@dataclass
class EEPoseAvoidController:
    runtime: MuJoCoRuntime
    name: str = "ee_pose_avoid"
    waypoint_index: int = 0
    phase: Phase = Phase.PLAN
    hold_elapsed: float = 0.0
    hold_target: np.ndarray | None = None
    tracker: JointPathTracker = field(
        default_factory=lambda: JointPathTracker(
            kp=TRACK_KP, kd=TRACK_KD, speed=PATH_SPEED, arrival_tol=JOINT_ARRIVAL_TOL
        )
    )
    planner: RRTConnectPlanner | None = None
    status_message: str = "初始化"
    last_plan_success: bool = False

    def reset(self, runtime: MuJoCoRuntime | None = None, config: dict | None = None) -> None:
        del config
        if runtime is not None:
            self.runtime = runtime
        self.planner = RRTConnectPlanner(
            self.runtime,
            COLLISION_MODEL,
            step_size=RRT_STEP_SIZE,
            goal_bias=RRT_GOAL_BIAS,
            max_iterations=RRT_MAX_ITERATIONS,
            edge_collision_samples=EDGE_COLLISION_SAMPLES,
        )
        self.tracker = JointPathTracker(
            kp=TRACK_KP,
            kd=TRACK_KD,
            speed=PATH_SPEED,
            arrival_tol=JOINT_ARRIVAL_TOL,
        )
        self.waypoint_index = 0
        self.phase = Phase.PLAN
        self.hold_elapsed = 0.0
        self.hold_target = None
        self.status_message = "重置：准备规划"
        self.runtime.set_joint_positions(HOME_QPOS)
        self.runtime.set_control(HOME_QPOS)
        self.runtime.forward()
        self._begin_planning()
        controller_logger.info("控制器已重置，开始路点序列")

    def step(self, runtime: MuJoCoRuntime, t: float, dt: float) -> dict:
        del t
        if self.phase == Phase.PLAN:
            return self._step_plan(runtime, dt)
        if self.phase == Phase.TRACK:
            return self._step_track(runtime, dt)
        return self._step_hold(runtime, dt)

    def _begin_planning(self) -> None:
        self.phase = Phase.PLAN
        self.hold_elapsed = 0.0
        self.hold_target = None
        waypoint = WAYPOINTS[self.waypoint_index]
        self.status_message = f"规划中：{waypoint.name}"
        controller_logger.info(
            "开始规划：waypoint=%s index=%d",
            waypoint.name,
            self.waypoint_index,
        )

    def _resolve_goal_q(
        self,
        runtime: MuJoCoRuntime,
        waypoint: Waypoint,
        q_start: np.ndarray,
    ) -> np.ndarray | None:
        q_goal = np.asarray(waypoint.joint_goal, dtype=float)
        if self._is_goal_free(runtime, q_goal):
            return q_goal

        target_rot = _quat_wxyz_to_matrix(waypoint.quaternion_wxyz)
        target_pos = np.asarray(waypoint.position, dtype=float)
        return solve_ik_multi_seed(
            runtime,
            target_pos,
            target_rot,
            (q_start, HOME_QPOS, q_start + np.array([0.1, -0.1, 0.1, 0.0, 0.0, 0.0])),
            IK_OPTIONS,
        )

    def _is_goal_free(self, runtime: MuJoCoRuntime, q_goal: np.ndarray) -> bool:
        return not is_configuration_colliding(runtime, q_goal, COLLISION_MODEL)

    def _control_target(self, runtime: MuJoCoRuntime, q_ref: np.ndarray) -> np.ndarray:
        """Map q_ref into the ctrl branch that minimizes |ctrl - qpos| per joint."""
        qpos, _ = runtime.read_joint_state()
        low, high = runtime.joint_limits()
        return actuator_joint_target(
            qpos[:6],
            np.asarray(q_ref, dtype=float)[:6],
            low[:6],
            high[:6],
        )

    def _joint_distance(self, q_from: np.ndarray, q_to: np.ndarray) -> float:
        return float(np.linalg.norm(shortest_joint_delta(q_from, q_to)))

    def _enter_hold(self, runtime: MuJoCoRuntime, hold: np.ndarray) -> dict:
        hold = self._control_target(runtime, hold)
        runtime.set_control(hold)
        self.hold_target = hold.copy()
        self.phase = Phase.HOLD
        self.hold_elapsed = 0.0
        return self._sample(runtime, hold)

    def _step_plan(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        del dt
        q_start = runtime.data.qpos[:6].copy()
        waypoint = WAYPOINTS[self.waypoint_index]
        q_goal = self._resolve_goal_q(runtime, waypoint, q_start)
        if q_goal is None:
            self.last_plan_success = False
            self.status_message = f"IK 失败：{waypoint.name}，保持当前位置"
            planner_logger.warning("IK 失败：waypoint=%s", waypoint.name)
            qpos, _ = runtime.read_joint_state()
            return self._enter_hold(runtime, qpos.copy())

        if self._joint_distance(q_start, q_goal) <= JOINT_ARRIVAL_TOL:
            self.last_plan_success = True
            self.status_message = f"已在路点：{waypoint.name}"
            controller_logger.info("已在路点，跳过规划：%s", waypoint.name)
            return self._enter_hold(runtime, q_start.copy())

        assert self.planner is not None
        path = self.planner.plan(q_start, q_goal)
        if path is None:
            self.last_plan_success = False
            self.status_message = f"RRT 失败：{waypoint.name}，保持当前位置"
            planner_logger.warning("RRT 失败：waypoint=%s", waypoint.name)
            qpos, _ = runtime.read_joint_state()
            return self._enter_hold(runtime, qpos.copy())

        path = snap_path_start(path, q_start)
        raw_nodes = len(path)
        path = shortcut_path(
            runtime,
            path,
            COLLISION_MODEL,
            edge_collision_samples=EDGE_COLLISION_SAMPLES,
        )
        path = unwrap_path(path, q_start)
        dense_path = densify_path(
            runtime,
            path,
            COLLISION_MODEL,
            max_joint_step=PATH_DENSIFY_STEP,
        )
        if dense_path is None:
            self.last_plan_success = False
            self.status_message = f"路径密化碰撞：{waypoint.name}，保持当前位置"
            planner_logger.warning("路径密化碰撞：waypoint=%s", waypoint.name)
            qpos, _ = runtime.read_joint_state()
            return self._enter_hold(runtime, qpos.copy())

        qpos, _ = runtime.read_joint_state()
        low, high = runtime.joint_limits()
        self.tracker.set_path(
            dense_path,
            q_anchor=qpos[:6],
            joint_limits=(low[:6], high[:6]),
        )
        self.last_plan_success = True
        self.phase = Phase.TRACK
        self.status_message = f"跟踪中：{waypoint.name}"
        planner_logger.info(
            "规划完成：waypoint=%s rrt_nodes=%d shortcut_nodes=%d dense_nodes=%d",
            waypoint.name,
            raw_nodes,
            len(path),
            len(dense_path),
        )
        controller_logger.info("进入跟踪阶段：waypoint=%s", waypoint.name)
        return self._step_track(runtime, 0.0)

    def _step_track(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        qpos, _ = runtime.read_joint_state()
        self.tracker.advance(dt)
        target = self._control_target(
            runtime, self.tracker.target_at_progress()
        )
        runtime.set_control(target)
        sample = self._sample(runtime, target)
        if self.tracker.is_complete(qpos):
            self.phase = Phase.HOLD
            self.hold_elapsed = 0.0
            self.hold_target = self._control_target(
                runtime, self.tracker.path[-1]
            )
            waypoint = WAYPOINTS[self.waypoint_index]
            self.status_message = f"到达：{waypoint.name}"
            controller_logger.info("到达路点：%s", waypoint.name)
        return sample

    def _step_hold(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        if self.hold_target is None:
            qpos, _ = runtime.read_joint_state()
            self.hold_target = qpos.copy()
        runtime.set_control(self.hold_target)
        self.hold_elapsed += dt
        if self.hold_elapsed >= HOLD_DURATION:
            next_index = (self.waypoint_index + 1) % len(WAYPOINTS)
            controller_logger.info(
                "Hold 完成，切换路点：%s -> %s",
                WAYPOINTS[self.waypoint_index].name,
                WAYPOINTS[next_index].name,
            )
            self.waypoint_index = next_index
            self._begin_planning()
            return self._step_plan(runtime, dt)
        return self._sample(runtime, self.hold_target)

    def _sample(
        self, runtime: MuJoCoRuntime, target: np.ndarray | None
    ) -> dict:
        if target is None:
            target = runtime.data.qpos[:6].copy()
        waypoint = WAYPOINTS[self.waypoint_index]
        return {
            "controller": self.name,
            "phase": self.phase.value,
            "waypoint": waypoint.name,
            "waypoint_index": self.waypoint_index,
            "status": self.status_message,
            "plan_success": self.last_plan_success,
            "target": target.copy(),
            "control": target.copy(),
        }
