from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.collision import (
    is_configuration_colliding,
    is_edge_colliding,
)
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import (
    COLLISION_MARGIN,
    EDGE_COLLISION_SAMPLES,
    OBSTACLE_GEOMS,
    RRT_GOAL_BIAS,
    RRT_MAX_ITERATIONS,
    RRT_STEP_SIZE,
)
from guinsoo_mujoco.logging_config import get_logger
from guinsoo_mujoco.runtime import MuJoCoRuntime

logger = get_logger("planner")


@dataclass
class RRTNode:
    q: np.ndarray
    parent: int = -1


@dataclass
class RRTConnectPlanner:
    runtime: MuJoCoRuntime
    step_size: float = RRT_STEP_SIZE
    goal_bias: float = RRT_GOAL_BIAS
    max_iterations: int = RRT_MAX_ITERATIONS
    obstacle_geom_names: tuple[str, ...] = OBSTACLE_GEOMS
    collision_margin: float = COLLISION_MARGIN

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray] | None:
        low, high = self.runtime.joint_limits()
        q_start = np.clip(np.asarray(q_start, dtype=float), low, high)
        q_goal = np.clip(np.asarray(q_goal, dtype=float), low, high)
        if not (self._is_free(q_start) and self._is_free(q_goal)):
            logger.warning("RRT 起终点构型碰撞，规划取消")
            return None
        if np.linalg.norm(q_goal - q_start) < self.step_size:
            logger.info("RRT 起终点距离很近，直接连接")
            return [q_start.copy(), q_goal.copy()]

        tree_a = [RRTNode(q_start.copy())]
        tree_b = [RRTNode(q_goal.copy())]

        for iteration in range(self.max_iterations):
            if iteration % 2 == 0:
                path = self._extend(tree_a, tree_b, q_goal, low, high)
                if path is not None:
                    logger.info(
                        "RRT-Connect 成功：nodes=%d iterations=%d",
                        len(path),
                        iteration + 1,
                    )
                    return path
            else:
                path = self._extend(tree_b, tree_a, q_start, low, high)
                if path is not None:
                    logger.info(
                        "RRT-Connect 成功：nodes=%d iterations=%d",
                        len(path),
                        iteration + 1,
                    )
                    return list(reversed(path))

        logger.warning(
            "RRT-Connect 失败：max_iterations=%d",
            self.max_iterations,
        )
        return None

    def _is_free(self, q: np.ndarray) -> bool:
        return not is_configuration_colliding(
            self.runtime,
            q,
            self.obstacle_geom_names,
            margin=self.collision_margin,
        )

    def _is_edge_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        if not self._is_free(q_to):
            return False
        return not is_edge_colliding(
            self.runtime,
            q_from,
            q_to,
            self.obstacle_geom_names,
            margin=self.collision_margin,
            samples=EDGE_COLLISION_SAMPLES,
        )

    def _sample(self, q_target: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if np.random.random() < self.goal_bias:
            return q_target.copy()
        return low + np.random.random(low.shape) * (high - low)

    def _nearest(self, tree: list[RRTNode], q: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node.q - q)) for node in tree]
        return int(np.argmin(distances))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        direction = q_to - q_from
        distance = float(np.linalg.norm(direction))
        if distance <= self.step_size:
            return q_to.copy()
        return q_from + direction / distance * self.step_size

    def _extend(
        self,
        tree_from: list[RRTNode],
        tree_to: list[RRTNode],
        q_target: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> list[np.ndarray] | None:
        q_rand = self._sample(q_target, low, high)
        nearest_index = self._nearest(tree_from, q_rand)
        q_near = tree_from[nearest_index].q
        q_new = np.clip(self._steer(q_near, q_rand), low, high)
        if not self._is_edge_free(q_near, q_new):
            return None
        tree_from.append(RRTNode(q_new, parent=nearest_index))

        nearest_other = self._nearest(tree_to, q_new)
        q_other = tree_to[nearest_other].q
        q_connect = np.clip(self._steer(q_new, q_other), low, high)
        if np.linalg.norm(q_connect - q_other) < self.step_size and self._is_edge_free(
            q_new, q_connect
        ):
            connect_index = nearest_other
            if np.linalg.norm(q_connect - q_new) > 1e-6:
                tree_to.append(RRTNode(q_connect, parent=connect_index))
                connect_index = len(tree_to) - 1
            return self._extract_path(tree_from, len(tree_from) - 1, tree_to, connect_index)
        return None

    def _extract_path(
        self,
        tree_a: list[RRTNode],
        index_a: int,
        tree_b: list[RRTNode],
        index_b: int,
    ) -> list[np.ndarray]:
        path_a: list[np.ndarray] = []
        current = index_a
        while current >= 0:
            path_a.append(tree_a[current].q.copy())
            current = tree_a[current].parent
        path_a.reverse()

        path_b: list[np.ndarray] = []
        current = index_b
        while current >= 0:
            path_b.append(tree_b[current].q.copy())
            current = tree_b[current].parent

        return path_a + path_b
