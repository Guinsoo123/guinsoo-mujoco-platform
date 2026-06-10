from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guinsoo_mujoco.operators.path.joint_unwrap import (
    interpolate_joints,
    shortest_joint_delta,
    unwrap_joint_target,
)


@dataclass
class JointPathTracker:
    path: list[np.ndarray] = field(default_factory=list)
    segment_lengths: list[float] = field(default_factory=list)
    total_length: float = 0.0
    progress: float = 0.0
    kp: float = 20.0
    kd: float = 2.0
    speed: float = 1.5
    arrival_tol: float = 0.05

    def set_path(self, path: list[np.ndarray]) -> None:
        if len(path) < 2:
            self.path = [np.asarray(path[0], dtype=float).copy()] if path else []
            self.segment_lengths = []
            self.total_length = 0.0
            self.progress = 0.0
            return
        raw = [np.asarray(node, dtype=float).copy() for node in path]
        self.path = [raw[0].copy()]
        for index in range(1, len(raw)):
            self.path.append(unwrap_joint_target(self.path[-1], raw[index]))
        self.segment_lengths = []
        self.total_length = 0.0
        for index in range(len(self.path) - 1):
            delta = self.path[index + 1] - self.path[index]
            length = float(np.linalg.norm(delta))
            self.segment_lengths.append(length)
            self.total_length += length
        self.progress = 0.0

    def advance(self, dt: float) -> None:
        if self.total_length <= 0.0:
            return
        self.progress = min(self.total_length, self.progress + self.speed * dt)

    def target_at_progress(self) -> np.ndarray:
        if not self.path:
            raise RuntimeError("path tracker has no path")
        if len(self.path) == 1 or self.total_length <= 0.0:
            return self.path[-1].copy()
        remaining = self.progress
        for index, length in enumerate(self.segment_lengths):
            if remaining <= length:
                alpha = remaining / length if length > 0 else 1.0
                return interpolate_joints(self.path[index], self.path[index + 1], alpha)
            remaining -= length
        return self.path[-1].copy()

    def is_complete(self, qpos: np.ndarray) -> bool:
        if not self.path:
            return True
        target = self.path[-1]
        distance = float(np.linalg.norm(shortest_joint_delta(qpos, target)))
        if distance <= self.arrival_tol:
            return True
        return (
            self.progress >= self.total_length - 1e-6
            and distance <= self.arrival_tol * 2.0
        )

    def control(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = self.target_at_progress()
        control = self.kp * (target - qpos) - self.kd * qvel
        return target, control
