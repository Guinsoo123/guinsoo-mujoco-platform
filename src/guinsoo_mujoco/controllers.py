from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RuntimeLike(Protocol):
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        ...

    def set_control(self, control: np.ndarray) -> None:
        ...


class Controller(Protocol):
    name: str

    def reset(self, runtime: RuntimeLike, config: dict | None = None) -> None:
        ...

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        ...


@dataclass
class JointPositionController:
    name: str
    target: list[float]
    kp: float = 20.0
    kd: float = 2.0

    def reset(self, runtime: RuntimeLike, config: dict | None = None) -> None:
        if config and "target" in config:
            self.target = list(config["target"])

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        qpos, qvel = runtime.read_joint_state()
        target = np.asarray(self.target, dtype=float)
        if target.shape != qpos.shape:
            raise ValueError(
                f"target shape {target.shape} does not match qpos shape {qpos.shape}"
            )
        control = self.kp * (target - qpos) - self.kd * qvel
        runtime.set_control(control)
        return {
            "controller": self.name,
            "time": float(t),
            "target": target.copy(),
            "control": control.copy(),
        }


@dataclass
class ReachMotionController:
    name: str
    home_target: list[float]
    amplitude: float = 0.35
    frequency: float = 0.4
    joint_index: int = 2
    kp: float = 20.0
    kd: float = 2.0

    def reset(self, runtime: RuntimeLike, config: dict | None = None) -> None:
        if config and "home_target" in config:
            self.home_target = list(config["home_target"])

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        qpos, qvel = runtime.read_joint_state()
        target = np.asarray(self.home_target, dtype=float).copy()
        if target.shape != qpos.shape:
            raise ValueError(
                f"home_target shape {target.shape} does not match qpos shape {qpos.shape}"
            )
        index = self.joint_index % target.size
        target[index] += self.amplitude * np.sin(2.0 * np.pi * self.frequency * t)
        control = self.kp * (target - qpos) - self.kd * qvel
        runtime.set_control(control)
        return {
            "controller": self.name,
            "time": float(t),
            "target": target.copy(),
            "control": control.copy(),
        }


def default_joint_target(dof: int) -> list[float]:
    if dof == 6:
        return [0.0, -1.2, 1.8, -0.8, -1.57, 0.0]
    return [0.0] * dof


def create_demo_controller(demo: str, dof: int) -> Controller:
    home = default_joint_target(dof)
    if demo == "joint_position":
        return JointPositionController(name="joint_position", target=home)
    if demo == "ik_reach":
        return ReachMotionController(name="ik_reach", home_target=home)
    if demo == "preview_motion":
        return JointPositionController(
            name="preview_motion",
            target=home,
            kp=12.0,
            kd=1.5,
        )
    raise ValueError(f"unknown demo: {demo}")
