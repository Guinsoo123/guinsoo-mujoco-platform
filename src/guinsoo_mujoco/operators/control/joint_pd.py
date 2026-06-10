from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guinsoo_mujoco.controllers import RuntimeLike


@dataclass
class JointPositionController:
    name: str
    target: list[float]
    kp: float = 20.0
    kd: float = 2.0

    def reset(self, runtime: RuntimeLike, config: dict | None = None) -> None:
        del runtime
        if config and "target" in config:
            self.target = list(config["target"])

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        del dt
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
        del runtime
        if config and "home_target" in config:
            self.home_target = list(config["home_target"])

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        del dt
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
