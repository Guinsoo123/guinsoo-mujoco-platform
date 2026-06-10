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
