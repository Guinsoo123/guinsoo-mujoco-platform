from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RuntimeState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


class MuJoCoRuntime:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is not installed in the active environment") from exc

        self.mujoco = mujoco
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

    def ensure_offscreen_size(self, width: int, height: int) -> None:
        """Grow MuJoCo's offscreen framebuffer to fit the requested viewport."""
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)

    def step(self) -> RuntimeState:
        self.mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self) -> RuntimeState:
        return RuntimeState(
            time=float(self.data.time),
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
        )

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos.copy(), self.data.qvel.copy()

    def actuator_names(self) -> list[str]:
        names: list[str] = []
        for index in range(self.model.nu):
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, index
            )
            names.append(name or f"actuator_{index}")
        return names

    def joint_names(self) -> list[str]:
        if self.model.nu > 0:
            return self.actuator_names()
        names: list[str] = []
        for index in range(self.model.njnt):
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, index
            )
            names.append(name or f"joint_{index}")
        return names

    def read_telemetry(self) -> dict[str, np.ndarray | float]:
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "actuator_force": self.data.actuator_force.copy(),
            "qfrc_actuator": self.data.qfrc_actuator.copy(),
        }

    def set_control(self, control: np.ndarray) -> None:
        value = np.asarray(control, dtype=float)
        if value.size > self.data.ctrl.size:
            raise ValueError("control vector is larger than runtime ctrl buffer")
        self.data.ctrl[: value.size] = value
