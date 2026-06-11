from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guinsoo_mujoco.logging_config import get_logger, install_mujoco_warning_handler

logger = get_logger("mujoco")


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
        install_mujoco_warning_handler()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        logger.info("MuJoCo 模型已加载：%s (nq=%d nu=%d)", self.model_path, self.model.nq, self.model.nu)

    def ensure_offscreen_size(self, width: int, height: int) -> None:
        """Grow MuJoCo's offscreen framebuffer to fit the requested viewport."""
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)

    def step(self) -> RuntimeState:
        try:
            self.mujoco.mj_step(self.model, self.data)
        except Exception as exc:
            logger.warning("mj_step 异常：%s", exc)
            raise
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

    def read_sensor(self, name: str) -> np.ndarray:
        sensor_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise ValueError(f"unknown sensor: {name}")
        start = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[start : start + dim].copy()

    def read_site_wrench(
        self,
        force_sensor: str,
        torque_sensor: str,
    ) -> np.ndarray:
        force = self.read_sensor(force_sensor)
        torque = self.read_sensor(torque_sensor)
        return np.concatenate([force, torque]).astype(float)

    def set_control(self, control: np.ndarray) -> None:
        value = np.asarray(control, dtype=float)
        if value.size > self.data.ctrl.size:
            raise ValueError("control vector is larger than runtime ctrl buffer")
        self.data.ctrl[: value.size] = value

    def forward(self, qpos: np.ndarray | None = None) -> None:
        if qpos is not None:
            self.set_joint_positions(qpos)
        self.mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, qpos: np.ndarray) -> None:
        value = np.asarray(qpos, dtype=float)
        if value.size > self.data.qpos.size:
            raise ValueError("qpos vector is larger than runtime qpos buffer")
        self.data.qpos[: value.size] = value

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self.model.jnt_range.size == 0:
            n = self.model.nq
            return np.full(n, -np.pi), np.full(n, np.pi)
        low = self.model.jnt_range[:, 0].copy()
        high = self.model.jnt_range[:, 1].copy()
        return low, high

    def site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise ValueError(f"unknown site: {site_name}")
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        return pos, mat

    def site_jacobian(self, site_name: str) -> np.ndarray:
        site_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise ValueError(f"unknown site: {site_name}")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr])
        return jac[:, : self.model.nq]

    def geom_id(self, geom_name: str) -> int:
        geom_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if geom_id < 0:
            raise ValueError(f"unknown geom: {geom_name}")
        return geom_id

    def geom_distance_ids(
        self,
        geom1_id: int,
        geom2_id: int,
        *,
        distmax: float = 10.0,
    ) -> float:
        fromto = np.zeros(6, dtype=np.float64)
        return float(
            self.mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(geom1_id),
                int(geom2_id),
                float(distmax),
                fromto,
            )
        )

    def geom_distance(
        self,
        geom1_name: str,
        geom2_name: str,
        *,
        distmax: float = 10.0,
    ) -> float:
        return self.geom_distance_ids(
            self.geom_id(geom1_name),
            self.geom_id(geom2_name),
            distmax=distmax,
        )

    def has_geom_contact(
        self,
        geom1_name: str,
        geom2_name: str,
        *,
        max_distance: float = 0.002,
    ) -> bool:
        geom1_id = self.geom_id(geom1_name)
        geom2_id = self.geom_id(geom2_name)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if {geom1, geom2} != {geom1_id, geom2_id}:
                continue
            if float(contact.dist) <= max_distance:
                return True
        return False

    def has_contacts(self) -> bool:
        return int(self.data.ncon) > 0

    def check_obstacle_collision(
        self,
        obstacle_geom_names: tuple[str, ...],
        arm_geom_names: tuple[str, ...],
        *,
        margin: float = 0.02,
    ) -> bool:
        for obstacle in obstacle_geom_names:
            for arm_geom in arm_geom_names:
                try:
                    distance = self.geom_distance(obstacle, arm_geom, distmax=1.0)
                except ValueError:
                    continue
                if distance < margin:
                    return True
        if self.has_contacts():
            return True
        return False
