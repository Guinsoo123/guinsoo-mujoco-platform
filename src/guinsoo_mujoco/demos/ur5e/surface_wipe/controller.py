from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import (
    ADMITTANCE,
    APPROACH_DURATION,
    APPROACH_IK_FALLBACK,
    APPROACH_IK_TOLERANCE,
    CONTACT_STANDOFF,
    DESCEND_SPEED,
    DESCEND_STANDOFF_TOL,
    EE_SITE,
    FORCE_SENSOR,
    IK_OPTIONS,
    MAX_CONTACT_FORCE,
    MAX_JOINT_STEP,
    MIN_DESCEND_TIME,
    MIN_NORMAL_OFFSET,
    PREP_QPOS,
    PRE_CONTACT_OFFSET,
    RETRACT_DISTANCE,
    RETRACT_SPEED,
    SURFACE,
    TANGENTIAL_SPEED,
    TORQUE_SENSOR,
    WAVE_GEOM,
    WIPE_IK_SEED,
    WIPE_LENGTH,
    TOOL_BODY_NAMES,
)
from guinsoo_mujoco.demos.ur5e.surface_wipe.workflow import Phase
from guinsoo_mujoco.logging_config import get_logger
from guinsoo_mujoco.operators.admittance import (
    NormalAdmittanceState,
    project_force_on_normal,
)
from guinsoo_mujoco.operators.ik import IkOptions, solve_ik_nearest
from guinsoo_mujoco.operators.path import (
    actuator_joint_target,
    interpolate_joints,
    shortest_joint_delta,
    unwrap_joint_target,
)
from guinsoo_mujoco.runtime import MuJoCoRuntime

controller_logger = get_logger("controller")


@dataclass
class SurfaceWipeController:
    runtime: MuJoCoRuntime
    name: str = "surface_wipe"
    phase: Phase = Phase.APPROACH
    path_s: float = 0.0
    descend_offset: float = 0.0
    descend_elapsed: float = 0.0
    retract_offset: float = 0.0
    approach_alpha: float = 0.0
    approach_q_start: np.ndarray | None = None
    approach_q_goal: np.ndarray | None = None
    admittance_state: NormalAdmittanceState = field(default_factory=NormalAdmittanceState)
    status_message: str = "初始化"
    last_target: np.ndarray | None = None
    last_force_normal: float = 0.0
    last_force_normal_raw: float = 0.0
    last_admittance_dn: float = 0.0
    last_tool_contact: bool = False

    def reset(self, runtime: MuJoCoRuntime | None = None, config: dict | None = None) -> None:
        del config
        if runtime is not None:
            self.runtime = runtime
        self.phase = Phase.APPROACH
        self.path_s = 0.0
        self.descend_offset = PRE_CONTACT_OFFSET
        self.descend_elapsed = 0.0
        self.retract_offset = 0.0
        self.approach_alpha = 0.0
        self.approach_q_start = None
        self.approach_q_goal = None
        self.admittance_state = ADMITTANCE.reset()
        self.status_message = "重置：准备接近曲面"
        self.last_target = None
        self.last_force_normal = 0.0
        self.last_force_normal_raw = 0.0
        self.last_admittance_dn = 0.0
        self.last_tool_contact = False
        self.runtime.set_joint_positions(PREP_QPOS)
        self.runtime.set_control(PREP_QPOS)
        self.runtime.forward()
        self.approach_q_start = PREP_QPOS.copy()
        controller_logger.info("曲面擦拭控制器已重置")

    def step(self, runtime: MuJoCoRuntime, t: float, dt: float) -> dict:
        del t
        if self.phase == Phase.APPROACH:
            sample = self._step_approach(runtime, dt)
        elif self.phase == Phase.DESCEND:
            sample = self._step_descend(runtime, dt)
        elif self.phase == Phase.FOLLOW:
            sample = self._step_follow(runtime, dt)
        elif self.phase == Phase.RETRACT:
            sample = self._step_retract(runtime, dt)
        else:
            sample = self._hold_current(runtime)

        sample["phase"] = self.phase.value
        sample["status"] = self.status_message
        sample["path_s"] = self.path_s
        sample["force_normal"] = self.last_force_normal
        sample["force_normal_raw"] = self.last_force_normal_raw
        sample["force_des"] = ADMITTANCE.force_des
        sample["admittance_dn"] = self.last_admittance_dn
        sample["tool_contact"] = float(self.last_tool_contact)
        sample["approach_alpha"] = self.approach_alpha
        return sample

    def _step_approach(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        if self.approach_q_goal is None:
            target_pose = self._desired_pose(
                s=0.0,
                normal_offset=PRE_CONTACT_OFFSET,
            )
            q_goal = self._solve_pose(
                runtime,
                target_pose,
                APPROACH_IK_TOLERANCE,
                prefer_nearest=True,
            )
            if q_goal is None:
                q_goal = self._solve_pose(
                    runtime,
                    target_pose,
                    APPROACH_IK_FALLBACK,
                    q_seed=WIPE_IK_SEED,
                )
            if q_goal is None:
                self.status_message = "接近失败：IK 无解"
                q_hold, _ = runtime.read_joint_state()
                self._apply_joint_target(runtime, q_hold)
                return self._sample(runtime, q_hold)
            q_start = (
                self.approach_q_start
                if self.approach_q_start is not None
                else runtime.read_joint_state()[0]
            )
            self.approach_q_goal = unwrap_joint_target(q_start, q_goal)

        duration = max(APPROACH_DURATION, dt)
        self.approach_alpha = min(1.0, self.approach_alpha + dt / duration)
        q_start = (
            self.approach_q_start
            if self.approach_q_start is not None
            else runtime.read_joint_state()[0]
        )
        q_target = interpolate_joints(q_start, self.approach_q_goal, self.approach_alpha)
        self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)

        if self.approach_alpha >= 1.0 - 1e-6:
            self.phase = Phase.DESCEND
            self.descend_offset = PRE_CONTACT_OFFSET
            self.descend_elapsed = 0.0
            self.status_message = "开始沿法向下降"
            controller_logger.info("进入下降阶段")
        else:
            self.status_message = f"接近中：blend={self.approach_alpha:.2f}"
        return self._sample(runtime, q_target)

    def _step_descend(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        self.descend_elapsed += dt
        force_normal = self._read_normal_force(runtime, s=0.0)
        tool_contact = self._tool_contacts_wave(runtime)
        self.descend_offset = max(
            MIN_NORMAL_OFFSET,
            self.descend_offset - DESCEND_SPEED * dt,
        )
        target_pose = self._desired_pose(
            s=0.0,
            normal_offset=self.descend_offset,
        )
        q_target = self._solve_pose(
            runtime,
            target_pose,
            IK_OPTIONS,
            prefer_nearest=True,
        )
        if q_target is None:
            self.status_message = "下降失败：IK 无解"
            q_target = runtime.read_joint_state()[0]
        else:
            self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)

        signed_standoff = self._signed_standoff(runtime, s=0.0)
        contact_ready = (
            tool_contact
            and signed_standoff <= CONTACT_STANDOFF + DESCEND_STANDOFF_TOL
        )
        if self.descend_elapsed >= MIN_DESCEND_TIME and contact_ready:
            self.phase = Phase.FOLLOW
            self.path_s = 0.0
            self.admittance_state = ADMITTANCE.reset()
            self.status_message = "接触建立，开始擦拭"
            controller_logger.info("进入跟随阶段：waypoint=s=0")
        else:
            self.status_message = (
                f"下降中：offset={self.descend_offset:.3f} Fn={force_normal:.1f}N"
            )
        return self._sample(runtime, q_target)

    def _step_follow(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        force_normal = self._read_normal_force(runtime, s=self.path_s)
        self.admittance_state, d_n = ADMITTANCE.step(
            self.admittance_state,
            force_normal,
            dt,
        )
        self.last_admittance_dn = d_n
        normal_offset = max(MIN_NORMAL_OFFSET, CONTACT_STANDOFF + d_n)
        target_pose = self._desired_pose(
            s=self.path_s,
            normal_offset=normal_offset,
        )
        q_target = self._solve_pose(
            runtime,
            target_pose,
            IK_OPTIONS,
            prefer_nearest=True,
        )
        if q_target is not None:
            self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)
            self.path_s = min(WIPE_LENGTH, self.path_s + TANGENTIAL_SPEED * dt)
        else:
            self.status_message = "跟随中：IK 失败，暂停切向推进"
            q_target = runtime.read_joint_state()[0]

        if self.path_s >= WIPE_LENGTH - 1e-6:
            self.phase = Phase.RETRACT
            self.retract_offset = 0.0
            self.status_message = "擦拭完成，开始抬离"
            controller_logger.info("进入抬离阶段")
        else:
            self.status_message = (
                f"擦拭中：s={self.path_s:.3f}/{WIPE_LENGTH:.3f} "
                f"Fn={force_normal:.1f}/{ADMITTANCE.force_des:.1f}N"
            )
        return self._sample(runtime, q_target)

    def _step_retract(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        self.retract_offset = min(
            RETRACT_DISTANCE,
            self.retract_offset + RETRACT_SPEED * dt,
        )
        target_pose = self._desired_pose(
            s=self.path_s,
            normal_offset=CONTACT_STANDOFF + self.retract_offset,
        )
        q_target = self._solve_pose(
            runtime,
            target_pose,
            IK_OPTIONS,
            prefer_nearest=True,
        )
        if q_target is None:
            q_target = runtime.read_joint_state()[0]
        else:
            self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)
        if self.retract_offset >= RETRACT_DISTANCE - 1e-6:
            self.phase = Phase.DONE
            self.status_message = "任务完成"
            controller_logger.info("擦拭任务完成")
        return self._sample(runtime, q_target)

    def _hold_current(self, runtime: MuJoCoRuntime) -> dict:
        qpos, _ = runtime.read_joint_state()
        self._apply_joint_target(runtime, qpos)
        return self._sample(runtime, qpos)

    def _desired_pose(
        self,
        *,
        s: float,
        normal_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        p_ref = SURFACE.position(s)
        n_hat = SURFACE.normal(s)
        pos = p_ref + normal_offset * n_hat
        rot = SURFACE.orientation(s)
        return pos, rot

    def _read_normal_force(self, runtime: MuJoCoRuntime, *, s: float) -> float:
        try:
            wrench = runtime.read_site_wrench(FORCE_SENSOR, TORQUE_SENSOR)
            force_site = wrench[:3]
        except ValueError:
            force_site = np.zeros(3, dtype=float)
        _, site_rot = runtime.site_pose(EE_SITE)
        force_world = site_rot @ force_site
        n_hat = SURFACE.normal(s)
        force_normal = abs(project_force_on_normal(force_world, n_hat))
        self.last_force_normal_raw = float(force_normal)
        self.last_force_normal = float(
            np.clip(force_normal, 0.0, MAX_CONTACT_FORCE)
        )
        return self.last_force_normal_raw

    def _tool_contacts_wave(self, runtime: MuJoCoRuntime) -> bool:
        try:
            wave_id = runtime.geom_id(WAVE_GEOM)
        except ValueError:
            self.last_tool_contact = False
            return False
        tool_body_ids = {
            runtime.mujoco.mj_name2id(
                runtime.model, runtime.mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            for body_name in TOOL_BODY_NAMES
        }
        for contact_index in range(int(runtime.data.ncon)):
            contact = runtime.data.contact[contact_index]
            if float(contact.dist) > 0.001:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if wave_id not in (geom1, geom2):
                continue
            other = geom2 if geom1 == wave_id else geom1
            body_id = int(runtime.model.geom_bodyid[other])
            if body_id in tool_body_ids:
                self.last_tool_contact = True
                return True
        self.last_tool_contact = False
        return False

    def _solve_pose(
        self,
        runtime: MuJoCoRuntime,
        target_pose: tuple[np.ndarray, np.ndarray],
        options: IkOptions,
        *,
        prefer_nearest: bool = True,
        q_seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        pos, rot = target_pose
        q_current, _ = runtime.read_joint_state()
        del prefer_nearest
        q = solve_ik_nearest(
            runtime,
            pos,
            rot,
            q_seed if q_seed is not None else q_current,
            options,
        )
        if q is None:
            return None

        q_saved = runtime.data.qpos.copy()
        runtime.forward(q)
        achieved_pos, achieved_rot = runtime.site_pose(EE_SITE)
        pos_err = float(np.linalg.norm(achieved_pos - pos))
        n_hat = rot[:, 2]
        orient_err = float(
            np.arccos(np.clip(float(np.dot(achieved_rot[:, 2], n_hat)), -1.0, 1.0))
        )
        runtime.set_joint_positions(q_saved)
        runtime.forward()

        if pos_err > options.position_tol * 2.5:
            return None
        if not options.position_only and orient_err > options.orientation_tol * 2.0:
            return None
        return unwrap_joint_target(q_current, q)

    def _apply_joint_target(
        self,
        runtime: MuJoCoRuntime,
        q_target: np.ndarray,
        *,
        max_step: float | None = None,
    ) -> None:
        qpos, _ = runtime.read_joint_state()
        q_unwrapped = unwrap_joint_target(qpos, q_target)
        if max_step is not None:
            delta = shortest_joint_delta(qpos, q_unwrapped)
            delta = np.clip(delta, -max_step, max_step)
            q_unwrapped = qpos + delta
        low, high = runtime.joint_limits()
        ctrl = actuator_joint_target(qpos, q_unwrapped, low, high)
        runtime.set_control(ctrl)
        self.last_target = q_unwrapped.copy()

    def _signed_standoff(self, runtime: MuJoCoRuntime, *, s: float) -> float:
        pos, _ = runtime.site_pose(EE_SITE)
        p_ref = SURFACE.position(s)
        n_hat = SURFACE.normal(s)
        return float(np.dot(pos - p_ref, n_hat))

    def _sample(self, runtime: MuJoCoRuntime, q_target: np.ndarray) -> dict:
        try:
            wrench = runtime.read_site_wrench(FORCE_SENSOR, TORQUE_SENSOR)
        except ValueError:
            wrench = np.zeros(6, dtype=float)
        pos, rot = runtime.site_pose(EE_SITE)
        p_ref = SURFACE.position(self.path_s)
        n_hat = SURFACE.normal(self.path_s)
        rot_des = SURFACE.orientation(self.path_s)
        orient_err = float(
            np.arccos(np.clip(float(np.dot(rot[:, 2], rot_des[:, 2])), -1.0, 1.0))
        )
        signed_standoff = self._signed_standoff(runtime, s=self.path_s)
        return {
            "target": q_target.copy(),
            "control": runtime.data.ctrl.copy(),
            "wrench_tool": wrench.astype(float),
            "ee_pos_error": float(np.linalg.norm(pos - p_ref)),
            "ee_orient_error": orient_err,
            "ee_signed_standoff": signed_standoff,
        }
