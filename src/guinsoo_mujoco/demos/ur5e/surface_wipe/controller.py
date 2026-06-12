from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import (
    ADMITTANCE,
    CONTACT_STANDOFF,
    CONTACT_SETTLE_QVEL,
    CONTACT_SETTLE_TIME,
    DESCEND_SPEED,
    DESCEND_STANDOFF_TOL,
    EE_SITE,
    FORCE_SENSOR,
    IK_OPTIONS,
    MAX_CONTACT_FORCE,
    FOLLOW_IK_DECIMATION,
    FOLLOW_MAX_JOINT_STEP,
    JOINT_TARGET_SMOOTHING,
    MAX_JOINT_STEP,
    MIN_DESCEND_TIME,
    MIN_NORMAL_OFFSET,
    PRE_CONTACT_OFFSET,
    RETRACT_DISTANCE,
    RETRACT_SPEED,
    START_QPOS,
    SURFACE,
    TANGENTIAL_SPEED,
    TORQUE_SENSOR,
    WAVE_GEOM,
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
    shortest_joint_delta,
    unwrap_joint_target,
)
from guinsoo_mujoco.runtime import MuJoCoRuntime

controller_logger = get_logger("controller")


@dataclass
class SurfaceWipeController:
    runtime: MuJoCoRuntime
    name: str = "surface_wipe"
    phase: Phase = Phase.DESCEND
    path_s: float = 0.0
    descend_offset: float = 0.0
    descend_elapsed: float = 0.0
    retract_offset: float = 0.0
    approach_alpha: float = 1.0
    admittance_state: NormalAdmittanceState = field(default_factory=NormalAdmittanceState)
    status_message: str = "初始化"
    last_target: np.ndarray | None = None
    last_force_normal: float = 0.0
    last_force_normal_raw: float = 0.0
    last_admittance_dn: float = 0.0
    last_tool_contact: bool = False
    smoothed_q_target: np.ndarray | None = None
    hold_q_target: np.ndarray | None = None
    contact_settle_q_target: np.ndarray | None = None
    contact_settle_elapsed: float = 0.0
    follow_ik_cache: np.ndarray | None = None
    follow_ik_step: int = 0
    last_normal_offset: float = 0.0

    def reset(self, runtime: MuJoCoRuntime | None = None, config: dict | None = None) -> None:
        del config
        if runtime is not None:
            self.runtime = runtime
        self.phase = Phase.DESCEND
        self.path_s = 0.0
        self.descend_offset = PRE_CONTACT_OFFSET
        self.descend_elapsed = 0.0
        self.retract_offset = 0.0
        self.approach_alpha = 1.0
        self.admittance_state = ADMITTANCE.reset()
        self.status_message = "重置：路径起点待命，准备下降"
        self.last_target = None
        self.last_force_normal = 0.0
        self.last_force_normal_raw = 0.0
        self.last_admittance_dn = 0.0
        self.last_tool_contact = False
        self.smoothed_q_target = None
        self.hold_q_target = None
        self.contact_settle_q_target = None
        self.contact_settle_elapsed = 0.0
        self.follow_ik_cache = None
        self.follow_ik_step = 0
        self.last_normal_offset = PRE_CONTACT_OFFSET
        q_start = unwrap_joint_target(
            self.runtime.read_joint_state()[0],
            START_QPOS,
        )
        self.runtime.set_joint_positions(q_start)
        self.runtime.set_control(q_start)
        self.runtime.forward()
        self.last_target = q_start.copy()
        controller_logger.info("曲面擦拭控制器已重置")

    def step(self, runtime: MuJoCoRuntime, t: float, dt: float) -> dict:
        del t
        if self.phase == Phase.DESCEND:
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

    def _step_descend(self, runtime: MuJoCoRuntime, dt: float) -> dict:
        self.descend_elapsed += dt
        force_normal = self._read_normal_force(runtime, s=0.0)
        tool_contact = self._tool_contacts_wave(runtime)
        qpos, qvel = runtime.read_joint_state()
        if self.contact_settle_q_target is None:
            self.descend_offset = max(
                MIN_NORMAL_OFFSET,
                self.descend_offset - DESCEND_SPEED * dt,
            )
        target_pose = self._desired_pose(
            s=0.0,
            normal_offset=self.descend_offset,
        )
        if self.contact_settle_q_target is not None:
            q_target = self.contact_settle_q_target
            self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)
        else:
            q_target = self._solve_pose(
                runtime,
                target_pose,
                IK_OPTIONS,
                prefer_nearest=True,
            )
            if q_target is None:
                self.status_message = "下降失败：IK 无解"
                q_target = qpos
            else:
                self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)

        signed_standoff = self._signed_standoff(runtime, s=0.0)
        contact_ready = (
            tool_contact
            and signed_standoff <= CONTACT_STANDOFF + DESCEND_STANDOFF_TOL
        )
        settling = self.contact_settle_q_target is not None
        if self.descend_elapsed >= MIN_DESCEND_TIME and (contact_ready or settling):
            if self.contact_settle_q_target is None:
                settle_pose = self._desired_pose(
                    s=0.0,
                    normal_offset=CONTACT_STANDOFF,
                )
                settle_target = self._solve_pose(
                    runtime,
                    settle_pose,
                    IK_OPTIONS,
                    prefer_nearest=True,
                )
                if settle_target is None:
                    settle_target = q_target
                self.contact_settle_q_target = unwrap_joint_target(qpos, settle_target)
                self.contact_settle_elapsed = 0.0
            q_target = self.contact_settle_q_target
            self._apply_joint_target(runtime, q_target, max_step=MAX_JOINT_STEP)
            self.contact_settle_elapsed += dt
            qvel_max = float(np.max(np.abs(qvel)))
            settled = (
                self.contact_settle_elapsed >= CONTACT_SETTLE_TIME
                and qvel_max <= CONTACT_SETTLE_QVEL
            )
        elif not settling:
            self.contact_settle_q_target = None
            self.contact_settle_elapsed = 0.0
            qvel_max = float(np.max(np.abs(qvel)))
            settled = False
        else:
            qvel_max = float(np.max(np.abs(qvel)))
            settled = False

        if settled:
            self.phase = Phase.FOLLOW
            self.path_s = 0.0
            self.smoothed_q_target = None
            self.contact_settle_q_target = None
            self.contact_settle_elapsed = 0.0
            self.follow_ik_cache = None
            self.follow_ik_step = 0
            self.admittance_state = ADMITTANCE.reset(initial_force=ADMITTANCE.force_des)
            self.status_message = "接触建立，开始擦拭"
            controller_logger.info("进入跟随阶段：waypoint=s=0")
        elif self.contact_settle_q_target is not None:
            self.status_message = (
                "接触稳定中："
                f"hold={self.contact_settle_elapsed:.2f}/{CONTACT_SETTLE_TIME:.2f}s "
                f"qvel={qvel_max:.3f}rad/s"
            )
        else:
            self.status_message = (
                f"下降中：offset={self.descend_offset:.3f} Fn={force_normal:.1f}N"
            )
        self.last_normal_offset = self.descend_offset
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
        self.follow_ik_step += 1
        refresh_ik = (
            self.follow_ik_cache is None
            or self.follow_ik_step % max(FOLLOW_IK_DECIMATION, 1) == 0
        )
        q_target = self.follow_ik_cache
        if refresh_ik:
            solved = self._solve_pose(
                runtime,
                target_pose,
                IK_OPTIONS,
                prefer_nearest=True,
            )
            if solved is not None:
                self.follow_ik_cache = solved
                q_target = solved
        if q_target is not None:
            self._apply_joint_target(
                runtime,
                q_target,
                max_step=FOLLOW_MAX_JOINT_STEP,
                smooth=True,
            )
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
        self.last_normal_offset = normal_offset
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
            self._enter_done_hold(runtime, q_target)
        sample_target = (
            self.hold_q_target if self.hold_q_target is not None else q_target
        )
        self.last_normal_offset = CONTACT_STANDOFF + self.retract_offset
        return self._sample(runtime, sample_target)

    def _enter_done_hold(self, runtime: MuJoCoRuntime, q_target: np.ndarray) -> None:
        del q_target
        qpos, _ = runtime.read_joint_state()
        self.hold_q_target = qpos.copy()
        low, high = runtime.joint_limits()
        ctrl = actuator_joint_target(qpos, self.hold_q_target, low, high)
        runtime.set_control(ctrl)
        self.last_target = self.hold_q_target.copy()
        self.smoothed_q_target = None
        self.phase = Phase.DONE
        self.status_message = "任务完成，保持静止"
        controller_logger.info("擦拭任务完成，锁定关节目标")

    def _hold_current(self, runtime: MuJoCoRuntime) -> dict:
        if self.hold_q_target is None:
            qpos, _ = runtime.read_joint_state()
            self.hold_q_target = qpos.copy()
        qpos, _ = runtime.read_joint_state()
        low, high = runtime.joint_limits()
        ctrl = actuator_joint_target(qpos, self.hold_q_target, low, high)
        runtime.set_control(ctrl)
        return self._sample(runtime, self.hold_q_target)

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
        q_saved = runtime.data.qpos.copy()
        qvel_saved = runtime.data.qvel.copy()
        q = solve_ik_nearest(
            runtime,
            pos,
            rot,
            q_seed if q_seed is not None else q_current,
            options,
        )
        if q is None:
            runtime.set_joint_positions(q_saved)
            runtime.data.qvel[: qvel_saved.size] = qvel_saved
            runtime.forward()
            return None

        runtime.forward(q)
        achieved_pos, achieved_rot = runtime.site_pose(EE_SITE)
        pos_err = float(np.linalg.norm(achieved_pos - pos))
        n_hat = rot[:, 2]
        orient_err = float(
            np.arccos(np.clip(float(np.dot(achieved_rot[:, 2], n_hat)), -1.0, 1.0))
        )
        runtime.set_joint_positions(q_saved)
        runtime.data.qvel[: qvel_saved.size] = qvel_saved
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
        smooth: bool = False,
    ) -> None:
        qpos, _ = runtime.read_joint_state()
        q_unwrapped = unwrap_joint_target(qpos, q_target)
        if max_step is not None:
            delta = shortest_joint_delta(qpos, q_unwrapped)
            delta = np.clip(delta, -max_step, max_step)
            q_unwrapped = qpos + delta
        if smooth:
            alpha = float(np.clip(JOINT_TARGET_SMOOTHING, 0.0, 1.0))
            if self.smoothed_q_target is None:
                self.smoothed_q_target = q_unwrapped.copy()
            else:
                self.smoothed_q_target = (
                    alpha * q_unwrapped + (1.0 - alpha) * self.smoothed_q_target
                )
            q_unwrapped = unwrap_joint_target(qpos, self.smoothed_q_target)
        low, high = runtime.joint_limits()
        ctrl = actuator_joint_target(qpos, q_unwrapped, low, high)
        runtime.set_control(ctrl)
        self.last_target = q_unwrapped.copy()

    def _signed_standoff(self, runtime: MuJoCoRuntime, *, s: float) -> float:
        pos, _ = runtime.site_pose(EE_SITE)
        p_ref = SURFACE.position(s)
        n_hat = SURFACE.normal(s)
        return float(np.dot(pos - p_ref, n_hat))

    def _cartesian_errors(
        self,
        pos: np.ndarray,
        *,
        s: float,
        normal_offset: float,
    ) -> dict[str, float]:
        p_ref = SURFACE.position(s)
        n_hat = SURFACE.normal(s)
        p_des = p_ref + normal_offset * n_hat
        delta = pos - p_des
        offset_vec = pos - p_ref
        signed_standoff = float(np.dot(offset_vec, n_hat))
        tangential_vec = offset_vec - signed_standoff * n_hat
        return {
            "ee_pose_error": float(np.linalg.norm(delta)),
            "ee_normal_error": signed_standoff - normal_offset,
            "ee_tangential_error": float(np.linalg.norm(tangential_vec)),
            "ee_surface_distance": float(np.linalg.norm(offset_vec)),
            "ee_signed_standoff": signed_standoff,
        }

    def _sample(self, runtime: MuJoCoRuntime, q_target: np.ndarray) -> dict:
        try:
            wrench = runtime.read_site_wrench(FORCE_SENSOR, TORQUE_SENSOR)
        except ValueError:
            wrench = np.zeros(6, dtype=float)
        pos, rot = runtime.site_pose(EE_SITE)
        rot_des = SURFACE.orientation(self.path_s)
        orient_err = float(
            np.arccos(np.clip(float(np.dot(rot[:, 2], rot_des[:, 2])), -1.0, 1.0))
        )
        cartesian = self._cartesian_errors(
            pos,
            s=self.path_s,
            normal_offset=self.last_normal_offset,
        )
        logged_target = self.last_target if self.last_target is not None else q_target
        return {
            "target": logged_target.copy(),
            "control": runtime.data.ctrl.copy(),
            "wrench_tool": wrench.astype(float),
            "ee_pos_error": cartesian["ee_pose_error"],
            "ee_pose_error": cartesian["ee_pose_error"],
            "ee_normal_error": cartesian["ee_normal_error"],
            "ee_tangential_error": cartesian["ee_tangential_error"],
            "ee_surface_distance": cartesian["ee_surface_distance"],
            "ee_orient_error": orient_err,
            "ee_signed_standoff": cartesian["ee_signed_standoff"],
        }
