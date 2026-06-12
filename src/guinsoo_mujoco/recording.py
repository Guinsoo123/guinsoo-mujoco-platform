from __future__ import annotations

from pathlib import Path

import numpy as np

from guinsoo_mujoco import __version__
from guinsoo_mujoco.app.session import SimSession
from guinsoo_mujoco.assets import AssetManifest, repo_root
from guinsoo_mujoco.data import RunMetadata, RunRecorder, default_runs_dir
from guinsoo_mujoco.runtime import MuJoCoRuntime

_PHASE_CODE = {
    "approach": 0.0,
    "descend": 1.0,
    "follow": 2.0,
    "retract": 3.0,
    "done": 4.0,
}


def create_run_recorder(
    session: SimSession,
    *,
    output_dir: str | Path | None = None,
    extra_config: dict | None = None,
) -> RunRecorder:
    manifest = AssetManifest.load(repo_root() / session.robot.asset_manifest)
    runtime = session.runtime
    config = {
        "demo": session.demo,
        "scene": str(runtime.model_path),
        "sim_dt": runtime.model.opt.timestep,
    }
    if extra_config:
        config.update(extra_config)
    metadata = RunMetadata(
        robot_id=session.robot.robot_id,
        controller=session.controller.name,
        demo=session.demo,
        mujoco_version=_mujoco_version(),
        asset_source=manifest.source_url,
        app_version=__version__,
        config=config,
        joint_names=tuple(runtime.joint_names()),
        actuator_names=tuple(runtime.actuator_names()),
    )
    return RunRecorder(output_dir or default_runs_dir(), metadata)


def append_step_sample(recorder: RunRecorder, runtime: MuJoCoRuntime, sample: dict) -> None:
    telemetry = runtime.read_telemetry()
    target = sample.get("target")
    if target is None:
        target = telemetry["qpos"]
    sensors = {
        "control_command": np.asarray(sample.get("control", telemetry["ctrl"]), dtype=float),
    }
    for key in (
        "wrench_tool",
        "force_normal",
        "force_normal_raw",
        "force_des",
        "admittance_dn",
        "path_s",
        "phase",
        "status",
        "ee_pos_error",
        "ee_pose_error",
        "ee_normal_error",
        "ee_tangential_error",
        "ee_surface_distance",
        "ee_orient_error",
        "ee_signed_standoff",
        "tool_contact",
    ):
        if key not in sample:
            continue
        value = sample[key]
        if key == "phase":
            sensors["phase_code"] = np.array(
                _PHASE_CODE.get(str(value), -1.0),
                dtype=float,
            )
            continue
        if key == "status":
            continue
        if isinstance(value, (int, float, np.floating)):
            sensors[key] = np.array(float(value), dtype=float)
        elif np.asarray(value).ndim == 0:
            sensors[key] = np.asarray(value, dtype=float)
        else:
            sensors[key] = np.asarray(value, dtype=float)
    recorder.append(
        t=float(sample.get("time", telemetry["time"])),
        qpos=telemetry["qpos"],
        qvel=telemetry["qvel"],
        ctrl=telemetry["ctrl"],
        target=np.asarray(target, dtype=float),
        actuator_force=telemetry["actuator_force"],
        qfrc_actuator=telemetry["qfrc_actuator"],
        sensors=sensors,
    )


def _mujoco_version() -> str:
    try:
        import mujoco
    except ImportError:
        return "unknown"
    return getattr(mujoco, "__version__", "unknown")
