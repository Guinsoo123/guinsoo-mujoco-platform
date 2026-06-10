from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class RunMetadata:
    robot_id: str
    controller: str
    demo: str
    mujoco_version: str
    asset_source: str
    app_version: str
    config: dict
    joint_names: tuple[str, ...] = ()
    actuator_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunArtifact:
    hdf5_path: Path
    metadata_path: Path
    plotjuggler_csv_path: Path | None = None


@dataclass(frozen=True)
class Episode:
    metadata: dict
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    target: np.ndarray
    actuator_force: np.ndarray
    qfrc_actuator: np.ndarray
    sensors: dict[str, np.ndarray]


def default_runs_dir() -> Path:
    return Path.home() / ".guinsoo_mujoco" / "runs"


class RunBrowser:
    @staticmethod
    def open(artifact: RunArtifact | tuple[str | Path, str | Path]) -> Episode:
        if isinstance(artifact, RunArtifact):
            hdf5_path = artifact.hdf5_path
            metadata_path = artifact.metadata_path
        else:
            hdf5_path = Path(artifact[0])
            metadata_path = Path(artifact[1])

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with h5py.File(hdf5_path, "r") as handle:
            sensors = {
                name: handle["sensors"][name][:]
                for name in handle.get("sensors", {}).keys()
            }
            target = _read_optional_dataset(handle, "target", handle["qpos"])
            actuator_force = _read_optional_dataset(handle, "actuator_force", handle["ctrl"])
            qfrc_actuator = _read_optional_dataset(
                handle, "qfrc_actuator", handle["ctrl"]
            )
            return Episode(
                metadata=metadata,
                time=handle["time"][:],
                qpos=handle["qpos"][:],
                qvel=handle["qvel"][:],
                ctrl=handle["ctrl"][:],
                target=target,
                actuator_force=actuator_force,
                qfrc_actuator=qfrc_actuator,
                sensors=sensors,
            )


class RunRecorder:
    def __init__(self, output_dir: str | Path, metadata: RunMetadata) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = metadata
        self._times: list[float] = []
        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._ctrl: list[np.ndarray] = []
        self._target: list[np.ndarray] = []
        self._actuator_force: list[np.ndarray] = []
        self._qfrc_actuator: list[np.ndarray] = []
        self._sensors: dict[str, list[np.ndarray]] = {}
        self._closed = False

    def append(
        self,
        *,
        t: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        target: np.ndarray | None = None,
        actuator_force: np.ndarray | None = None,
        qfrc_actuator: np.ndarray | None = None,
        sensors: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed RunRecorder")
        self._times.append(float(t))
        self._qpos.append(np.asarray(qpos, dtype=float).copy())
        self._qvel.append(np.asarray(qvel, dtype=float).copy())
        self._ctrl.append(np.asarray(ctrl, dtype=float).copy())
        if target is None:
            target = np.full_like(qpos, np.nan, dtype=float)
        self._target.append(np.asarray(target, dtype=float).copy())
        if actuator_force is None:
            actuator_force = np.full_like(ctrl, np.nan, dtype=float)
        self._actuator_force.append(np.asarray(actuator_force, dtype=float).copy())
        if qfrc_actuator is None:
            qfrc_actuator = np.full_like(ctrl, np.nan, dtype=float)
        self._qfrc_actuator.append(np.asarray(qfrc_actuator, dtype=float).copy())
        for name, value in (sensors or {}).items():
            self._sensors.setdefault(name, []).append(
                np.asarray(value, dtype=float).copy()
            )

    def close(self) -> RunArtifact:
        self._closed = True
        run_dir = self._run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = "episode"
        hdf5_path = run_dir / f"{stem}.h5"
        metadata_path = run_dir / f"{stem}.json"
        plotjuggler_csv_path = run_dir / f"{stem}_plotjuggler.csv"

        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("time", data=np.asarray(self._times, dtype=float))
            handle.create_dataset("qpos", data=np.asarray(self._qpos, dtype=float))
            handle.create_dataset("qvel", data=np.asarray(self._qvel, dtype=float))
            handle.create_dataset("ctrl", data=np.asarray(self._ctrl, dtype=float))
            handle.create_dataset("target", data=np.asarray(self._target, dtype=float))
            handle.create_dataset(
                "actuator_force", data=np.asarray(self._actuator_force, dtype=float)
            )
            handle.create_dataset(
                "qfrc_actuator", data=np.asarray(self._qfrc_actuator, dtype=float)
            )
            sensors_group = handle.create_group("sensors")
            for name, values in self._sensors.items():
                sensors_group.create_dataset(name, data=np.asarray(values, dtype=float))

        metadata = asdict(self.metadata)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        metadata["run_dir"] = str(run_dir)
        metadata["hdf5_file"] = hdf5_path.name
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        export_plotjuggler_csv(
            hdf5_path=hdf5_path,
            metadata_path=metadata_path,
            csv_path=plotjuggler_csv_path,
        )
        metadata["plotjuggler_csv"] = plotjuggler_csv_path.name
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return RunArtifact(
            hdf5_path=hdf5_path,
            metadata_path=metadata_path,
            plotjuggler_csv_path=plotjuggler_csv_path,
        )

    def _run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{timestamp}-{self.metadata.demo}"
        return self.output_dir / folder_name


def export_plotjuggler_csv(
    *,
    hdf5_path: str | Path,
    metadata_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> Path:
    """Export a flat CSV time series for PlotJuggler and similar tools."""
    hdf5_path = Path(hdf5_path)
    if metadata_path is None:
        metadata_path = hdf5_path.with_suffix(".json")
    else:
        metadata_path = Path(metadata_path)
    if csv_path is None:
        csv_path = hdf5_path.with_name(f"{hdf5_path.stem}_plotjuggler.csv")
    else:
        csv_path = Path(csv_path)

    episode = RunBrowser.open((hdf5_path, metadata_path))
    columns = build_plotjuggler_columns(episode)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        row_count = len(episode.time)
        for index in range(row_count):
            writer.writerow(columns[col][index] for col in columns)
    return csv_path


def build_plotjuggler_columns(episode: Episode) -> dict[str, list[float]]:
    names = _signal_names(episode.metadata)
    columns: dict[str, list[float]] = {"timestamp": episode.time.astype(float).tolist()}
    if not names and episode.qpos.ndim > 1:
        names = [f"joint_{index}" for index in range(episode.qpos.shape[1])]

    for index, name in enumerate(names):
        safe = _safe_signal_name(name, index)
        columns[f"joint/{safe}/qpos"] = episode.qpos[:, index].astype(float).tolist()
        columns[f"joint/{safe}/qvel"] = episode.qvel[:, index].astype(float).tolist()
        columns[f"joint/{safe}/target"] = episode.target[:, index].astype(float).tolist()
        columns[f"joint/{safe}/ctrl"] = episode.ctrl[:, index].astype(float).tolist()
        columns[f"joint/{safe}/actuator_force"] = episode.actuator_force[
            :, index
        ].astype(float).tolist()
        columns[f"joint/{safe}/qfrc_actuator"] = episode.qfrc_actuator[
            :, index
        ].astype(float).tolist()
        columns[f"joint/{safe}/tracking_error"] = (
            episode.target[:, index] - episode.qpos[:, index]
        ).astype(float).tolist()

    for sensor_name, values in episode.sensors.items():
        if values.ndim == 1:
            columns[f"sensor/{sensor_name}"] = values.astype(float).tolist()
        else:
            width = values.shape[1]
            for index in range(width):
                suffix = names[index] if index < len(names) else str(index)
                columns[f"sensor/{sensor_name}/{_safe_signal_name(suffix, index)}"] = (
                    values[:, index].astype(float).tolist()
                )

    return columns


def _signal_names(metadata: dict) -> list[str]:
    names = metadata.get("actuator_names") or metadata.get("joint_names") or []
    if not names:
        return []
    return [str(name) for name in names]


def _safe_signal_name(name: str, index: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    return cleaned or f"joint_{index}"


def _read_optional_dataset(handle: h5py.File, name: str, template) -> np.ndarray:
    if name in handle:
        return handle[name][:]
    values = template[:]
    return np.full_like(values, np.nan, dtype=float)
