from __future__ import annotations

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
    mujoco_version: str
    asset_source: str
    app_version: str
    config: dict


@dataclass(frozen=True)
class RunArtifact:
    hdf5_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class Episode:
    metadata: dict
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    sensors: dict[str, np.ndarray]


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
            return Episode(
                metadata=metadata,
                time=handle["time"][:],
                qpos=handle["qpos"][:],
                qvel=handle["qvel"][:],
                ctrl=handle["ctrl"][:],
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
        self._sensors: dict[str, list[np.ndarray]] = {}
        self._closed = False

    def append(
        self,
        *,
        t: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        sensors: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed RunRecorder")
        self._times.append(float(t))
        self._qpos.append(np.asarray(qpos, dtype=float).copy())
        self._qvel.append(np.asarray(qvel, dtype=float).copy())
        self._ctrl.append(np.asarray(ctrl, dtype=float).copy())
        for name, value in (sensors or {}).items():
            self._sensors.setdefault(name, []).append(np.asarray(value, dtype=float).copy())

    def close(self) -> RunArtifact:
        self._closed = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._artifact_stem()
        hdf5_path = self.output_dir / f"{stem}.h5"
        metadata_path = self.output_dir / f"{stem}.json"

        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("time", data=np.asarray(self._times, dtype=float))
            handle.create_dataset("qpos", data=np.asarray(self._qpos, dtype=float))
            handle.create_dataset("qvel", data=np.asarray(self._qvel, dtype=float))
            handle.create_dataset("ctrl", data=np.asarray(self._ctrl, dtype=float))
            sensors_group = handle.create_group("sensors")
            for name, values in self._sensors.items():
                sensors_group.create_dataset(name, data=np.asarray(values, dtype=float))

        metadata = asdict(self.metadata)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        metadata["hdf5_file"] = hdf5_path.name
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return RunArtifact(hdf5_path=hdf5_path, metadata_path=metadata_path)

    def _artifact_stem(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{self.metadata.robot_id}-{self.metadata.controller}"
