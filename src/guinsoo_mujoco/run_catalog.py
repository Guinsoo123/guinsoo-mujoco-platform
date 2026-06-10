from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from guinsoo_mujoco.data import RunArtifact, default_runs_dir, export_plotjuggler_csv


@dataclass(frozen=True)
class RunSummary:
    stem: str
    artifact: RunArtifact
    robot_id: str
    demo: str
    controller: str
    created_at: str
    sample_count: int | None

    @property
    def label(self) -> str:
        created = self.created_at[:19].replace("T", " ") if self.created_at else self.stem
        samples = f", {self.sample_count} 帧" if self.sample_count is not None else ""
        return f"{created} | {self.robot_id} | {self.demo}{samples}"


def list_runs(runs_dir: str | Path | None = None) -> list[RunSummary]:
    root = Path(runs_dir) if runs_dir else default_runs_dir()
    if not root.exists():
        return []

    summaries: list[RunSummary] = []
    metadata_paths = sorted(root.glob("**/*.json"), reverse=True)
    for metadata_path in metadata_paths:
        try:
            summaries.append(_load_summary(metadata_path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return summaries


def delete_run(summary: RunSummary | RunArtifact) -> None:
    artifact = summary.artifact if isinstance(summary, RunSummary) else summary
    for path in (
        artifact.hdf5_path,
        artifact.metadata_path,
        artifact.plotjuggler_csv_path,
        artifact.hdf5_path.with_name(f"{artifact.hdf5_path.stem}_analysis.plotlayout.xml"),
    ):
        if path is not None and path.exists():
            path.unlink()


def ensure_plotjuggler_csv(artifact: RunArtifact) -> Path:
    if artifact.plotjuggler_csv_path is not None and artifact.plotjuggler_csv_path.exists():
        return artifact.plotjuggler_csv_path
    return export_plotjuggler_csv(
        hdf5_path=artifact.hdf5_path,
        metadata_path=artifact.metadata_path,
    )


def artifact_from_hdf5(hdf5_path: str | Path) -> RunArtifact:
    hdf5_path = Path(hdf5_path)
    metadata_path = hdf5_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    return _artifact_from_metadata(metadata_path)


def _load_summary(metadata_path: Path) -> RunSummary:
    artifact = _artifact_from_metadata(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_count = _read_sample_count(artifact.hdf5_path)
    return RunSummary(
        stem=metadata_path.parent.name,
        artifact=artifact,
        robot_id=str(metadata.get("robot_id", "unknown")),
        demo=str(metadata.get("demo", metadata.get("controller", "unknown"))),
        controller=str(metadata.get("controller", "unknown")),
        created_at=str(metadata.get("created_at", "")),
        sample_count=sample_count,
    )


def _artifact_from_metadata(metadata_path: Path) -> RunArtifact:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = metadata_path.parent
    hdf5_name = str(metadata.get("hdf5_file", metadata_path.with_suffix(".h5").name))
    hdf5_path = root / hdf5_name
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    csv_name = metadata.get("plotjuggler_csv")
    csv_path = root / str(csv_name) if csv_name else hdf5_path.with_name(f"{hdf5_path.stem}_plotjuggler.csv")
    if not csv_path.exists():
        csv_path = None
    return RunArtifact(
        hdf5_path=hdf5_path,
        metadata_path=metadata_path,
        plotjuggler_csv_path=csv_path,
    )


def _read_sample_count(hdf5_path: Path) -> int | None:
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(hdf5_path, "r") as handle:
            if "time" not in handle:
                return None
            return int(handle["time"].shape[0])
    except OSError:
        return None
