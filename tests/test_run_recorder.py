import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from guinsoo_mujoco.data import (
    RunBrowser,
    RunMetadata,
    RunRecorder,
    build_plotjuggler_columns,
    export_plotjuggler_csv,
)


def test_run_recorder_writes_joint_telemetry_and_plotjuggler_csv(tmp_path: Path):
    metadata = RunMetadata(
        robot_id="ur5e",
        controller="joint_position",
        demo="joint_position",
        mujoco_version="test",
        asset_source="https://github.com/google-deepmind/mujoco_menagerie",
        app_version="0.1.0",
        config={"duration": 0.02},
        joint_names=("j1", "j2"),
        actuator_names=("j1", "j2"),
    )
    recorder = RunRecorder(tmp_path, metadata)

    recorder.append(
        t=0.0,
        qpos=np.array([1.0, 2.0]),
        qvel=np.array([0.1, 0.2]),
        ctrl=np.array([0.0, 0.5]),
        target=np.array([1.1, 2.1]),
        actuator_force=np.array([3.0, 4.0]),
        qfrc_actuator=np.array([5.0, 6.0]),
        sensors={"control_command": np.array([0.0, 0.5])},
    )
    recorder.append(
        t=0.01,
        qpos=np.array([1.1, 2.1]),
        qvel=np.array([0.2, 0.3]),
        ctrl=np.array([0.1, 0.6]),
        target=np.array([1.2, 2.2]),
        actuator_force=np.array([3.1, 4.1]),
        qfrc_actuator=np.array([5.1, 6.1]),
        sensors={"control_command": np.array([0.1, 0.6])},
    )

    artifact = recorder.close()

    assert artifact.hdf5_path.exists()
    assert artifact.metadata_path.exists()
    assert artifact.plotjuggler_csv_path is not None
    assert artifact.plotjuggler_csv_path.exists()
    with h5py.File(artifact.hdf5_path, "r") as handle:
        np.testing.assert_allclose(handle["time"][:], [0.0, 0.01])
        np.testing.assert_allclose(handle["qpos"][:], [[1.0, 2.0], [1.1, 2.1]])
        np.testing.assert_allclose(handle["target"][:], [[1.1, 2.1], [1.2, 2.2]])
        np.testing.assert_allclose(handle["actuator_force"][:], [[3.0, 4.0], [3.1, 4.1]])
        np.testing.assert_allclose(handle["qfrc_actuator"][:], [[5.0, 6.0], [5.1, 6.1]])
        np.testing.assert_allclose(
            handle["sensors/control_command"][:], [[0.0, 0.5], [0.1, 0.6]]
        )

    saved_metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["robot_id"] == "ur5e"
    assert saved_metadata["demo"] == "joint_position"
    assert saved_metadata["actuator_names"] == ["j1", "j2"]
    assert saved_metadata["plotjuggler_csv"].endswith("_plotjuggler.csv")

    episode = RunBrowser.open(artifact)
    columns = build_plotjuggler_columns(episode)
    assert "joint/j1/qpos" in columns
    assert "joint/j1/tracking_error" in columns
    assert columns["joint/j1/tracking_error"][0] == pytest.approx(0.1)

    csv_path = export_plotjuggler_csv(
        hdf5_path=artifact.hdf5_path,
        metadata_path=artifact.metadata_path,
        csv_path=tmp_path / "manual.csv",
    )
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "timestamp" in header
    assert "joint/j1/actuator_force" in header
