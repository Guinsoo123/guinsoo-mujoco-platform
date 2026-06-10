import json
from pathlib import Path

import h5py
import numpy as np

from guinsoo_mujoco.data import RunMetadata, RunRecorder


def test_run_recorder_writes_hdf5_samples_and_json_metadata(tmp_path: Path):
    metadata = RunMetadata(
        robot_id="ur5e",
        controller="joint_position",
        mujoco_version="test",
        asset_source="https://github.com/google-deepmind/mujoco_menagerie",
        app_version="0.1.0",
        config={"duration": 0.02},
    )
    recorder = RunRecorder(tmp_path, metadata)

    recorder.append(
        t=0.0,
        qpos=np.array([1.0, 2.0]),
        qvel=np.array([0.1, 0.2]),
        ctrl=np.array([0.0, 0.5]),
        sensors={"ee_error": np.array([0.03])},
    )
    recorder.append(
        t=0.01,
        qpos=np.array([1.1, 2.1]),
        qvel=np.array([0.2, 0.3]),
        ctrl=np.array([0.1, 0.6]),
        sensors={"ee_error": np.array([0.02])},
    )

    artifact = recorder.close()

    assert artifact.hdf5_path.exists()
    assert artifact.metadata_path.exists()
    with h5py.File(artifact.hdf5_path, "r") as handle:
        np.testing.assert_allclose(handle["time"][:], [0.0, 0.01])
        np.testing.assert_allclose(handle["qpos"][:], [[1.0, 2.0], [1.1, 2.1]])
        np.testing.assert_allclose(handle["qvel"][:], [[0.1, 0.2], [0.2, 0.3]])
        np.testing.assert_allclose(handle["ctrl"][:], [[0.0, 0.5], [0.1, 0.6]])
        np.testing.assert_allclose(handle["sensors/ee_error"][:], [[0.03], [0.02]])

    saved_metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["robot_id"] == "ur5e"
    assert saved_metadata["controller"] == "joint_position"
    assert saved_metadata["config"] == {"duration": 0.02}
