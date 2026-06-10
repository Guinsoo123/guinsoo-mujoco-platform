from guinsoo_mujoco.data import RunMetadata, RunRecorder
from guinsoo_mujoco.run_catalog import delete_run, list_runs


def test_list_and_delete_runs(tmp_path):
    recorder = RunRecorder(
        tmp_path,
        RunMetadata(
            robot_id="ur5e",
            controller="joint_position",
            demo="joint_position",
            mujoco_version="test",
            asset_source="source",
            app_version="0.1.0",
            config={},
            actuator_names=("shoulder_pan",),
        ),
    )
    import numpy as np

    recorder.append(
        t=0.0,
        qpos=np.array([0.0]),
        qvel=np.array([0.0]),
        ctrl=np.array([0.0]),
        target=np.array([0.1]),
        actuator_force=np.array([1.0]),
        qfrc_actuator=np.array([2.0]),
    )
    artifact = recorder.close()

    summaries = list_runs(tmp_path)
    assert len(summaries) == 1
    assert summaries[0].robot_id == "ur5e"
    assert summaries[0].sample_count == 1
    assert summaries[0].artifact.hdf5_path == artifact.hdf5_path

    delete_run(summaries[0])
    assert list_runs(tmp_path) == []
    assert not artifact.hdf5_path.exists()
    assert not artifact.metadata_path.exists()


def test_list_runs_skips_invalid_metadata(tmp_path):
    (tmp_path / "broken.json").write_text('{"robot_id":"x"}', encoding="utf-8")
    assert list_runs(tmp_path) == []
