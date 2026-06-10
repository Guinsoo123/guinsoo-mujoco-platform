import numpy as np

from guinsoo_mujoco.data import RunBrowser, RunMetadata, RunRecorder


def test_run_browser_reads_recorded_episode(tmp_path):
    recorder = RunRecorder(
        tmp_path,
        RunMetadata(
            robot_id="ur5e",
            controller="joint_position",
            mujoco_version="test",
            asset_source="source",
            app_version="0.1.0",
            config={},
        ),
    )
    recorder.append(
        t=0.0,
        qpos=np.array([0.0]),
        qvel=np.array([0.0]),
        ctrl=np.array([1.0]),
        sensors={"error": np.array([0.1])},
    )
    artifact = recorder.close()

    episode = RunBrowser.open(artifact)

    assert episode.metadata["robot_id"] == "ur5e"
    np.testing.assert_allclose(episode.time, [0.0])
    np.testing.assert_allclose(episode.qpos, [[0.0]])
    np.testing.assert_allclose(episode.sensors["error"], [[0.1]])
