import numpy as np

from guinsoo_mujoco.data import RunBrowser, RunMetadata, RunRecorder


def test_run_browser_reads_recorded_episode(tmp_path):
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
            actuator_names=("j1",),
        ),
    )
    recorder.append(
        t=0.0,
        qpos=np.array([0.0]),
        qvel=np.array([0.0]),
        ctrl=np.array([1.0]),
        target=np.array([0.2]),
        actuator_force=np.array([2.0]),
        qfrc_actuator=np.array([3.0]),
        sensors={"control_command": np.array([1.0])},
    )
    artifact = recorder.close()

    episode = RunBrowser.open(artifact)

    assert episode.metadata["robot_id"] == "ur5e"
    np.testing.assert_allclose(episode.time, [0.0])
    np.testing.assert_allclose(episode.qpos, [[0.0]])
    np.testing.assert_allclose(episode.target, [[0.2]])
    np.testing.assert_allclose(episode.actuator_force, [[2.0]])
    np.testing.assert_allclose(episode.sensors["control_command"], [[1.0]])
