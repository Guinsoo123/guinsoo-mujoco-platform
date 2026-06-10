import pytest

from guinsoo_mujoco.app.session import SimSession
from guinsoo_mujoco.assets import missing_required_assets, repo_root
from guinsoo_mujoco.assets import AssetManifest
from guinsoo_mujoco.data import RunRecorder
from guinsoo_mujoco.recording import append_step_sample, create_run_recorder


def test_create_run_recorder_captures_actuator_names_from_session():
    manifest = AssetManifest.load(repo_root() / "assets/robots/ur5e.json")
    if missing_required_assets(manifest):
        pytest.skip("UR5e assets are not cached locally")

    session = SimSession.load("ur5e", "joint_position")
    recorder = create_run_recorder(session)
    sample = session.step(0.002)

    append_step_sample(recorder, session.runtime, sample)

    assert isinstance(recorder, RunRecorder)
    assert len(recorder._times) == 1
    assert recorder.metadata.actuator_names
    assert recorder.metadata.demo == "joint_position"
