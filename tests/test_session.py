from pathlib import Path

import pytest

from guinsoo_mujoco.app.session import AssetNotReadyError, SimSession
from guinsoo_mujoco.assets import AssetManifest, repo_root


def test_sim_session_loads_cached_ur5e_when_assets_present():
    manifest = AssetManifest.load(repo_root() / "assets/robots/ur5e.json")
    from guinsoo_mujoco.assets import default_cache_root, missing_required_assets

    if missing_required_assets(manifest):
        pytest.skip("UR5e assets are not cached locally")

    session = SimSession.load("ur5e", "joint_position")

    assert session.robot.robot_id == "ur5e"
    assert session.runtime.model.nq == 6
    sample = session.step(0.002)
    assert sample["controller"] == "joint_position"
    assert sample["control"].shape == (6,)


def test_sim_session_raises_when_assets_missing(tmp_path: Path):
    try:
        SimSession.load(
            "ur5e",
            "joint_position",
            cache_root=tmp_path / "empty-cache",
            project_root=repo_root(),
        )
    except AssetNotReadyError as exc:
        assert exc.robot_id == "ur5e"
        assert exc.missing
    else:
        raise AssertionError("expected AssetNotReadyError for missing cache")
