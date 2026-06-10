from pathlib import Path

from guinsoo_mujoco.assets import AssetManifest


def test_repo_robot_manifests_are_valid():
    root = Path(__file__).resolve().parents[1]
    manifest_paths = sorted((root / "assets" / "robots").glob("*.json"))

    assert [path.name for path in manifest_paths] == [
        "openarm.json",
        "ur5e.json",
        "xlerobot.json",
    ]
    manifests = [AssetManifest.load(path) for path in manifest_paths]
    assert {manifest.robot_id for manifest in manifests} == {
        "openarm",
        "ur5e",
        "xlerobot",
    }
