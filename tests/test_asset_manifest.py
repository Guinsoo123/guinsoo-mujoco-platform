from pathlib import Path

from guinsoo_mujoco.assets import AssetManifest


def test_asset_manifest_loads_source_and_license_metadata(tmp_path: Path):
    manifest_path = tmp_path / "ur5e.json"
    manifest_path.write_text(
        """
        {
          "robot_id": "ur5e",
          "source_url": "https://github.com/google-deepmind/mujoco_menagerie",
          "license": "Apache-2.0",
          "cache_subdir": "mujoco_menagerie/universal_robots_ur5e",
          "entrypoint": "scene.xml",
          "required_files": ["scene.xml", "ur5e.xml"]
        }
        """,
        encoding="utf-8",
    )

    manifest = AssetManifest.load(manifest_path)

    assert manifest.robot_id == "ur5e"
    assert manifest.license == "Apache-2.0"
    assert manifest.entrypoint == "scene.xml"
    assert manifest.required_files == ("scene.xml", "ur5e.xml")


def test_asset_manifest_validates_required_fields(tmp_path: Path):
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text('{"robot_id": "ur5e"}', encoding="utf-8")

    try:
        AssetManifest.load(manifest_path)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("manifest without required fields should fail")
