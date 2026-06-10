from pathlib import Path

from guinsoo_mujoco.assets import (
    AssetManifest,
    missing_required_assets,
    resolve_asset_dir,
    resolve_scene_path,
)


def test_resolve_scene_path_uses_manifest_cache_subdir(tmp_path: Path):
    manifest = AssetManifest(
        robot_id="ur5e",
        source_url="https://example.com",
        license="Apache-2.0",
        cache_subdir="mujoco_menagerie/universal_robots_ur5e",
        entrypoint="scene.xml",
        required_files=("scene.xml",),
    )
    cache_root = tmp_path / "cache"
    scene_dir = cache_root / manifest.cache_subdir
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")

    assert resolve_scene_path(manifest, cache_root) == scene_dir / "scene.xml"
    assert resolve_asset_dir(manifest, cache_root) == scene_dir
    assert missing_required_assets(manifest, cache_root) == ()


def test_missing_required_assets_reports_absent_paths(tmp_path: Path):
    manifest = AssetManifest(
        robot_id="ur5e",
        source_url="https://example.com",
        license="Apache-2.0",
        cache_subdir="mujoco_menagerie/universal_robots_ur5e",
        entrypoint="scene.xml",
        required_files=("scene.xml", "assets"),
    )

    missing = missing_required_assets(manifest, tmp_path)

    assert missing == ("scene.xml", "assets")
