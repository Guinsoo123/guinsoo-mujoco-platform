from pathlib import Path

import pytest

from guinsoo_mujoco.assets import AssetManifest, repo_root
from guinsoo_mujoco.demos.registry import create_demo_registry
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.config import WAYPOINTS


def test_demo_registry_lists_ur5e_demos():
    registry = create_demo_registry()
    demo_ids = registry.demo_ids_for_robot("ur5e")
    assert "joint_position" in demo_ids
    assert "ik_reach" in demo_ids
    assert "ee_pose_avoid" in demo_ids


def test_ee_pose_avoid_spec_has_documentation():
    registry = create_demo_registry()
    spec = registry.get("ur5e", "ee_pose_avoid")
    doc_path = spec.documentation_path()
    assert doc_path is not None
    assert doc_path.exists()
    assert "RRT" in doc_path.read_text(encoding="utf-8")


def test_ee_pose_avoid_scene_template_resolves(tmp_path: Path):
    registry = create_demo_registry()
    spec = registry.get("ur5e", "ee_pose_avoid")
    manifest = AssetManifest.load(repo_root() / "assets/robots/ur5e.json")
    cache_root = tmp_path / "cache"
    asset_dir = cache_root / manifest.cache_subdir
    asset_dir.mkdir(parents=True)
    (asset_dir / "ur5e.xml").write_text("<mujoco/>", encoding="utf-8")

    scene_path = spec.resolve_scene(manifest, cache_root=cache_root)
    content = scene_path.read_text(encoding="utf-8")
    assert scene_path == asset_dir / "guinsoo_ee_pose_avoid_scene.xml"
    assert '<include file="ur5e.xml"/>' in content
    assert "{{UR5E_DIR}}" not in content


def test_demo_registry_rejects_unknown_demo():
    registry = create_demo_registry()
    with pytest.raises(KeyError):
        registry.get("ur5e", "missing_demo")


def test_waypoints_defined_for_demo_sequence():
    assert len(WAYPOINTS) == 4
    names = [waypoint.name for waypoint in WAYPOINTS]
    assert names == ["home", "approach", "over_obstacle", "place"]
