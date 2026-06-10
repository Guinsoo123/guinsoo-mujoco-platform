import pytest

from guinsoo_mujoco.robots import (
    RobotAdapter,
    RobotRegistry,
    create_default_robot_registry,
)


def test_default_registry_exposes_one_stable_and_two_preview_robots():
    registry = create_default_robot_registry()

    robots = registry.list()

    assert [robot.robot_id for robot in robots] == ["ur5e", "openarm", "xlerobot"]
    assert registry.get("ur5e").support_level == "stable"
    assert registry.get("openarm").support_level == "preview"
    assert registry.get("xlerobot").support_level == "experimental"
    assert "ee_pose_avoid" in registry.get("ur5e").demos


def test_registry_rejects_duplicate_robot_ids():
    registry = RobotRegistry()
    adapter = RobotAdapter(
        robot_id="ur5e",
        display_name="UR5e",
        support_level="stable",
        asset_manifest="assets/robots/ur5e.json",
        default_scene="scene.xml",
        demos=["joint_position"],
        description="UR5e 示例",
    )

    registry.register(adapter)

    with pytest.raises(ValueError, match="duplicate robot id"):
        registry.register(adapter)


def test_registry_reports_unknown_robot_id():
    registry = create_default_robot_registry()

    with pytest.raises(KeyError, match="unknown robot id"):
        registry.get("missing")
