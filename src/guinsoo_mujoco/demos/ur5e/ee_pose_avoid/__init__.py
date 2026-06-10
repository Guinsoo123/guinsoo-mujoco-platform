from __future__ import annotations

from pathlib import Path

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.controller import EEPoseAvoidController
from guinsoo_mujoco.runtime import MuJoCoRuntime

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_ee_pose_avoid_spec() -> DemoSpec:
    return DemoSpec(
        demo_id="ee_pose_avoid",
        display_name="末端避障到点 (RRT)",
        description=(
            "RRT-Connect 关节空间避障规划 + 路径跟踪。"
            "场景含障碍物与 4 个目标末端位姿标记，依次到达各路点。"
        ),
        robot_id="ur5e",
        package_dir=_PACKAGE_DIR,
        scene_template="scene.xml",
        doc_path="DESIGN.md",
        manifest_entrypoint=None,
        create_controller=lambda runtime: EEPoseAvoidController(runtime),
    )
