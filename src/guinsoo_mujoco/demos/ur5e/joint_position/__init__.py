from __future__ import annotations

from pathlib import Path

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.ur5e.joint_position.controller import create_controller

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_joint_position_spec() -> DemoSpec:
    return DemoSpec(
        demo_id="joint_position",
        display_name="关节位置保持",
        description="经典关节空间 PD，保持固定 home 姿态。",
        robot_id="ur5e",
        package_dir=_PACKAGE_DIR,
        scene_template=None,
        doc_path="DESIGN.md",
        manifest_entrypoint="scene.xml",
        create_controller=create_controller,
    )
