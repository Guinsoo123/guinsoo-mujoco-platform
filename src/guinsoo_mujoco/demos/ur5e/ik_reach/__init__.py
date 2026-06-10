from __future__ import annotations

from pathlib import Path

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.ur5e.ik_reach.controller import create_controller

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_ik_reach_spec() -> DemoSpec:
    return DemoSpec(
        demo_id="ik_reach",
        display_name="关节正弦运动",
        description="单关节正弦扰动示例（非笛卡尔 IK）。",
        robot_id="ur5e",
        package_dir=_PACKAGE_DIR,
        scene_template=None,
        doc_path="DESIGN.md",
        manifest_entrypoint="scene.xml",
        create_controller=create_controller,
    )
