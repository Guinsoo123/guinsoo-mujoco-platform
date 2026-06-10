from __future__ import annotations

from pathlib import Path

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.openarm.preview_motion.controller import create_controller

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_preview_motion_spec() -> DemoSpec:
    return DemoSpec(
        demo_id="preview_motion",
        display_name="预览运动",
        description="较低增益的关节位置保持，用于预览机器人。",
        robot_id="openarm",
        package_dir=_PACKAGE_DIR,
        scene_template=None,
        doc_path="DESIGN.md",
        manifest_entrypoint="scene.xml",
        create_controller=create_controller,
    )
