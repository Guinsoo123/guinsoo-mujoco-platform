from __future__ import annotations

from pathlib import Path

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.ur5e.surface_wipe.controller import SurfaceWipeController
from guinsoo_mujoco.demos.ur5e.surface_wipe.scene_vars import surface_wipe_scene_template_vars
from guinsoo_mujoco.runtime import MuJoCoRuntime

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_surface_wipe_spec() -> DemoSpec:
    return DemoSpec(
        demo_id="surface_wipe",
        display_name="曲面擦拭 (导纳)",
        description=(
            "正弦波浪面擦拭：法向二阶导纳恒压 + 沿 X 切向轨迹跟踪。"
            "腕部力传感器反馈，内环为 Menagerie 位置伺服。"
        ),
        robot_id="ur5e",
        package_dir=_PACKAGE_DIR,
        scene_template="scene.xml",
        doc_path="DESIGN.md",
        manifest_entrypoint=None,
        create_controller=lambda runtime: SurfaceWipeController(runtime),
        scene_template_vars=surface_wipe_scene_template_vars,
    )
