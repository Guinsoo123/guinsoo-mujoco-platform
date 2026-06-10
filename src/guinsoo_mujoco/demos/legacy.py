from __future__ import annotations

from guinsoo_mujoco.controllers import create_demo_controller
from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.runtime import MuJoCoRuntime


def _legacy_controller_factory(demo_id: str):
    def factory(runtime: MuJoCoRuntime):
        return create_demo_controller(demo_id, runtime.model.nq)

    return factory


LEGACY_DEMOS: tuple[DemoSpec, ...] = (
    DemoSpec(
        demo_id="joint_position",
        display_name="关节位置保持",
        description="经典关节空间 PD，保持固定 home 姿态。",
        robot_id="ur5e",
        package_dir=None,
        scene_template=None,
        doc_path=None,
        manifest_entrypoint="scene.xml",
        create_controller=_legacy_controller_factory("joint_position"),
    ),
    DemoSpec(
        demo_id="ik_reach",
        display_name="关节正弦运动",
        description="单关节正弦扰动示例（非笛卡尔 IK）。",
        robot_id="ur5e",
        package_dir=None,
        scene_template=None,
        doc_path=None,
        manifest_entrypoint="scene.xml",
        create_controller=_legacy_controller_factory("ik_reach"),
    ),
    DemoSpec(
        demo_id="preview_motion",
        display_name="预览运动",
        description="较低增益的关节位置保持，用于预览机器人。",
        robot_id="openarm",
        package_dir=None,
        scene_template=None,
        doc_path=None,
        manifest_entrypoint="scene.xml",
        create_controller=_legacy_controller_factory("preview_motion"),
    ),
    DemoSpec(
        demo_id="preview_motion",
        display_name="预览运动",
        description="较低增益的关节位置保持，用于 XLeRobot 预览。",
        robot_id="xlerobot",
        package_dir=None,
        scene_template=None,
        doc_path=None,
        manifest_entrypoint="scene.xml",
        create_controller=_legacy_controller_factory("preview_motion"),
    ),
)
