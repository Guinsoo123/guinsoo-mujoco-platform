from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SupportLevel = Literal["stable", "preview", "experimental"]


@dataclass(frozen=True)
class RobotAdapter:
    robot_id: str
    display_name: str
    support_level: SupportLevel
    asset_manifest: str
    default_scene: str
    demos: list[str]
    description: str


class RobotRegistry:
    def __init__(self) -> None:
        self._robots: dict[str, RobotAdapter] = {}

    def register(self, adapter: RobotAdapter) -> None:
        if adapter.robot_id in self._robots:
            raise ValueError(f"duplicate robot id: {adapter.robot_id}")
        self._robots[adapter.robot_id] = adapter

    def get(self, robot_id: str) -> RobotAdapter:
        try:
            return self._robots[robot_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot id: {robot_id}") from exc

    def list(self) -> list[RobotAdapter]:
        return list(self._robots.values())


def create_default_robot_registry() -> RobotRegistry:
    registry = RobotRegistry()
    registry.register(
        RobotAdapter(
            robot_id="ur5e",
            display_name="Universal Robots UR5e",
            support_level="stable",
            asset_manifest="assets/robots/ur5e.json",
            default_scene="scene.xml",
            demos=["joint_position", "ik_reach"],
            description="稳定样板机器人，来源优先使用 DeepMind MuJoCo Menagerie。",
        )
    )
    registry.register(
        RobotAdapter(
            robot_id="openarm",
            display_name="OpenArm",
            support_level="preview",
            asset_manifest="assets/robots/openarm.json",
            default_scene="scene.xml",
            demos=["preview_motion"],
            description="预览适配器，参考 enactic/openarm 与 dora-openarm-mujoco。",
        )
    )
    registry.register(
        RobotAdapter(
            robot_id="xlerobot",
            display_name="XLeRobot-style Mobile Bimanual",
            support_level="experimental",
            asset_manifest="assets/robots/xlerobot.json",
            default_scene="scene.xml",
            demos=["preview_motion"],
            description="实验性轮式双臂机器人适配器，模型来源需完成许可证检查。",
        )
    )
    return registry
