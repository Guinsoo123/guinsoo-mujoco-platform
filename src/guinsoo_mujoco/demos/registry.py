from __future__ import annotations

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.openarm.preview_motion import create_preview_motion_spec
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid import create_ee_pose_avoid_spec
from guinsoo_mujoco.demos.ur5e.ik_reach import create_ik_reach_spec
from guinsoo_mujoco.demos.ur5e.joint_position import create_joint_position_spec
from guinsoo_mujoco.demos.xlerobot.preview_motion import create_xlerobot_preview_motion_spec


class DemoRegistry:
    def __init__(self) -> None:
        self._demos: dict[tuple[str, str], DemoSpec] = {}

    def register(self, spec: DemoSpec) -> None:
        key = (spec.robot_id, spec.demo_id)
        if key in self._demos:
            raise ValueError(f"duplicate demo: {spec.robot_id}/{spec.demo_id}")
        self._demos[key] = spec

    def get(self, robot_id: str, demo_id: str) -> DemoSpec:
        try:
            return self._demos[(robot_id, demo_id)]
        except KeyError as exc:
            raise KeyError(f"unknown demo: {robot_id}/{demo_id}") from exc

    def list_for_robot(self, robot_id: str) -> list[DemoSpec]:
        return [
            spec
            for (rid, _), spec in sorted(self._demos.items())
            if rid == robot_id
        ]

    def demo_ids_for_robot(self, robot_id: str) -> list[str]:
        return [spec.demo_id for spec in self.list_for_robot(robot_id)]


def create_demo_registry() -> DemoRegistry:
    registry = DemoRegistry()
    registry.register(create_joint_position_spec())
    registry.register(create_ik_reach_spec())
    registry.register(create_ee_pose_avoid_spec())
    registry.register(create_preview_motion_spec())
    registry.register(create_xlerobot_preview_motion_spec())
    return registry
