from __future__ import annotations

from guinsoo_mujoco.demos.base import DemoSpec
from guinsoo_mujoco.demos.legacy import LEGACY_DEMOS
from guinsoo_mujoco.demos.ur5e.ee_pose_avoid import create_ee_pose_avoid_spec


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
    for spec in LEGACY_DEMOS:
        registry.register(spec)
    registry.register(create_ee_pose_avoid_spec())
    return registry
