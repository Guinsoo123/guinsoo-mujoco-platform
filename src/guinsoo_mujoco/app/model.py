from __future__ import annotations

from dataclasses import dataclass

from guinsoo_mujoco.robots import RobotAdapter, create_default_robot_registry


SUPPORT_LABELS = {
    "stable": "稳定样板",
    "preview": "预览适配",
    "experimental": "实验性预览",
}


@dataclass(frozen=True)
class RobotSelection:
    robot: RobotAdapter
    demo: str


class SimStudioModel:
    def __init__(self) -> None:
        self.registry = create_default_robot_registry()

    def robot_cards(self) -> list[dict[str, str]]:
        cards: list[dict[str, str]] = []
        for robot in self.registry.list():
            cards.append(
                {
                    "robot_id": robot.robot_id,
                    "display_name": robot.display_name,
                    "support_level": robot.support_level,
                    "support_label": SUPPORT_LABELS[robot.support_level],
                    "description": robot.description,
                    "demos": ", ".join(robot.demos),
                }
            )
        return cards

    def select(self, robot_id: str, demo: str | None = None) -> RobotSelection:
        robot = self.registry.get(robot_id)
        selected_demo = demo or robot.demos[0]
        if selected_demo not in robot.demos:
            raise ValueError(f"demo {selected_demo!r} is not available for {robot_id}")
        return RobotSelection(robot=robot, demo=selected_demo)
