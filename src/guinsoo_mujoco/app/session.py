from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from guinsoo_mujoco.assets import (
    AssetManifest,
    default_cache_root,
    missing_required_assets,
    repo_root,
    resolve_scene_path,
)
from guinsoo_mujoco.controllers import Controller
from guinsoo_mujoco.demos.registry import create_demo_registry
from guinsoo_mujoco.logging_config import get_logger
from guinsoo_mujoco.robots import RobotAdapter, create_default_robot_registry
from guinsoo_mujoco.runtime import MuJoCoRuntime

logger = get_logger("sim")


class AssetNotReadyError(FileNotFoundError):
    def __init__(self, robot_id: str, missing: tuple[str, ...]) -> None:
        self.robot_id = robot_id
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(
            f"{robot_id} assets are not cached; missing: {names}. "
            f"Run: python -m guinsoo_mujoco.cli fetch-assets {robot_id}"
        )


@dataclass
class SimSession:
    robot: RobotAdapter
    demo: str
    runtime: MuJoCoRuntime
    controller: Controller

    @classmethod
    def load(
        cls,
        robot_id: str,
        demo: str,
        *,
        cache_root: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "SimSession":
        registry = create_default_robot_registry()
        robot = registry.get(robot_id)
        root = Path(project_root) if project_root else repo_root()
        manifest = AssetManifest.load(root / robot.asset_manifest)
        missing = missing_required_assets(manifest, cache_root)
        if missing:
            logger.warning(
                "资产未缓存：robot=%s missing=%s",
                robot.robot_id,
                ", ".join(missing),
            )
            raise AssetNotReadyError(robot.robot_id, missing)
        demo_registry = create_demo_registry()
        demo_spec = demo_registry.get(robot_id, demo)
        scene_path = demo_spec.resolve_scene(manifest, cache_root=cache_root, project_root=root)
        logger.info(
            "加载会话：robot=%s demo=%s scene=%s",
            robot_id,
            demo,
            scene_path,
        )
        runtime = MuJoCoRuntime(scene_path)
        controller = demo_spec.create_controller(runtime)
        controller.reset(runtime)
        logger.info("控制器已就绪：%s", controller.name)
        return cls(robot=robot, demo=demo, runtime=runtime, controller=controller)

    def reset(self) -> None:
        self.runtime.reset()
        self.runtime.data.qvel[:] = 0.0
        self.runtime.data.qacc[:] = 0.0
        self.controller.reset(self.runtime)
        self.runtime.forward()
        logger.info(
            "会话已重置：robot=%s demo=%s time=%.3f",
            self.robot.robot_id,
            self.demo,
            float(self.runtime.data.time),
        )

    def step(self, dt: float) -> dict:
        sample = self.controller.step(
            self.runtime, float(self.runtime.data.time), dt
        )
        self.runtime.step()
        telemetry = self.runtime.read_telemetry()
        sample["time"] = telemetry["time"]
        sample["qpos"] = telemetry["qpos"]
        sample["qvel"] = telemetry["qvel"]
        sample["ctrl"] = telemetry["ctrl"]
        sample["actuator_force"] = telemetry["actuator_force"]
        sample["qfrc_actuator"] = telemetry["qfrc_actuator"]
        return sample

    @staticmethod
    def fetch_hint(robot_id: str) -> str:
        cache = default_cache_root()
        return (
            f"请先下载资产：python -m guinsoo_mujoco.cli fetch-assets {robot_id}\n"
            f"默认缓存目录：{cache}"
        )
