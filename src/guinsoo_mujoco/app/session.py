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
from guinsoo_mujoco.controllers import Controller, create_demo_controller
from guinsoo_mujoco.robots import RobotAdapter, create_default_robot_registry
from guinsoo_mujoco.runtime import MuJoCoRuntime


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
            raise AssetNotReadyError(robot.robot_id, missing)
        scene_path = resolve_scene_path(manifest, cache_root)
        runtime = MuJoCoRuntime(scene_path)
        controller = create_demo_controller(demo, runtime.model.nq)
        controller.reset(runtime)
        return cls(robot=robot, demo=demo, runtime=runtime, controller=controller)

    def reset(self) -> None:
        self.runtime.reset()
        self.controller.reset(self.runtime)

    def step(self, dt: float) -> dict:
        sample = self.controller.step(
            self.runtime, float(self.runtime.data.time), dt
        )
        self.runtime.step()
        return sample

    @staticmethod
    def fetch_hint(robot_id: str) -> str:
        cache = default_cache_root()
        return (
            f"请先下载资产：python -m guinsoo_mujoco.cli fetch-assets {robot_id}\n"
            f"默认缓存目录：{cache}"
        )
