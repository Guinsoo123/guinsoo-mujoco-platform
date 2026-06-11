from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from guinsoo_mujoco.assets import (
    AssetManifest,
    resolve_asset_dir,
    resolve_scene_path,
)
from guinsoo_mujoco.controllers import Controller
from guinsoo_mujoco.runtime import MuJoCoRuntime


class ControllerFactory(Protocol):
    def __call__(self, runtime: MuJoCoRuntime) -> Controller:
        ...


@dataclass(frozen=True)
class DemoSpec:
    demo_id: str
    display_name: str
    description: str
    robot_id: str
    package_dir: Path | None
    scene_template: str | None
    doc_path: str | None
    manifest_entrypoint: str | None
    create_controller: ControllerFactory
    scene_template_vars: Callable[[], Mapping[str, str]] | None = None

    def documentation_path(self) -> Path | None:
        if self.doc_path is None or self.package_dir is None:
            return None
        return self.package_dir / self.doc_path

    def resolve_scene(
        self,
        manifest: AssetManifest,
        *,
        cache_root: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        if self.scene_template is not None and self.package_dir is not None:
            template_path = self.package_dir / self.scene_template
            if not template_path.exists():
                raise FileNotFoundError(template_path)
            template = template_path.read_text(encoding="utf-8")
            asset_dir = resolve_asset_dir(manifest, cache_root)
            rendered = (
                template.replace("{{UR5E_DIR}}", str(asset_dir))
                .replace("{{ROBOT_DIR}}", str(asset_dir))
            )
            if self.scene_template_vars is not None:
                for key, value in self.scene_template_vars().items():
                    rendered = rendered.replace(f"{{{{{key}}}}}", value)
            # Write beside ur5e.xml so relative <include file="ur5e.xml"/> and
            # meshdir="assets" inside Menagerie resolve like the stock scene.
            scene_path = asset_dir / f"guinsoo_{self.demo_id}_scene.xml"
            scene_path.write_text(rendered, encoding="utf-8")
            return scene_path
        entrypoint = self.manifest_entrypoint or manifest.entrypoint
        return resolve_scene_path(manifest, cache_root, entrypoint)
