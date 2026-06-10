from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AssetManifest:
    robot_id: str
    source_url: str
    license: str
    cache_subdir: str
    entrypoint: str
    required_files: tuple[str, ...]
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "AssetManifest":
        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "robot_id",
            "source_url",
            "license",
            "cache_subdir",
            "entrypoint",
            "required_files",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(
                f"missing required fields in {manifest_path}: {', '.join(missing)}"
            )
        required_files = data["required_files"]
        if not isinstance(required_files, list) or not required_files:
            raise ValueError("required_files must be a non-empty list")
        return cls(
            robot_id=str(data["robot_id"]),
            source_url=str(data["source_url"]),
            license=str(data["license"]),
            cache_subdir=str(data["cache_subdir"]),
            entrypoint=str(data["entrypoint"]),
            required_files=tuple(str(item) for item in required_files),
            notes=str(data.get("notes", "")),
        )


def default_cache_root() -> Path:
    return Path.home() / ".guinsoo_mujoco" / "assets"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_asset_dir(
    manifest: AssetManifest, cache_root: str | Path | None = None
) -> Path:
    root = Path(cache_root) if cache_root else default_cache_root()
    return root / manifest.cache_subdir


def resolve_scene_path(
    manifest: AssetManifest,
    cache_root: str | Path | None = None,
    entrypoint: str | None = None,
) -> Path:
    return resolve_asset_dir(manifest, cache_root) / (entrypoint or manifest.entrypoint)


def missing_required_assets(
    manifest: AssetManifest, cache_root: str | Path | None = None
) -> tuple[str, ...]:
    base = resolve_asset_dir(manifest, cache_root)
    return tuple(
        name for name in manifest.required_files if not (base / name).exists()
    )
