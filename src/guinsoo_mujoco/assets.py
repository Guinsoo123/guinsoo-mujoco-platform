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
