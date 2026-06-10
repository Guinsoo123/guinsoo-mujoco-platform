from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from guinsoo_mujoco.assets import AssetManifest, default_cache_root


@dataclass(frozen=True)
class GitHubTreeUrl:
    owner: str
    repo: str
    branch: str
    path: str


def parse_github_tree_url(url: str) -> GitHubTreeUrl:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc != "github.com" or len(parts) < 5 or parts[2] != "tree":
        raise ValueError(f"unsupported GitHub tree URL: {url}")
    return GitHubTreeUrl(
        owner=parts[0],
        repo=parts[1],
        branch=parts[3],
        path="/".join(parts[4:]),
    )


class AssetDownloader:
    def __init__(self, cache_root: str | Path | None = None) -> None:
        self.cache_root = Path(cache_root) if cache_root else default_cache_root()

    def fetch(self, manifest: AssetManifest) -> Path:
        if manifest.license == "UNKNOWN":
            raise ValueError(
                f"{manifest.robot_id} asset license is UNKNOWN; review source before fetching"
            )
        tree = parse_github_tree_url(manifest.source_url)
        destination = self.cache_root / manifest.cache_subdir
        destination.mkdir(parents=True, exist_ok=True)
        self._download_tree(tree, tree.path, destination)
        return destination

    def _download_tree(self, tree: GitHubTreeUrl, api_path: str, destination: Path) -> None:
        api_url = (
            f"https://api.github.com/repos/{tree.owner}/{tree.repo}/contents/"
            f"{api_path}?ref={tree.branch}"
        )
        with urlopen(api_url) as response:
            entries = json.loads(response.read().decode("utf-8"))
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            target = destination / Path(entry["path"]).relative_to(tree.path)
            if entry["type"] == "dir":
                target.mkdir(parents=True, exist_ok=True)
                self._download_tree(tree, entry["path"], destination)
            elif entry["type"] == "file":
                target.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(entry["download_url"]) as response:
                    target.write_bytes(response.read())
