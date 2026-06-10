from __future__ import annotations

import argparse
from pathlib import Path
import sys

from guinsoo_mujoco.asset_downloader import AssetDownloader
from guinsoo_mujoco.assets import AssetManifest
from guinsoo_mujoco.robots import create_default_robot_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guinsoo-mujoco")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-assets", help="下载并缓存机器人 MuJoCo 资产")
    fetch.add_argument("robot_id", help="机器人 id，例如 ur5e")
    fetch.add_argument("--cache-root", default=None, help="自定义资产缓存目录")
    args = parser.parse_args(argv)

    if args.command == "fetch-assets":
        registry = create_default_robot_registry()
        robot = registry.get(args.robot_id)
        root = Path(__file__).resolve().parents[2]
        manifest = AssetManifest.load(root / robot.asset_manifest)
        if manifest.license == "UNKNOWN":
            print(
                f"无法下载 {robot.robot_id}：资产许可证为 UNKNOWN，请先完成来源和许可证复核。",
                file=sys.stderr,
            )
            return 2
        cache_path = AssetDownloader(args.cache_root).fetch(manifest)
        print(f"资产已缓存到: {cache_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
