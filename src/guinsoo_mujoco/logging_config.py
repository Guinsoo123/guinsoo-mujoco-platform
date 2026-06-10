from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_CONFIGURED = False
_MUJOCO_WARNING_INSTALLED = False
_LOG_FILE_PATH: Path | None = None


def default_log_dir() -> Path:
    return Path.home() / ".guinsoo_mujoco" / "logs"


def current_log_file() -> Path | None:
    return _LOG_FILE_PATH


def resolve_log_level() -> int:
    level_name = os.environ.get("GUINSOO_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("guinsoo"):
        name = f"guinsoo.{name}"
    return logging.getLogger(name)


def reset_logging_state() -> None:
    global _CONFIGURED, _LOG_FILE_PATH
    root = logging.getLogger("guinsoo")
    root.handlers.clear()
    _CONFIGURED = False
    _LOG_FILE_PATH = None


def setup_logging(
    *,
    app_name: str = "guinsoo",
    level: int | None = None,
    enable_console: bool = True,
    log_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    global _CONFIGURED, _LOG_FILE_PATH

    log_level = level if level is not None else resolve_log_level()
    root = get_logger("guinsoo")
    root.setLevel(log_level)

    if _CONFIGURED and not force:
        root.setLevel(log_level)
        for handler in root.handlers:
            handler.setLevel(log_level)
        if _LOG_FILE_PATH is not None:
            return _LOG_FILE_PATH

    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    directory = Path(log_dir) if log_dir else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = directory / f"{app_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    for child_name in ("ui", "sim", "controller", "planner", "mujoco", "cli"):
        child = logging.getLogger(f"guinsoo.{child_name}")
        child.setLevel(log_level)
        child.propagate = True

    _CONFIGURED = True
    _LOG_FILE_PATH = log_path
    root.info("日志系统已初始化，输出文件：%s", log_path)
    return log_path


def install_mujoco_warning_handler() -> None:
    global _MUJOCO_WARNING_INSTALLED
    if _MUJOCO_WARNING_INSTALLED:
        return

    try:
        import mujoco
    except ImportError:
        return

    logger = get_logger("mujoco")

    def _user_warning(message: str) -> None:
        text = str(message).strip()
        if text:
            logger.warning(text)

    mujoco.set_mju_user_warning(_user_warning)
    _MUJOCO_WARNING_INSTALLED = True
    logger.debug("MuJoCo user warning handler 已安装")
