import logging
from pathlib import Path

from guinsoo_mujoco.logging_config import (
    default_log_dir,
    get_logger,
    install_mujoco_warning_handler,
    reset_logging_state,
    setup_logging,
)


def test_default_log_dir_points_to_guinsoo_mujoco_logs():
    assert default_log_dir() == Path.home() / ".guinsoo_mujoco" / "logs"


def test_setup_logging_creates_log_file(tmp_path: Path):
    reset_logging_state()
    log_path = setup_logging(
        app_name="test_app",
        level=logging.DEBUG,
        enable_console=False,
        log_dir=tmp_path,
        force=True,
    )

    assert log_path.exists()
    assert log_path.parent == tmp_path
    assert log_path.name.startswith("test_app_")
    assert log_path.suffix == ".log"

    logger = get_logger("sim")
    logger.info("hello from test")
    content = log_path.read_text(encoding="utf-8")
    assert "guinsoo.sim" in content
    assert "hello from test" in content


def test_get_logger_prefixes_guinsoo_namespace():
    logger = get_logger("controller")
    assert logger.name == "guinsoo.controller"


def test_install_mujoco_warning_handler_is_idempotent():
    install_mujoco_warning_handler()
    install_mujoco_warning_handler()
