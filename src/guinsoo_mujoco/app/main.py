from __future__ import annotations

import logging
import sys
import traceback

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from guinsoo_mujoco.app.log_handler import QtLogHandler
from guinsoo_mujoco.app.model import SimStudioModel
from guinsoo_mujoco.app.runs_panel import RunsPanel
from guinsoo_mujoco.app.session import AssetNotReadyError, SimSession
from guinsoo_mujoco.app.viewer import MujocoGLWidget
from guinsoo_mujoco.data import RunRecorder, default_runs_dir
from guinsoo_mujoco.logging_config import (
    current_log_file,
    get_logger,
    resolve_log_level,
    setup_logging,
)
from guinsoo_mujoco.recording import append_step_sample, create_run_recorder

logger = get_logger("ui")
_MAX_LOG_LINES = 2000


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimStudioModel()
        self.current_robot_id = "ur5e"
        self.current_demo = "joint_position"
        self.session: SimSession | None = None
        self._recorder: RunRecorder | None = None
        self._sim_dt = 0.002
        self._steps_per_tick = 16
        self._last_controller_status: str | None = None
        self._last_controller_phase: str | None = None
        self._last_controller_waypoint: str | None = None
        self.setWindowTitle("Guinsoo Sim Studio")
        self.resize(1280, 820)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._attach_qt_log_handler()
        logger.info("Sim Studio 界面已就绪，日志面板位于实时曲线下方。")

    def _attach_qt_log_handler(self) -> None:
        root = get_logger("guinsoo")
        self._qt_log_handler = QtLogHandler(level=resolve_log_level())
        self._qt_log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        self._qt_log_handler.log_record.connect(self._append_log_line)
        root.addHandler(self._qt_log_handler)
        for name in ("ui", "sim", "controller", "planner", "mujoco", "cli"):
            get_logger(name).propagate = True

    def _append_log_line(self, line: str) -> None:
        self.log_view.append(line)
        doc = self.log_view.document()
        if doc.blockCount() > _MAX_LOG_LINES:
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def _build_ui(self) -> None:
        splitter = QSplitter()
        robot_panel = self._build_robot_panel()
        center_panel = self._build_center_panel()
        control_panel = self._build_control_panel()
        splitter.addWidget(robot_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(control_panel)
        splitter.setSizes([260, 760, 260])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("选择机器人和 demo 后点击运行。")
        self.robot_list.blockSignals(True)
        self.robot_list.setCurrentRow(0)
        self.robot_list.blockSignals(False)
        QTimer.singleShot(0, self._init_robot_selection)

    def _build_robot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("机器人示例"))
        self.robot_list = QListWidget()
        for card in self.model.robot_cards():
            item = QListWidgetItem(
                f"{card['display_name']}\n{card['support_label']}"
            )
            item.setData(256, card["robot_id"])
            self.robot_list.addItem(item)
        self.robot_list.currentItemChanged.connect(self._select_robot)
        layout.addWidget(self.robot_list)

        layout.addWidget(QLabel("Demo 场景"))
        self.demo_list = QListWidget()
        self.demo_list.currentItemChanged.connect(self._select_demo_item)
        layout.addWidget(self.demo_list)

        self.demo_description = QLabel("选择 Demo 查看说明。")
        self.demo_description.setWordWrap(True)
        layout.addWidget(self.demo_description)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(
            QLabel("左键旋转 | 右键平移 | 滚轮缩放 | Ctrl+左键拖动物体")
        )
        self.viewer = MujocoGLWidget()
        layout.addWidget(self.viewer, stretch=4)
        self.plot = pg.PlotWidget(title="实时控制曲线")
        self.plot.setLabel("left", "控制量")
        self.plot.setLabel("bottom", "时间", units="s")
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#45a3ff", width=2))
        self._plot_t: list[float] = []
        self._plot_y: list[float] = []
        layout.addWidget(self.plot, stretch=1)

        log_file = current_log_file()
        log_hint = (
            str(log_file.parent)
            if log_file is not None
            else "~/.guinsoo_mujoco/logs/<timestamp>/"
        )
        layout.addWidget(QLabel(f"运行日志（文件：{log_hint}）"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        layout.addWidget(self.log_view, stretch=1)
        return panel

    def _build_control_panel(self) -> QWidget:
        tabs = QTabWidget()
        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self._toggle_run)
        self.record_button = QPushButton("开始记录")
        self.record_button.clicked.connect(self._toggle_recording)
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self._reset)
        self.reset_camera_button = QPushButton("重置视角")
        self.reset_camera_button.clicked.connect(self.viewer.reset_camera)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.record_button)
        run_layout.addWidget(self.reset_button)
        run_layout.addWidget(self.reset_camera_button)
        run_layout.addWidget(QLabel("运行状态"))
        self.status_summary = QLabel("等待加载场景。")
        self.status_summary.setWordWrap(True)
        run_layout.addWidget(self.status_summary)
        run_layout.addWidget(
            QLabel(f"Episode 默认保存目录：{default_runs_dir()}")
        )
        tabs.addTab(run_tab, "运行控制")

        self.runs_panel = RunsPanel()
        tabs.addTab(self.runs_panel, "数据包")
        return tabs

    def _set_status_summary(self, text: str) -> None:
        self.status_summary.setText(text)

    def _init_robot_selection(self) -> None:
        self._select_robot(self.robot_list.currentItem())

    def _select_robot(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        self._discard_recorder()
        was_running = self._timer.isActive()
        if was_running:
            self._timer.stop()
            self.run_button.setText("运行")
        self.current_robot_id = current.data(256)
        robot = self.model.registry.get(self.current_robot_id)
        logger.info("选择机器人：%s", robot.display_name)
        self._populate_demo_list(robot.robot_id)
        self._load_session()
        if was_running and self.session is not None:
            self._timer.start(33)
            self.run_button.setText("暂停")
        self.statusBar().showMessage(f"已选择 {robot.display_name}")

    def _populate_demo_list(self, robot_id: str) -> None:
        cards = self.model.demo_cards(robot_id)
        self.demo_list.blockSignals(True)
        self.demo_list.clear()
        for card in cards:
            item = QListWidgetItem(card["display_name"])
            item.setData(256, card["demo_id"])
            item.setToolTip(card["description"])
            self.demo_list.addItem(item)
        if cards:
            self.demo_list.setCurrentRow(0)
            self.current_demo = cards[0]["demo_id"]
            self.demo_description.setText(cards[0]["description"])
        self.demo_list.blockSignals(False)

    def _select_demo_item(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        demo = current.data(256)
        cards = self.model.demo_cards(self.current_robot_id)
        for card in cards:
            if card["demo_id"] == demo:
                self.demo_description.setText(card["description"])
                break
        self._select_demo(demo)

    def _select_demo(self, demo: str) -> None:
        if not demo or demo == self.current_demo:
            return
        self._discard_recorder()
        was_running = self._timer.isActive()
        if was_running:
            self._timer.stop()
            self.run_button.setText("运行")
        self.current_demo = demo
        logger.info("切换 Demo：%s", demo)
        self._load_session()
        if was_running and self.session is not None:
            self._timer.start(33)
            self.run_button.setText("暂停")

    def _load_session(self) -> None:
        self.session = None
        self._reset_controller_log_state()
        robot = self.model.registry.get(self.current_robot_id)
        try:
            self.session = SimSession.load(
                self.current_robot_id,
                self.current_demo,
            )
        except AssetNotReadyError as exc:
            self.viewer.set_status(
                f"{robot.display_name}\n资产未就绪\n\n{SimSession.fetch_hint(robot.robot_id)}"
            )
            logger.error("资产未就绪：%s", exc)
            self._set_status_summary(str(exc))
            return
        except Exception as exc:
            self.viewer.set_status(f"{robot.display_name}\n加载失败：{exc}")
            logger.error("场景加载失败：%s", exc, exc_info=True)
            self._set_status_summary(f"加载失败：{exc}")
            return

        self.viewer.set_runtime(self.session.runtime)
        self.viewer.refresh()
        demo_cards = {
            card["demo_id"]: card["display_name"]
            for card in self.model.demo_cards(self.current_robot_id)
        }
        demo_label = demo_cards.get(self.current_demo, self.current_demo)
        summary = (
            f"已加载 {robot.display_name}\n"
            f"Demo: {demo_label}\n"
            f"场景: {self.session.runtime.model_path}"
        )
        self._set_status_summary(summary)
        logger.info(
            "场景加载成功：robot=%s demo=%s scene=%s",
            robot.robot_id,
            self.current_demo,
            self.session.runtime.model_path,
        )

    def _toggle_recording(self) -> None:
        if self._recorder is not None:
            artifact = self._recorder.close()
            self._recorder = None
            self.record_button.setText("开始记录")
            csv_hint = ""
            if artifact.plotjuggler_csv_path is not None:
                csv_hint = f"\nPlotJuggler CSV: {artifact.plotjuggler_csv_path}"
            message = (
                f"已保存 episode：\nHDF5: {artifact.hdf5_path}\n"
                f"元数据: {artifact.metadata_path}{csv_hint}"
            )
            self._set_status_summary(message)
            logger.info("Episode 已保存：%s", artifact.hdf5_path.parent)
            self.statusBar().showMessage(f"已保存 episode 到 {artifact.hdf5_path.parent}")
            self.runs_panel.refresh()
            return
        if self.session is None:
            self._load_session()
        if self.session is None:
            logger.warning("无法记录：场景未加载")
            self.statusBar().showMessage("无法记录：场景未加载。")
            return
        if not self._timer.isActive():
            logger.warning("无法记录：仿真未运行")
            self.statusBar().showMessage("请先点击“运行”开始仿真，再记录 episode。")
            return
        self._recorder = create_run_recorder(self.session)
        self.record_button.setText("停止并保存")
        logger.info("开始记录 episode，目录：%s", default_runs_dir())
        self.statusBar().showMessage(f"正在记录到 {default_runs_dir()}")

    def _toggle_run(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.run_button.setText("运行")
            self.viewer.set_simulation_running(False)
            logger.info("仿真已暂停")
            self.statusBar().showMessage("仿真已暂停。")
            return
        if self.session is None:
            self._load_session()
        if self.session is None:
            logger.warning("无法运行：场景未加载")
            self.statusBar().showMessage("无法运行：场景未加载。")
            return
        selection = self.model.select(self.current_robot_id, self.current_demo)
        self._timer.start(33)
        self.run_button.setText("暂停")
        self.viewer.set_simulation_running(True)
        logger.info(
            "仿真运行中：%s / %s",
            selection.robot.display_name,
            selection.demo,
        )
        self.statusBar().showMessage(
            f"仿真运行中：{selection.robot.display_name} / {selection.demo}"
        )

    def _discard_recorder(self) -> None:
        if self._recorder is None:
            return
        self._recorder = None
        self.record_button.setText("开始记录")

    def _reset(self) -> None:
        self._discard_recorder()
        self._plot_t.clear()
        self._plot_y.clear()
        self.curve.setData([], [])
        self._reset_controller_log_state()
        if self.session is not None:
            self.session.reset()
            self.viewer.refresh()
            logger.info("仿真已重置")
        self.statusBar().showMessage("仿真已重置。")

    def _reset_controller_log_state(self) -> None:
        self._last_controller_status = None
        self._last_controller_phase = None
        self._last_controller_waypoint = None

    def _log_controller_sample(self, sample: dict) -> None:
        status = sample.get("status")
        phase = sample.get("phase")
        waypoint = sample.get("waypoint")
        if status and status != self._last_controller_status:
            self._last_controller_status = str(status)
            self._set_status_summary(str(status))
            logger.info("控制器状态：%s", status)
        if phase and phase != self._last_controller_phase:
            self._last_controller_phase = str(phase)
            logger.info("控制器阶段：%s", phase)
        if waypoint and waypoint != self._last_controller_waypoint:
            self._last_controller_waypoint = str(waypoint)
            logger.info("当前路点：%s", waypoint)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "step t=%.3f phase=%s waypoint=%s plan_success=%s",
                float(sample.get("time", 0.0)),
                phase,
                waypoint,
                sample.get("plan_success"),
            )

    def _tick(self) -> None:
        if self.session is None:
            return
        try:
            self.viewer.apply_perturbation(self.session.runtime, sim_running=True)
            sample: dict | None = None
            for _ in range(self._steps_per_tick):
                sample = self.session.step(self._sim_dt)
                if self._recorder is not None and sample is not None:
                    append_step_sample(self._recorder, self.session.runtime, sample)
            if sample is None:
                return
            self._log_controller_sample(sample)
            t = float(sample["time"])
            control = sample["control"]
            metric = float(control[0]) if control.size else 0.0
            self._plot_t.append(t)
            self._plot_y.append(metric)
            self.curve.setData(self._plot_t[-300:], self._plot_y[-300:])
            self.viewer.refresh()
        except Exception as exc:
            self._handle_tick_error(exc)

    def _handle_tick_error(self, exc: BaseException) -> None:
        self._timer.stop()
        self.run_button.setText("运行")
        self.viewer.set_simulation_running(False)
        summary = f"仿真异常已暂停：{exc}"
        self._set_status_summary(summary)
        logger.error("仿真步进异常，已自动暂停：%s", exc)
        logger.error(traceback.format_exc())


def _install_exception_logging() -> None:
    def _log_unhandled(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical(
            "未捕获异常：%s",
            exc,
            exc_info=(exc_type, exc, tb),
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_unhandled


def main() -> int:
    setup_logging(app_name="sim_studio")
    _install_exception_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
