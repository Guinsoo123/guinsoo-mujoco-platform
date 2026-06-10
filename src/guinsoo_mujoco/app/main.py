from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from guinsoo_mujoco.app.model import SimStudioModel
from guinsoo_mujoco.app.session import AssetNotReadyError, SimSession
from guinsoo_mujoco.app.viewer import MujocoGLWidget
from guinsoo_mujoco.data import RunRecorder, default_runs_dir
from guinsoo_mujoco.recording import append_step_sample, create_run_recorder


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
        self.setWindowTitle("Guinsoo Sim Studio")
        self.resize(1280, 820)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()

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
                f"{card['display_name']}\n{card['support_label']}\n{card['demos']}"
            )
            item.setData(256, card["robot_id"])
            self.robot_list.addItem(item)
        self.robot_list.currentItemChanged.connect(self._select_robot)
        layout.addWidget(self.robot_list)
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
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.demo_combo = QComboBox()
        self.demo_combo.currentTextChanged.connect(self._select_demo)
        form.addRow("Demo", self.demo_combo)
        layout.addLayout(form)
        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self._toggle_run)
        self.record_button = QPushButton("开始记录")
        self.record_button.clicked.connect(self._toggle_recording)
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self._reset)
        self.reset_camera_button = QPushButton("重置视角")
        self.reset_camera_button.clicked.connect(self.viewer.reset_camera)
        layout.addWidget(self.run_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.reset_camera_button)
        layout.addWidget(QLabel("运行状态"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setText(
            "选择机器人后会尝试加载本地缓存的 MuJoCo 场景。\n"
            "若资产未下载，请运行：python -m guinsoo_mujoco.cli fetch-assets ur5e\n\n"
            f"Episode 默认保存目录：{default_runs_dir()}"
        )
        layout.addWidget(self.log)
        layout.addStretch()
        return panel

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
        self.demo_combo.blockSignals(True)
        self.demo_combo.clear()
        self.demo_combo.addItems(robot.demos)
        self.demo_combo.blockSignals(False)
        self.current_demo = robot.demos[0]
        self._load_session()
        if was_running and self.session is not None:
            self._timer.start(33)
            self.run_button.setText("暂停")
        self.statusBar().showMessage(f"已选择 {robot.display_name}")

    def _select_demo(self, demo: str) -> None:
        if not demo or demo == self.current_demo:
            return
        self._discard_recorder()
        was_running = self._timer.isActive()
        if was_running:
            self._timer.stop()
            self.run_button.setText("运行")
        self.current_demo = demo
        self._load_session()
        if was_running and self.session is not None:
            self._timer.start(33)
            self.run_button.setText("暂停")

    def _load_session(self) -> None:
        self.session = None
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
            self.log.setText(str(exc))
            return
        except Exception as exc:
            self.viewer.set_status(f"{robot.display_name}\n加载失败：{exc}")
            self.log.setText(f"加载失败：{exc}")
            return

        self.viewer.set_runtime(self.session.runtime)
        self.viewer.refresh()
        self.log.setText(
            f"已加载 {robot.display_name}\n"
            f"Demo: {self.current_demo}\n"
            f"场景: {self.session.runtime.model_path}"
        )

    def _toggle_recording(self) -> None:
        if self._recorder is not None:
            artifact = self._recorder.close()
            self._recorder = None
            self.record_button.setText("开始记录")
            csv_hint = ""
            if artifact.plotjuggler_csv_path is not None:
                csv_hint = f"\nPlotJuggler CSV: {artifact.plotjuggler_csv_path}"
            self.log.append(
                f"\n已保存 episode：\nHDF5: {artifact.hdf5_path}\n元数据: {artifact.metadata_path}{csv_hint}"
            )
            self.statusBar().showMessage(f"已保存 episode 到 {artifact.hdf5_path.parent}")
            return
        if self.session is None:
            self._load_session()
        if self.session is None:
            self.statusBar().showMessage("无法记录：场景未加载。")
            return
        if not self._timer.isActive():
            self.statusBar().showMessage("请先点击“运行”开始仿真，再记录 episode。")
            return
        self._recorder = create_run_recorder(self.session)
        self.record_button.setText("停止并保存")
        self.statusBar().showMessage(f"正在记录到 {default_runs_dir()}")

    def _toggle_run(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.run_button.setText("运行")
            self.viewer.set_simulation_running(False)
            self.statusBar().showMessage("仿真已暂停。")
            return
        if self.session is None:
            self._load_session()
        if self.session is None:
            self.statusBar().showMessage("无法运行：场景未加载。")
            return
        selection = self.model.select(self.current_robot_id, self.current_demo)
        self._timer.start(33)
        self.run_button.setText("暂停")
        self.viewer.set_simulation_running(True)
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
        if self.session is not None:
            self.session.reset()
            self.viewer.refresh()
        self.statusBar().showMessage("仿真已重置。")

    def _tick(self) -> None:
        if self.session is None:
            return
        self.viewer.apply_perturbation(self.session.runtime, sim_running=True)
        sample: dict | None = None
        for _ in range(self._steps_per_tick):
            sample = self.session.step(self._sim_dt)
            if self._recorder is not None and sample is not None:
                append_step_sample(self._recorder, self.session.runtime, sample)
        if sample is None:
            return
        t = float(sample["time"])
        control = sample["control"]
        metric = float(control[0]) if control.size else 0.0
        self._plot_t.append(t)
        self._plot_y.append(metric)
        self.curve.setData(self._plot_t[-300:], self._plot_y[-300:])
        self.viewer.refresh()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
