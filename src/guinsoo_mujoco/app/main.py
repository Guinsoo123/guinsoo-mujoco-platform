from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
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
import pyqtgraph as pg

from guinsoo_mujoco.app.model import SimStudioModel
from guinsoo_mujoco.app.viewer import MujocoGLWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimStudioModel()
        self.current_robot_id = "ur5e"
        self.current_demo = "joint_position"
        self.elapsed = 0.0
        self.setWindowTitle("Guinsoo Sim Studio")
        self.resize(1280, 820)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

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
        self.robot_list.setCurrentRow(0)

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
        self.record_button = QPushButton("记录 episode")
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self._reset)
        layout.addWidget(self.run_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(QLabel("运行状态"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setText(
            "第一版界面骨架已接入机器人注册表、Demo 选择、OpenGL 视图和实时曲线。\n"
            "UR5e 完整仿真会在资产下载和 MuJoCoRuntime 接入后运行。"
        )
        layout.addWidget(self.log)
        layout.addStretch()
        return panel

    def _select_robot(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.current_robot_id = current.data(256)
        robot = self.model.registry.get(self.current_robot_id)
        self.demo_combo.blockSignals(True)
        self.demo_combo.clear()
        self.demo_combo.addItems(robot.demos)
        self.demo_combo.blockSignals(False)
        self.current_demo = robot.demos[0]
        self.viewer.set_status(f"{robot.display_name}\n{robot.description}")
        self.statusBar().showMessage(f"已选择 {robot.display_name}")

    def _select_demo(self, demo: str) -> None:
        if demo:
            self.current_demo = demo

    def _toggle_run(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.run_button.setText("运行")
            self.statusBar().showMessage("仿真已暂停。")
            return
        selection = self.model.select(self.current_robot_id, self.current_demo)
        self.viewer.set_status(
            f"运行中：{selection.robot.display_name}\nDemo: {selection.demo}"
        )
        self._timer.start(33)
        self.run_button.setText("暂停")
        self.statusBar().showMessage("仿真循环运行中。")

    def _reset(self) -> None:
        self.elapsed = 0.0
        self._plot_t.clear()
        self._plot_y.clear()
        self.curve.setData([], [])
        self.viewer.set_status("已重置，等待运行。")

    def _tick(self) -> None:
        self.elapsed += 0.033
        self._plot_t.append(self.elapsed)
        self._plot_y.append((self.elapsed % 1.0) - 0.5)
        self.curve.setData(self._plot_t[-300:], self._plot_y[-300:])


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
