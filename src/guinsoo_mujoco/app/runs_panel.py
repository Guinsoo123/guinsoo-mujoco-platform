from __future__ import annotations

import json
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from guinsoo_mujoco.data import default_runs_dir
from guinsoo_mujoco.plotjuggler_analysis import (
    PlotJugglerNotFoundError,
    launch_motion_control_analysis,
)
from guinsoo_mujoco.run_catalog import RunSummary, delete_run, list_runs


class RunsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._summaries: list[RunSummary] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("已录制数据包"))
        layout.addWidget(QLabel(f"目录：{default_runs_dir()}"))
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_list.currentItemChanged.connect(self._show_details)
        layout.addWidget(self.run_list, stretch=3)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh)
        self.analyze_button = QPushButton("PlotJuggler 分析")
        self.analyze_button.clicked.connect(self._analyze_selected)
        self.reveal_button = QPushButton("在 Finder 显示")
        self.reveal_button.clicked.connect(self._reveal_selected)
        self.delete_button = QPushButton("删除")
        self.delete_button.clicked.connect(self._delete_selected)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.analyze_button)
        buttons.addWidget(self.reveal_button)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("数据包详情"))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, stretch=2)

    def refresh(self) -> None:
        current_stem = self._selected_stem()
        self._summaries = list_runs()
        self.run_list.clear()
        restore_row = 0
        for index, summary in enumerate(self._summaries):
            item = QListWidgetItem(summary.label)
            item.setData(Qt.ItemDataRole.UserRole, summary.stem)
            self.run_list.addItem(item)
            if summary.stem == current_stem:
                restore_row = index
        if self._summaries:
            self.run_list.setCurrentRow(restore_row)
        else:
            self.details.setText("暂无录制数据包。运行仿真后点击“开始记录”。")

    def _selected_summary(self) -> RunSummary | None:
        item = self.run_list.currentItem()
        if item is None:
            return None
        stem = item.data(Qt.ItemDataRole.UserRole)
        for summary in self._summaries:
            if summary.stem == stem:
                return summary
        return None

    def _selected_stem(self) -> str | None:
        item = self.run_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _show_details(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            return
        summary = self._selected_summary()
        if summary is None:
            return
        metadata = json.loads(summary.artifact.metadata_path.read_text(encoding="utf-8"))
        lines = [
            f"名称: {summary.stem}",
            f"机器人: {summary.robot_id}",
            f"Demo: {summary.demo}",
            f"控制器: {summary.controller}",
            f"创建时间: {summary.created_at}",
            f"采样帧数: {summary.sample_count or '未知'}",
            f"HDF5: {summary.artifact.hdf5_path}",
            f"JSON: {summary.artifact.metadata_path}",
        ]
        csv_path = summary.artifact.plotjuggler_csv_path
        if csv_path is not None:
            lines.append(f"PlotJuggler CSV: {csv_path}")
        joints = metadata.get("actuator_names") or metadata.get("joint_names") or []
        if joints:
            lines.append(f"关节: {', '.join(joints)}")
        self.details.setText("\n".join(lines))

    def _analyze_selected(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            QMessageBox.information(self, "PlotJuggler 分析", "请先选择一个数据包。")
            return
        try:
            csv_path, layout_path = launch_motion_control_analysis(summary.artifact)
        except PlotJugglerNotFoundError as exc:
            QMessageBox.warning(self, "PlotJuggler 未安装", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "PlotJuggler 分析失败", str(exc))
            return
        QMessageBox.information(
            self,
            "PlotJuggler 分析",
            "已启动 PlotJuggler，并加载标准运控分析布局。\n"
            "每个页面按 2x3 网格展示 6 个关节窗口：\n"
            "1. 关节位置跟踪（qpos vs target）\n"
            "2. 跟踪误差\n"
            "3. 控制力矩（ctrl vs actuator_force）\n"
            "4. 关节速度\n"
            "5. 关节力矩反馈\n\n"
            f"CSV: {csv_path}\n"
            f"布局: {layout_path}",
        )

    def _reveal_selected(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return
        subprocess.Popen(
            ["open", "-R", str(summary.artifact.hdf5_path)],
            start_new_session=True,
        )

    def _delete_selected(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            QMessageBox.information(self, "删除数据包", "请先选择一个数据包。")
            return
        answer = QMessageBox.question(
            self,
            "删除数据包",
            f"确认删除以下数据包？\n\n{summary.label}\n\n此操作不可恢复。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        delete_run(summary)
        self.refresh()
