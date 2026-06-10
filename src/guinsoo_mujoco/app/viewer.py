from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from guinsoo_mujoco.runtime import MuJoCoRuntime


def qt_to_mj_rel_position(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    width = max(width, 1)
    height = max(height, 1)
    return x / width, 1.0 - (y / height)


def qt_to_mj_rel_delta(dx: float, dy: float, width: int, height: int) -> tuple[float, float]:
    width = max(width, 1)
    height = max(height, 1)
    return dx / width, -dy / height


class MujocoGLWidget(QWidget):
    """Qt viewport with MuJoCo offscreen rendering and simulate-style mouse controls."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._status = "等待加载 MuJoCo 场景"
        self._runtime: MuJoCoRuntime | None = None
        self._renderer = None
        self._camera = None
        self._vopt = None
        self._perturb = None
        self._renderer_size: tuple[int, int] | None = None
        self._frame: np.ndarray | None = None
        self._frame_image: QImage | None = None
        self._drag_button: Qt.MouseButton | None = None
        self._drag_ctrl = False
        self._last_mouse_pos: tuple[float, float] | None = None
        self._drag_mode: str | None = None
        self._sim_running = False
        self.setMinimumSize(720, 420)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def set_runtime(self, runtime: MuJoCoRuntime | None) -> None:
        self._release_mouse_state()
        self._dispose_renderer()
        self._runtime = runtime
        self._frame = None
        self._frame_image = None
        self._camera = None
        self._vopt = None
        self._perturb = None
        if runtime is not None:
            self._status = ""
            self._init_interaction_state()
            self.refresh()
        else:
            self.update()

    def set_status(self, text: str) -> None:
        self._status = text
        self.set_runtime(None)

    def reset_camera(self) -> None:
        if self._runtime is None:
            return
        self._init_camera()
        self.refresh()

    def set_simulation_running(self, running: bool) -> None:
        self._sim_running = running

    def apply_perturbation(self, runtime: MuJoCoRuntime, sim_running: bool) -> None:
        if self._perturb is None or not self._perturb.active:
            return
        import mujoco

        if sim_running:
            mujoco.mjv_applyPerturbForce(runtime.model, runtime.data, self._perturb)
        else:
            mujoco.mjv_applyPerturbPose(runtime.model, runtime.data, self._perturb, 1)

    def refresh(self) -> None:
        if self._runtime is not None:
            try:
                self._render_scene()
                self._frame_image = self._frame_to_image(self._frame)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                self._dispose_renderer()
                self._status = f"渲染失败：{exc}"
                self._frame_image = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._runtime is None:
            return
        size = (max(self.width(), 1), max(self.height(), 1))
        if self._renderer_size != size:
            self._dispose_renderer()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._runtime is None:
            super().mousePressEvent(event)
            return
        button = event.button()
        if button not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            super().mousePressEvent(event)
            return
        self._drag_button = button
        self._drag_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._last_mouse_pos = (float(event.position().x()), float(event.position().y()))
        if self._drag_ctrl and button == Qt.MouseButton.LeftButton:
            self._begin_perturb(event)
            self._drag_mode = "perturb"
        else:
            self._drag_mode = "camera"
        self.grabMouse()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._runtime is None or self._drag_button is None or self._last_mouse_pos is None:
            super().mouseMoveEvent(event)
            return
        pos = (float(event.position().x()), float(event.position().y()))
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        if dx == 0.0 and dy == 0.0:
            event.accept()
            return
        if self._drag_mode == "perturb":
            self._move_perturb(dx, dy)
        else:
            self._move_camera(self._drag_button, dx, dy)
        self._last_mouse_pos = pos
        self.refresh()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_button is not None and event.button() == self._drag_button:
            self._release_mouse_state()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._runtime is None:
            super().wheelEvent(event)
            return
        import mujoco

        self._ensure_interaction_scene()
        assert self._camera is not None
        assert self._renderer is not None
        delta = event.angleDelta().y() / 120.0
        mujoco.mjv_moveCamera(
            self._runtime.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * delta,
            self._renderer.scene,
            self._camera,
        )
        self.refresh()
        event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(13, 15, 18))
        if self._frame_image is not None and not self._frame_image.isNull():
            painter.drawImage(self.rect(), self._frame_image)
        elif self._status:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QColor("#dce3ea"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status)
        painter.end()

    def _init_interaction_state(self) -> None:
        import mujoco

        self._init_camera()
        self._vopt = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()

    def _init_camera(self) -> None:
        if self._runtime is None:
            return
        import mujoco

        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        center = np.asarray(self._runtime.model.stat.center, dtype=float)
        self._camera.lookat[:] = center
        self._camera.distance = max(float(self._runtime.model.stat.extent) * 2.5, 1.5)
        self._camera.azimuth = 120.0
        self._camera.elevation = -20.0

    def _ensure_renderer(self) -> None:
        if self._runtime is None:
            return
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        size = (width, height)
        if self._renderer is not None and self._renderer_size == size:
            return
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_size = None
        import mujoco

        if self._camera is None:
            self._init_camera()
        self._runtime.ensure_offscreen_size(width, height)
        self._renderer = mujoco.Renderer(self._runtime.model, height=height, width=width)

    def _ensure_interaction_scene(self) -> None:
        self._ensure_renderer()
        assert self._renderer is not None
        assert self._camera is not None
        self._renderer.update_scene(self._runtime.data, camera=self._camera)

    def _render_scene(self) -> None:
        self._ensure_interaction_scene()
        assert self._renderer is not None
        self._frame = self._renderer.render()

    def _aspect_ratio(self) -> float:
        return max(self.width(), 1) / max(self.height(), 1)

    def _rel_position(self, x: float, y: float) -> tuple[float, float]:
        return qt_to_mj_rel_position(x, y, self.width(), self.height())

    def _rel_delta(self, dx: float, dy: float) -> tuple[float, float]:
        return qt_to_mj_rel_delta(dx, dy, self.width(), self.height())

    def _move_camera(
        self, button: Qt.MouseButton, dx: float, dy: float
    ) -> None:
        import mujoco

        self._ensure_interaction_scene()
        assert self._renderer is not None
        assert self._camera is not None
        reldx, rely = self._rel_delta(dx, dy)
        if button == Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif button == Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(
            self._runtime.model,
            action,
            reldx,
            rely,
            self._renderer.scene,
            self._camera,
        )

    def _begin_perturb(self, event: QMouseEvent) -> None:
        import mujoco

        self._ensure_interaction_scene()
        assert self._renderer is not None
        assert self._vopt is not None
        assert self._perturb is not None
        relx, rely = self._rel_position(
            float(event.position().x()), float(event.position().y())
        )
        selpnt = np.zeros((3, 1))
        geomid = np.array([[-1]], dtype=np.int32)
        flexid = np.array([[-1]], dtype=np.int32)
        skinid = np.array([[-1]], dtype=np.int32)
        bodyid = mujoco.mjv_select(
            self._runtime.model,
            self._runtime.data,
            self._vopt,
            self._aspect_ratio(),
            relx,
            rely,
            self._renderer.scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )
        self._perturb.select = bodyid
        if bodyid >= 0:
            mujoco.mjv_initPerturb(
                self._runtime.model,
                self._runtime.data,
                self._renderer.scene,
                self._perturb,
            )

    def _move_perturb(self, dx: float, dy: float) -> None:
        if self._perturb is None or not self._perturb.active:
            return
        import mujoco

        self._ensure_interaction_scene()
        assert self._renderer is not None
        reldx, rely = self._rel_delta(dx, dy)
        mujoco.mjv_movePerturb(
            self._runtime.model,
            self._runtime.data,
            mujoco.mjtMouse.mjMOUSE_MOVE_V,
            reldx,
            rely,
            self._renderer.scene,
            self._perturb,
        )
        if not self._sim_running:
            mujoco.mjv_applyPerturbPose(
                self._runtime.model, self._runtime.data, self._perturb, 1
            )

    def _release_mouse_state(self) -> None:
        if self._perturb is not None:
            self._perturb.active = 0
        self._drag_button = None
        self._drag_ctrl = False
        self._last_mouse_pos = None
        self._drag_mode = None
        if self.mouseGrabber() is self:
            self.releaseMouse()

    @staticmethod
    def _frame_to_image(frame: np.ndarray | None) -> QImage | None:
        if frame is None:
            return None
        height, width, channels = frame.shape
        if channels != 3:
            return None
        buffer = np.ascontiguousarray(frame)
        return QImage(
            buffer.data,
            width,
            height,
            3 * width,
            QImage.Format.Format_RGB888,
        ).copy()

    def _dispose_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._renderer_size = None
        self._frame = None

    def closeEvent(self, event) -> None:
        self._release_mouse_state()
        self._dispose_renderer()
        super().closeEvent(event)
