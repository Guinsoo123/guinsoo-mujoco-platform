from __future__ import annotations

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QPainter
from PySide6.QtCore import Qt


class MujocoGLWidget(QOpenGLWidget):
    """Qt OpenGL viewport placeholder for MuJoCo's native renderer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._status = "等待加载 MuJoCo 场景"
        self.setMinimumSize(720, 420)

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def initializeGL(self) -> None:
        from OpenGL import GL

        GL.glClearColor(0.05, 0.06, 0.07, 1.0)

    def paintGL(self) -> None:
        from OpenGL import GL

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("#dce3ea"))
        painter.drawText(self.rect(), Qt.AlignCenter, self._status)
        painter.end()
