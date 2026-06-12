from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.astype(float)
    return vec / norm


@dataclass(frozen=True)
class SineSheetSurface:
    """Parametric sine wave sheet z(x) = z0 + amp * sin(2*pi*x / wavelength).

    Path parameter ``s`` advances along ``direction``: world x = x0 + direction * s.
    Use ``direction=-1`` to wipe back toward the robot base for a more natural reach.
    """

    x0: float
    y0: float
    z0: float
    amplitude: float
    wavelength: float
    direction: float = 1.0

    def _x_at(self, s: float) -> float:
        return float(self.x0 + self.direction * s)

    def height(self, x: float) -> float:
        return float(
            self.z0 + self.amplitude * np.sin(2.0 * np.pi * x / self.wavelength)
        )

    def slope(self, x: float) -> float:
        return float(
            self.amplitude
            * (2.0 * np.pi / self.wavelength)
            * np.cos(2.0 * np.pi * x / self.wavelength)
        )

    def position(self, s: float) -> np.ndarray:
        x = self._x_at(s)
        return np.array([x, self.y0, self.height(x)], dtype=float)

    def normal(self, s: float) -> np.ndarray:
        x = self._x_at(s)
        dz_dx = self.slope(x)
        return _normalize(np.array([-dz_dx, 0.0, 1.0], dtype=float))

    def tangent(self, s: float) -> np.ndarray:
        x = self._x_at(s)
        dz_dx = self.slope(x)
        return _normalize(
            np.array([self.direction, 0.0, self.direction * dz_dx], dtype=float)
        )

    def orientation(self, s: float) -> np.ndarray:
        """Rotation matrix for attachment_site: +Z points into the surface (-n)."""
        n = self.normal(s)
        tool_z = -n
        t = self.tangent(s)
        y_axis = _normalize(np.cross(tool_z, t))
        t_axis = _normalize(np.cross(y_axis, tool_z))
        return np.column_stack([t_axis, y_axis, tool_z])
