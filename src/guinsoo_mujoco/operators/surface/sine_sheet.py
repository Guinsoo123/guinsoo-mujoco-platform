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
    """Parametric sine wave sheet z(x) = z0 + amp * sin(2*pi*x / wavelength)."""

    x0: float
    y0: float
    z0: float
    amplitude: float
    wavelength: float

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
        x = self.x0 + s
        return np.array([x, self.y0, self.height(x)], dtype=float)

    def normal(self, s: float) -> np.ndarray:
        x = self.x0 + s
        dz_dx = self.slope(x)
        return _normalize(np.array([-dz_dx, 0.0, 1.0], dtype=float))

    def tangent(self, s: float) -> np.ndarray:
        x = self.x0 + s
        dz_dx = self.slope(x)
        return _normalize(np.array([1.0, 0.0, dz_dx], dtype=float))

    def orientation(self, s: float) -> np.ndarray:
        """Rotation matrix for attachment_site: +Z aligned to surface outward normal."""
        n = self.normal(s)
        t = self.tangent(s)
        y_axis = _normalize(np.cross(n, t))
        t_axis = _normalize(np.cross(y_axis, n))
        return np.column_stack([t_axis, y_axis, n])
