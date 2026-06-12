from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guinsoo_mujoco.operators.surface.sine_sheet import SineSheetSurface


@dataclass(frozen=True)
class SineSheetHfield:
    """MuJoCo hfield payload derived from a :class:`SineSheetSurface`."""

    elevation_str: str
    nrow: int
    ncol: int
    size: tuple[float, float, float, float]
    body_pos: tuple[float, float, float]

    @property
    def size_str(self) -> str:
        sx, sy, height, base = self.size
        return f"{sx} {sy} {height} {base}"

    @property
    def body_pos_str(self) -> str:
        return " ".join(str(value) for value in self.body_pos)


def build_sine_sheet_hfield(
    surface: SineSheetSurface,
    *,
    wipe_length: float,
    x_margin: float = 0.025,
    y_half: float = 0.12,
    ncol: int = 51,
    nrow: int = 3,
    z_padding: float = 0.001,
) -> SineSheetHfield:
    """Build hfield elevation/size/body pose from the analytic sine sheet.

    MuJoCo stores elevation in row-major order over an ``nrow x ncol`` matrix
    where rows span the y-axis and columns span the x-axis. World height at each
    grid point follows ``body_z + base + elevation * height``.
    """
    if wipe_length <= 0.0:
        raise ValueError("wipe_length must be positive")
    if ncol < 2 or nrow < 1:
        raise ValueError("ncol must be >= 2 and nrow must be >= 1")

    half_x = wipe_length / 2.0 + x_margin
    body_x = surface.x0 + surface.direction * wipe_length / 2.0
    body_y = surface.y0
    x_local = np.linspace(-half_x, half_x, ncol)
    z_world = np.array(
        [surface.height(body_x + float(x_offset)) for x_offset in x_local],
        dtype=float,
    )

    base_extent = max(z_padding, 1e-4)
    z_base_world = float(np.min(z_world)) - base_extent
    height = float(np.max(z_world) - np.min(z_world) + base_extent)
    normalized = [
        float(
            np.clip(
                (float(z_value) - z_base_world - base_extent) / height,
                0.0,
                1.0,
            )
        )
        for z_value in z_world
    ]
    elevations: list[float] = []
    for _row in range(nrow):
        elevations.extend(normalized)

    return SineSheetHfield(
        elevation_str=" ".join(f"{value:.6f}" for value in elevations),
        nrow=nrow,
        ncol=ncol,
        size=(half_x, y_half, height, base_extent),
        body_pos=(body_x, body_y, z_base_world),
    )


def interpolate_hfield_height(
    payload: SineSheetHfield,
    x_world: float,
    y_world: float,
) -> float:
    """Bilinear height sample from generated hfield metadata."""
    sx, sy, height, base = payload.size
    body_x, body_y, body_z = payload.body_pos
    ncol = payload.ncol
    nrow = payload.nrow
    values = np.fromstring(payload.elevation_str, sep=" ", dtype=float).reshape(
        nrow, ncol
    )
    lx = float(x_world) - body_x
    ly = float(y_world) - body_y
    u = (lx + sx) / (2.0 * sx) * (ncol - 1)
    v = (ly + sy) / (2.0 * sy) * (nrow - 1)
    col = int(np.floor(u))
    row = int(np.floor(v))
    col = int(np.clip(col, 0, ncol - 2))
    row = int(np.clip(row, 0, nrow - 2))
    du = u - col
    dv = v - row
    elevation = (
        (1.0 - du) * (1.0 - dv) * values[row, col]
        + du * (1.0 - dv) * values[row, col + 1]
        + (1.0 - du) * dv * values[row + 1, col]
        + du * dv * values[row + 1, col + 1]
    )
    return float(body_z + base + elevation * height)
