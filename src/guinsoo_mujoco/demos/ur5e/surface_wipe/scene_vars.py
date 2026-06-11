from __future__ import annotations

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import SURFACE, WIPE_LENGTH
from guinsoo_mujoco.operators.surface import build_sine_sheet_hfield


def surface_wipe_scene_template_vars() -> dict[str, str]:
    payload = build_sine_sheet_hfield(SURFACE, wipe_length=WIPE_LENGTH)
    start = SURFACE.position(0.0)
    end = SURFACE.position(WIPE_LENGTH)
    stand_z = payload.body_pos[2] - 0.0425
    return {
        "HFIELD_ELEVATION": payload.elevation_str,
        "HFIELD_SIZE": payload.size_str,
        "HFIELD_NROW": str(payload.nrow),
        "HFIELD_NCOL": str(payload.ncol),
        "WAVE_BODY_POS": payload.body_pos_str,
        "WAVE_STAND_POS": f"{payload.body_pos[0]} {payload.body_pos[1]} {stand_z}",
        "WIPE_START_POS": f"{start[0]} {start[1]} {start[2] + 0.02}",
        "WIPE_END_POS": f"{end[0]} {end[1]} {end[2] + 0.02}",
    }
