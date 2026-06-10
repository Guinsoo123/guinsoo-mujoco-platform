from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    PLAN = "plan"
    TRACK = "track"
    HOLD = "hold"
