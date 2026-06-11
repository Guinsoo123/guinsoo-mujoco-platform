from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    FOLLOW = "follow"
    RETRACT = "retract"
    DONE = "done"
