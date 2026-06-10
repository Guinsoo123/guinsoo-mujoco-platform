from __future__ import annotations

from typing import Protocol

import numpy as np


class RuntimeLike(Protocol):
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        ...

    def set_control(self, control: np.ndarray) -> None:
        ...


class Controller(Protocol):
    name: str

    def reset(self, runtime: RuntimeLike, config: dict | None = None) -> None:
        ...

    def step(self, runtime: RuntimeLike, t: float, dt: float) -> dict:
        ...
