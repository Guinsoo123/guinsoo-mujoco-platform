from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_force_on_normal(force: np.ndarray, normal: np.ndarray) -> float:
    n = np.asarray(normal, dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return 0.0
    return float(np.dot(np.asarray(force, dtype=float), n / n_norm))


@dataclass
class NormalAdmittanceState:
    d_n: float = 0.0
    d_n_dot: float = 0.0
    filtered_force: float = 0.0


@dataclass(frozen=True)
class NormalAdmittance:
    mass: float = 1.0
    damping: float = 50.0
    stiffness: float = 0.0
    force_des: float = 10.0
    d_n_limit: float = 0.02
    force_lpf_alpha: float = 0.2

    def reset(self) -> NormalAdmittanceState:
        return NormalAdmittanceState()

    def step(
        self,
        state: NormalAdmittanceState,
        force_normal: float,
        dt: float,
    ) -> tuple[NormalAdmittanceState, float]:
        if dt <= 0.0:
            return state, state.d_n

        alpha = np.clip(self.force_lpf_alpha, 0.0, 1.0)
        filtered = (1.0 - alpha) * state.filtered_force + alpha * force_normal
        force_error = filtered - self.force_des
        mass = max(self.mass, 1e-6)
        d_n_ddot = (force_error - self.damping * state.d_n_dot - self.stiffness * state.d_n) / mass
        d_n_dot = state.d_n_dot + d_n_ddot * dt
        d_n = state.d_n + d_n_dot * dt
        limit = abs(self.d_n_limit)
        d_n = float(np.clip(d_n, -limit, limit))
        return NormalAdmittanceState(d_n=d_n, d_n_dot=d_n_dot, filtered_force=filtered), d_n
