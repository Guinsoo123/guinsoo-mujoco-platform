from __future__ import annotations


def default_joint_target(dof: int) -> list[float]:
    if dof == 6:
        return [0.0, -1.2, 1.8, -0.8, -1.57, 0.0]
    return [0.0] * dof
