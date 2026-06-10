import numpy as np

from guinsoo_mujoco.operators.collision import CollisionModel, is_edge_colliding


class FakeCollisionRuntime:
    def __init__(self):
        self.qpos = np.zeros(2)
        self.ncon = 0
        self.contact = []

    @property
    def data(self):
        return self

    @property
    def model(self):
        return self

    def set_joint_positions(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)

    def forward(self, qpos=None):
        if qpos is not None:
            self.set_joint_positions(qpos)


def test_is_edge_colliding_detects_blocked_midpoint(monkeypatch):
    runtime = FakeCollisionRuntime()
    model = CollisionModel(
        robot_body_names=frozenset(),
        ignore_body_names=frozenset(),
        obstacle_geom_names=(),
    )

    def _colliding(_runtime, q, _model):
        return float(q[0]) > 0.5

    monkeypatch.setattr(
        "guinsoo_mujoco.operators.collision.checker.is_configuration_colliding",
        _colliding,
    )

    assert is_edge_colliding(
        runtime,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        model,
        samples=5,
    )
