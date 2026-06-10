import numpy as np

from guinsoo_mujoco.demos.ur5e.ee_pose_avoid.rrt import RRTConnectPlanner


class FakeCollisionRuntime:
    def __init__(self, dof: int = 2):
        self.model_nq = dof
        self._blocked_center = np.array([0.5, 0.5])
        self._blocked_radius = 0.15

    def joint_limits(self):
        return np.zeros(self.model_nq), np.ones(self.model_nq)

    def set_joint_positions(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)

    def forward(self, qpos=None):
        if qpos is not None:
            self.set_joint_positions(qpos)

    @property
    def model(self):
        return self

    @property
    def nq(self):
        return self.model_nq

    @property
    def data(self):
        return self


def _is_free(runtime: FakeCollisionRuntime, q: np.ndarray) -> bool:
    return float(np.linalg.norm(q - runtime._blocked_center)) > runtime._blocked_radius


def test_rrt_connect_finds_path_around_blocked_region(monkeypatch):
    runtime = FakeCollisionRuntime(dof=2)
    planner = RRTConnectPlanner(
        runtime,
        step_size=0.1,
        goal_bias=0.3,
        max_iterations=2000,
        obstacle_geom_names=(),
        collision_margin=0.03,
    )
    def _colliding(_runtime, q, _obstacles, margin=0.03):
        return not _is_free(runtime, q)

    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.rrt.is_configuration_colliding",
        _colliding,
    )
    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.rrt.is_edge_colliding",
        lambda _runtime, q_from, q_to, _obstacles, margin=0.03, samples=8: not (
            _is_free(runtime, q_from) and _is_free(runtime, q_to)
        ),
    )

    q_start = np.array([0.1, 0.1])
    q_goal = np.array([0.9, 0.9])
    path = planner.plan(q_start, q_goal)

    assert path is not None
    assert len(path) >= 2
    np.testing.assert_allclose(path[0], q_start, atol=0.11)
    np.testing.assert_allclose(path[-1], q_goal, atol=0.11)
    for node in path:
        assert _is_free(runtime, node)


def test_rrt_connect_returns_none_when_start_in_collision(monkeypatch):
    runtime = FakeCollisionRuntime(dof=2)
    planner = RRTConnectPlanner(runtime, max_iterations=10, obstacle_geom_names=())
    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.rrt.is_configuration_colliding",
        lambda _runtime, q, _obstacles, margin=0.03: not _is_free(runtime, q),
    )
    monkeypatch.setattr(
        "guinsoo_mujoco.demos.ur5e.ee_pose_avoid.rrt.is_edge_colliding",
        lambda *_args, **_kwargs: True,
    )

    path = planner.plan(np.array([0.5, 0.5]), np.array([0.9, 0.9]))
    assert path is None
