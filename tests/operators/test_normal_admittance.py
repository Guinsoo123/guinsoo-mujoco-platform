from guinsoo_mujoco.operators.admittance import NormalAdmittance, project_force_on_normal


def test_project_force_on_normal():
    force = [0.0, 2.0, 10.0]
    normal = [0.0, 0.0, 1.0]
    assert abs(project_force_on_normal(force, normal) - 10.0) < 1e-9


def test_normal_admittance_moves_toward_force_setpoint():
    admittance = NormalAdmittance(
        mass=1.0,
        damping=20.0,
        stiffness=0.0,
        force_des=10.0,
        d_n_limit=0.05,
        force_lpf_alpha=1.0,
    )
    state = admittance.reset()
    d_n_values = []
    for _ in range(200):
        state, d_n = admittance.step(state, force_normal=20.0, dt=0.002)
        d_n_values.append(d_n)
    assert d_n_values[-1] > d_n_values[0]
    assert abs(d_n_values[-1]) <= 0.05


def test_normal_admittance_respects_output_limit():
    admittance = NormalAdmittance(
        mass=0.5,
        damping=5.0,
        stiffness=0.0,
        force_des=0.0,
        d_n_limit=0.01,
        force_lpf_alpha=1.0,
    )
    state = admittance.reset()
    for _ in range(500):
        state, d_n = admittance.step(state, force_normal=100.0, dt=0.002)
    assert abs(d_n) <= 0.01 + 1e-9
