from guinsoo_mujoco.app.viewer import qt_to_mj_rel_delta, qt_to_mj_rel_position


def test_qt_to_mj_coordinate_mapping():
    relx, rely = qt_to_mj_rel_position(500, 0, 1000, 500)
    rdx, rdy = qt_to_mj_rel_delta(100, -50, 1000, 500)

    assert relx == 0.5
    assert rely == 1.0
    assert rdx == 0.1
    assert rdy == 0.1
