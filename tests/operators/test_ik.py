import numpy as np

from guinsoo_mujoco.operators.ik.damped_least_squares import _orientation_error


def test_orientation_error_is_zero_for_identity():
    current = np.eye(3)
    target = np.eye(3)
    err = _orientation_error(current, target)
    np.testing.assert_allclose(err, np.zeros(3), atol=1e-6)
