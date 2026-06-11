import numpy as np
import pytest

from guinsoo_mujoco.operators.surface import SineSheetSurface


def test_sine_surface_normal_and_tangent_are_orthogonal():
    surface = SineSheetSurface(
        x0=0.4,
        y0=0.0,
        z0=0.45,
        amplitude=0.015,
        wavelength=0.12,
    )
    for s in (0.0, 0.05, 0.12, 0.2):
        n = surface.normal(s)
        t = surface.tangent(s)
        assert abs(np.dot(n, t)) < 1e-6
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6
        assert abs(np.linalg.norm(t) - 1.0) < 1e-6


def test_sine_surface_position_follows_path_parameter():
    surface = SineSheetSurface(
        x0=0.4,
        y0=-0.05,
        z0=0.45,
        amplitude=0.01,
        wavelength=0.1,
    )
    p0 = surface.position(0.0)
    p1 = surface.position(0.025)
    assert p0[0] == 0.4
    assert p1[0] == pytest.approx(0.425)
    assert p0[1] == -0.05
    assert p1[2] != p0[2]


def test_sine_surface_orientation_has_tool_z_pointing_into_surface():
    surface = SineSheetSurface(
        x0=0.4,
        y0=0.0,
        z0=0.45,
        amplitude=0.015,
        wavelength=0.12,
    )
    rot = surface.orientation(0.08)
    n = surface.normal(0.08)
    tool_z = rot[:, 2]
    np.testing.assert_allclose(tool_z, -n, atol=1e-6)
    assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-6)
