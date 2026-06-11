import numpy as np
import pytest

from guinsoo_mujoco.demos.ur5e.surface_wipe.config import SURFACE, WIPE_LENGTH
from guinsoo_mujoco.operators.surface import (
    SineSheetSurface,
    build_sine_sheet_hfield,
    interpolate_hfield_height,
)


def test_build_sine_sheet_hfield_dimensions():
    payload = build_sine_sheet_hfield(SURFACE, wipe_length=WIPE_LENGTH)
    values = payload.elevation_str.split()
    assert len(values) == payload.nrow * payload.ncol
    assert payload.size[0] > WIPE_LENGTH / 2.0
    assert payload.body_pos[0] == pytest.approx(SURFACE.x0 + WIPE_LENGTH / 2.0)
    assert payload.size[3] > 0.0


def test_sine_hfield_matches_analytic_surface():
    surface = SineSheetSurface(
        x0=0.30,
        y0=-0.28,
        z0=0.36,
        amplitude=0.015,
        wavelength=0.12,
    )
    payload = build_sine_sheet_hfield(surface, wipe_length=0.25)
    errors = []
    for s in np.linspace(0.0, 0.25, 21):
        x = surface.x0 + float(s)
        z_analytic = surface.height(x)
        z_hfield = interpolate_hfield_height(payload, x, surface.y0)
        errors.append(abs(z_hfield - z_analytic))
    assert max(errors) < 0.002


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not installed",
)
def test_surface_wipe_scene_template_renders(tmp_path):
    from guinsoo_mujoco.assets import AssetManifest, repo_root
    from guinsoo_mujoco.demos.registry import create_demo_registry

    registry = create_demo_registry()
    spec = registry.get("ur5e", "surface_wipe")
    manifest = AssetManifest.load(repo_root() / "assets/robots/ur5e.json")
    cache_root = tmp_path / "cache"
    asset_dir = cache_root / manifest.cache_subdir
    asset_dir.mkdir(parents=True)
    (asset_dir / "ur5e.xml").write_text("<mujoco/>", encoding="utf-8")

    scene_path = spec.resolve_scene(manifest, cache_root=cache_root)
    content = scene_path.read_text(encoding="utf-8")
    assert "{{HFIELD_ELEVATION}}" not in content
    assert "size=\"0.15 0.12" in content
    assert "elevation=\"" in content
