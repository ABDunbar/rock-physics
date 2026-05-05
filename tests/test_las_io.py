from pathlib import Path

import numpy as np

from rockphys.las_io import load_las_well


ROOT = Path(__file__).resolve().parents[1]


def test_load_las_well_normalizes_demo_curves():
    well = load_las_well(ROOT / "data/well_2.las")

    assert len(well) == 4117
    assert {"depth", "Vp", "Vs", "rho", "GR", "nphi"}.issubset(well.columns)
    assert np.isclose(well["depth"].min(), 2013.2528)
    assert well[["Vp", "Vs", "rho", "GR"]].notna().all().all()


def test_load_las_well_keeps_existing_workflow_units():
    well = load_las_well(ROOT / "data/well_2.las")

    assert well["Vp"].median() < 4.0
    assert well["Vs"].median() < 3.0
    assert well["rho"].median() < 3.0


def test_load_las_well_can_feed_rock_physics_calculation():
    from rockphys.physics import compute_rock_physics

    well = load_las_well(ROOT / "data/well_2.las")
    result = compute_rock_physics(well)

    expected = {"Vsh", "phi", "K0", "G0", "Ksat", "Kd_K0", "PR", "AI", "VpVs"}
    assert expected.issubset(result.columns)
    assert np.isfinite(result["Kd_K0"]).mean() > 0.99
