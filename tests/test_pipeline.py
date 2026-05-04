from pathlib import Path

import numpy as np

from rockphys.config import DryRockTrendConfig, FluidProperties, ReservoirProperties
from rockphys.pipeline import (
    DEFAULT_CONFIG,
    FIGURE_BUILDERS,
    SimmWorkflowConfig,
    run_simm_workflow,
)


ROOT = Path(__file__).resolve().parents[1]


def test_run_simm_workflow_returns_oil_gas_and_default_outputs():
    result = run_simm_workflow(DEFAULT_CONFIG)

    expected = {
        "Kd_K0",
        "Kphi_K0",
        "Vp_oil",
        "Vp_gas",
        "Vp_default_oil",
        "Vp_default_gas",
        "AI_oil",
        "AI_gas",
    }

    assert expected.issubset(result.well.columns)
    assert len(result.well) == 4117
    assert result.coeffs.shape == (3,)
    assert np.isfinite(result.coeffs).all()


def test_run_simm_workflow_can_write_figures_to_configured_output_dir(tmp_path):
    config = SimmWorkflowConfig(
        data_dir=ROOT / "data",
        output_dir=tmp_path,
        write_figures=True,
    )

    result = run_simm_workflow(config)

    assert set(result.figure_paths) == set(FIGURE_BUILDERS)
    for figure_name, figure_path in result.figure_paths.items():
        assert figure_path == tmp_path / figure_name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0


def test_default_config_matches_explicit_default_config():
    implicit = run_simm_workflow()
    explicit = run_simm_workflow(DEFAULT_CONFIG)

    assert np.allclose(implicit.coeffs, explicit.coeffs)
    assert np.allclose(implicit.well["Vp_gas"], explicit.well["Vp_gas"], equal_nan=True)
    assert np.allclose(implicit.well["Vp_oil"], explicit.well["Vp_oil"], equal_nan=True)


def test_custom_gas_fluid_changes_gas_substitution():
    default = run_simm_workflow()
    custom = run_simm_workflow(
        SimmWorkflowConfig(
            data_dir=ROOT / "data",
            gas_fluid=FluidProperties(bulk_modulus_gpa=0.10, density_g_cm3=0.30),
        )
    )

    assert custom.well["rho_gas"].mean() > default.well["rho_gas"].mean()
    assert not np.allclose(custom.well["Vp_gas"], default.well["Vp_gas"], equal_nan=True)


def test_custom_reservoir_properties_change_oil_bulk_modulus():
    default = run_simm_workflow()
    custom = run_simm_workflow(
        SimmWorkflowConfig(
            data_dir=ROOT / "data",
            reservoir=ReservoirProperties(
                temperature_c=90.0,
                pressure_mpa=25.0,
                oil_api=35.0,
                oil_gor_sm3=80.0,
            ),
        )
    )

    assert custom.oil_bulk_modulus_gpa != default.oil_bulk_modulus_gpa


def test_custom_conditioned_facies_changes_adaptive_oil_substitution():
    default = run_simm_workflow()
    custom = run_simm_workflow(
        SimmWorkflowConfig(
            data_dir=ROOT / "data",
            dry_rock_trend=DryRockTrendConfig(
                conditioned_facies=("Clean Sand", "Silty Sand 1", "Silty Sand 2"),
            ),
        )
    )

    clean_sand = default.well["facies"] == "Clean Sand"
    assert not np.allclose(
        custom.well.loc[clean_sand, "Vp_oil"],
        default.well.loc[clean_sand, "Vp_oil"],
    )
