from pathlib import Path

import numpy as np

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
