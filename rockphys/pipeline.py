"""Importable Simm 2007 rock-physics workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

from .config import (
    DEFAULT_DRY_ROCK_TREND,
    DEFAULT_RESERVOIR,
    GAS_FLUID,
    OIL_FLUID,
    DryRockTrendConfig,
    FluidProperties,
    ReservoirProperties,
)
from .io import assign_facies, load_facies, load_well
from .physics import batzle_wang_oil, compute_rock_physics
from .plotting import (
    fig_crossplots,
    fig_default_vs_simm_differences,
    fig_facies_comparison,
    fig_fluid_sensitivity,
    fig_kd_k0,
    fig_well_logs,
    fig_well_logs_zoomed,
)
from .substitution import (
    apply_default_fluid_substitution,
    apply_fluid_substitution,
    fit_dry_rock_trend,
)


FIGURE_BUILDERS = {
    "fig1_well_logs.png": lambda well, coeffs: fig_well_logs(well),
    "fig1b_well_logs_zoomed.png": lambda well, coeffs: fig_well_logs_zoomed(well),
    "fig2_kd_k0_template.png": lambda well, coeffs: fig_kd_k0(well, coeffs),
    "fig3_crossplots.png": lambda well, coeffs: fig_crossplots(well),
    "fig4_fluid_sensitivity.png": lambda well, coeffs: fig_fluid_sensitivity(well),
    "fig5_facies_comparison.png": lambda well, coeffs: fig_facies_comparison(well),
    "fig6_differences.png": lambda well, coeffs: fig_default_vs_simm_differences(well),
}


@dataclass(frozen=True)
class SimmWorkflowConfig:
    data_dir: Path = Path("data")
    output_dir: Path | None = None
    write_figures: bool = False
    reservoir: ReservoirProperties = DEFAULT_RESERVOIR
    oil_fluid: FluidProperties = OIL_FLUID
    gas_fluid: FluidProperties = GAS_FLUID
    dry_rock_trend: DryRockTrendConfig = DEFAULT_DRY_ROCK_TREND


@dataclass(frozen=True)
class SimmWorkflowResult:
    well: object
    coeffs: object
    oil_bulk_modulus_gpa: float
    oil_density_g_cm3: float
    figure_paths: dict[str, Path] = field(default_factory=dict)


DEFAULT_CONFIG = SimmWorkflowConfig()


def _compute_oil_bulk_modulus() -> float:
    return _compute_oil_bulk_modulus_for(DEFAULT_CONFIG)


def _compute_oil_bulk_modulus_for(config: SimmWorkflowConfig) -> float:
    _, _, vp_oil_mps = batzle_wang_oil(
        config.reservoir.temperature_c,
        config.reservoir.pressure_mpa,
        config.reservoir.oil_api,
        config.reservoir.oil_gor_sm3,
    )
    return config.oil_fluid.density_g_cm3 * (vp_oil_mps / 1000.0) ** 2


def _write_figures(well, coeffs, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {}
    for figure_name, builder in FIGURE_BUILDERS.items():
        fig = builder(well, coeffs)
        figure_path = output_dir / figure_name
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths[figure_name] = figure_path
    return figure_paths


def run_simm_workflow(config: SimmWorkflowConfig = DEFAULT_CONFIG) -> SimmWorkflowResult:
    """Run the full Simm notebook workflow without requiring notebook execution."""
    data_dir = Path(config.data_dir)
    well = load_well(data_dir)
    well = assign_facies(well, load_facies(data_dir))
    well = compute_rock_physics(well)

    coeffs = fit_dry_rock_trend(well, config.dry_rock_trend)
    k_oil = _compute_oil_bulk_modulus_for(config)

    well = apply_fluid_substitution(
        well,
        coeffs,
        k_oil,
        config.oil_fluid.density_g_cm3,
        suffix="oil",
        config=config.dry_rock_trend,
    )
    well = apply_fluid_substitution(
        well,
        coeffs,
        config.gas_fluid.bulk_modulus_gpa,
        config.gas_fluid.density_g_cm3,
        suffix="gas",
        config=config.dry_rock_trend,
    )
    well = apply_default_fluid_substitution(
        well,
        k_oil,
        config.oil_fluid.density_g_cm3,
        suffix="oil",
    )
    well = apply_default_fluid_substitution(
        well,
        config.gas_fluid.bulk_modulus_gpa,
        config.gas_fluid.density_g_cm3,
        suffix="gas",
    )

    figure_paths = {}
    if config.write_figures:
        output_dir = Path(config.output_dir) if config.output_dir is not None else data_dir.parent
        figure_paths = _write_figures(well, coeffs, output_dir)

    return SimmWorkflowResult(
        well=well,
        coeffs=coeffs,
        oil_bulk_modulus_gpa=k_oil,
        oil_density_g_cm3=config.oil_fluid.density_g_cm3,
        figure_paths=figure_paths,
    )
