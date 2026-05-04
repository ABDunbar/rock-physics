"""Configuration objects for the Simm rock-physics workflow."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    K_BR,
    K_GAS,
    OIL_API,
    OIL_GOR,
    RESERVOIR_P,
    RESERVOIR_TEMP,
    RHO_BR,
    RHO_GAS,
    RHO_OIL,
)


@dataclass(frozen=True)
class FluidProperties:
    bulk_modulus_gpa: float
    density_g_cm3: float


@dataclass(frozen=True)
class ReservoirProperties:
    temperature_c: float = RESERVOIR_TEMP
    pressure_mpa: float = RESERVOIR_P
    oil_api: float = OIL_API
    oil_gor_sm3: float = OIL_GOR


@dataclass(frozen=True)
class DryRockTrendConfig:
    fit_facies: tuple[str, ...] = (
        "Clean Sand",
        "Cemented Sand",
        "Silty Sand 1",
        "Silty Sand 2",
    )
    conditioned_facies: tuple[str, ...] = ("Silty Sand 1", "Silty Sand 2")
    min_kd_k0: float = 0.05
    max_kd_k0: float = 1.00
    min_porosity: float = 0.05
    conditioned_min_kd_k0: float = 0.05
    conditioned_max_kd_k0: float = 0.99


BRINE_FLUID = FluidProperties(K_BR, RHO_BR)
GAS_FLUID = FluidProperties(K_GAS, RHO_GAS)
OIL_FLUID = FluidProperties(bulk_modulus_gpa=0.0, density_g_cm3=RHO_OIL)
DEFAULT_DRY_ROCK_TREND = DryRockTrendConfig()
DEFAULT_RESERVOIR = ReservoirProperties()
