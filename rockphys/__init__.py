from .constants import *
from .io import load_well, load_facies, assign_facies
from .physics import batzle_wang_oil, vrh, gassmann_inv, gassmann_fwd, poisson, compute_rock_physics
from .substitution import fit_dry_rock_trend, apply_fluid_substitution, apply_default_fluid_substitution
from .plotting import (FACIES_COLORS, fig_well_logs, fig_well_logs_zoomed, fig_kd_k0,
                       fig_crossplots, fig_fluid_sensitivity, fig_facies_comparison,
                       fig_default_vs_simm_differences)
