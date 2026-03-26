import numpy as np
from .physics import gassmann_fwd, poisson
from .constants import K_BR, RHO_BR


def fit_dry_rock_trend(well):
    """
    Fit Kd/K0 = aφ² + bφ + 1 to sand data, anchored at mineral point (φ=0).
    Returns coefficients [a, b, 1.0] for np.polyval.
    """
    sand_facies = ['Clean Sand', 'Cemented Sand', 'Silty Sand 1', 'Silty Sand 2']
    mask = (
        well['facies'].isin(sand_facies)
        & (well['Kd_K0'] > 0.05) & (well['Kd_K0'] < 1.00)
        & (well['phi']   > 0.05)
    )
    sand = well[mask]
    if len(sand) < 5:
        print("  WARNING: very few valid sand points for trend fitting.")
    phi = sand['phi'].values
    y   = sand['Kd_K0'].values - 1.0   # shift so anchor = 0
    X, _, _, _ = np.linalg.lstsq(np.column_stack([phi**2, phi]), y, rcond=None)
    a, b = X
    return np.array([a, b, 1.0])


def apply_fluid_substitution(well, coeffs, K_fl_new, RHO_fl_new, suffix):
    """Simm adaptive Gassmann: applies dry-rock trend selectively to silty sands."""
    w = well.copy()
    conditioned = ['Silty Sand 1', 'Silty Sand 2']
    w['Kd_K0_c'] = w['Kd_K0'].where(
        ~w['facies'].isin(conditioned),
        np.polyval(coeffs, w['phi']).clip(0.05, 0.99))

    w[f'Ksat_{suffix}'] = gassmann_fwd(w['Kd_K0_c'], w['K0'], K_fl_new, w['phi'])
    w[f'rho_{suffix}']  = w['rho'] - w['phi'] * RHO_BR + w['phi'] * RHO_fl_new
    w[f'Vp_{suffix}']   = np.sqrt(
        (w[f'Ksat_{suffix}'] + (4.0/3.0)*w['mu']) / w[f'rho_{suffix}'])
    w[f'Vs_{suffix}']   = np.sqrt(w['mu'] / w[f'rho_{suffix}'])
    w[f'PR_{suffix}']   = poisson(w[f'Vp_{suffix}'], w[f'Vs_{suffix}'])
    w[f'AI_{suffix}']   = w[f'rho_{suffix}'] * w[f'Vp_{suffix}']
    return w


def apply_default_fluid_substitution(well, K_fl_new, RHO_fl_new, suffix):
    """Baseline Gassmann: raw Kd/K0 (no conditioning) for all facies."""
    w = well.copy()
    w['Kd_K0_raw'] = w['Kd_K0'].clip(0.05, 0.99)

    w[f'Ksat_default_{suffix}'] = gassmann_fwd(w['Kd_K0_raw'], w['K0'], K_fl_new, w['phi'])
    w[f'rho_default_{suffix}']  = w['rho'] - w['phi'] * RHO_BR + w['phi'] * RHO_fl_new
    w[f'Vp_default_{suffix}']   = np.sqrt(
        (w[f'Ksat_default_{suffix}'] + (4.0/3.0)*w['mu']) / w[f'rho_default_{suffix}'])
    w[f'Vs_default_{suffix}']   = np.sqrt(w['mu'] / w[f'rho_default_{suffix}'])
    w[f'PR_default_{suffix}']   = poisson(w[f'Vp_default_{suffix}'], w[f'Vs_default_{suffix}'])
    w[f'AI_default_{suffix}']   = w[f'rho_default_{suffix}'] * w[f'Vp_default_{suffix}']
    return w
