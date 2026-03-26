import numpy as np
from .constants import RHO_SH, RHO_QTZ, RHO_BR, K_SH, K_QTZ, G_SH, G_QTZ, K_BR


def batzle_wang_oil(T_c, P_mpa, API, GOR_sm3):
    """
    Batzle & Wang (1992) live-oil bulk modulus at reservoir conditions.
    Returns (K_oil GPa, rho_live g/cc, Vp_live m/s).
    """
    G_g = 0.6  # dissolved-gas gravity (relative to air)
    rho_0 = 141.5 / (131.5 + API)               # surface dead-oil density
    R_s   = GOR_sm3 * 5.615                      # GOR: sm³/sm³ → scf/stb
    T_f   = T_c * 9.0 / 5.0 + 32.0             # °C → °F
    B_0   = 0.972 + 0.00038 * (2.4 * R_s * np.sqrt(G_g / rho_0) + T_f + 17.8)**1.175
    rho_live = (rho_0 + 0.001224 * G_g * R_s) / B_0

    def _dead_vp(rho):                           # B-W Eq. 20 [m/s]
        return (2096.0 * np.sqrt(rho / (2.6 - rho))
                - 3.7 * T_c + 4.64 * P_mpa
                + 0.0115 * (4.12 * np.sqrt(1.08 / rho - 1.0) - 1.0) * T_c * P_mpa)

    Vp_live = _dead_vp(rho_live)                 # B-W Eq. 22: use live density
    K_oil   = rho_live * (Vp_live / 1000.0)**2  # [GPa]
    return K_oil, rho_live, Vp_live


def vrh(f, M1, M2):
    """Voigt-Reuss-Hill average. f = volume fraction of M1."""
    Mv = f * M1 + (1 - f) * M2
    Mr = 1.0 / (f / M1 + (1 - f) / M2)
    return 0.5 * (Mv + Mr)


def gassmann_inv(Ksat, K0, Kfl, phi):
    """Invert Gassmann for normalised dry bulk modulus Kd/K0 (Simm 2007 Eq. 4)."""
    beta = phi * K0 / Kfl + 1.0 - phi
    y    = Ksat / K0
    return (y * beta - 1.0) / (y + beta - 2.0)


def gassmann_fwd(Kd_K0, K0, Kfl_new, phi):
    """Forward Gassmann: new saturated bulk modulus (GPa)."""
    x    = Kd_K0
    beta = phi * K0 / Kfl_new + 1.0 - phi
    return K0 * (x + (1.0 - x)**2 / (beta - x))


def poisson(Vp, Vs):
    """Poisson's ratio from Vp and Vs."""
    return (Vp**2 - 2*Vs**2) / (2*(Vp**2 - Vs**2))


def compute_rock_physics(well):
    """Steps 2-4: Vsh, porosity, VRH moduli, Gassmann inversion, derived quantities."""
    w = well.copy()

    # ── Vsh from GR (data-driven end-points) ─────────────────────────────────
    gr_clean = np.percentile(w['GR'], 5)
    gr_shale = np.percentile(w['GR'], 95)
    w['Vsh'] = ((w['GR'] - gr_clean) / (gr_shale - gr_clean)).clip(0.0, 1.0)

    # ── Effective porosity from density ──────────────────────────────────────
    rho_min  = w['Vsh'] * RHO_SH + (1.0 - w['Vsh']) * RHO_QTZ
    w['phi'] = ((rho_min - w['rho']) / (rho_min - RHO_BR)).clip(0.01, 0.55)

    # ── Mineral moduli via VRH ────────────────────────────────────────────────
    w['K0'] = vrh(w['Vsh'], K_SH, K_QTZ)
    w['G0'] = vrh(w['Vsh'], G_SH, G_QTZ)

    # ── Saturated moduli from logs (ρ [g/cc] × V² [km/s]² = GPa) ─────────────
    w['mu']   = w['rho'] * w['Vs']**2
    w['Ksat'] = w['rho'] * w['Vp']**2 - (4.0/3.0) * w['mu']

    # ── Gassmann inversion (assumes brine saturation) ─────────────────────────
    w['Kd_K0'] = gassmann_inv(w['Ksat'], w['K0'], K_BR, w['phi'])
    w['Kd']    = w['Kd_K0'] * w['K0']

    # Normalised pore stiffness  Kφ/K0 = (Kd/K0) / (1 − Kd/K0)
    denom = (1.0 - w['Kd_K0']).where(lambda x: x.abs() > 1e-4, other=np.nan)
    w['Kphi_K0'] = w['Kd_K0'] / denom

    # ── Derived seismic quantities ────────────────────────────────────────────
    w['PR']   = poisson(w['Vp'], w['Vs'])
    w['AI']   = w['rho'] * w['Vp']
    w['VpVs'] = w['Vp'] / w['Vs']

    return w
