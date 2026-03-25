#!/usr/bin/env python3
"""
Practical Gassmann Fluid Substitution in Sand/Shale Sequences
==============================================================
Demonstration following Simm (2007), First Break, Vol 25, December 2007.

Workflow (Simm's 6-step adaptive approach):
  1. Load well logs (Vp, Vs, density, GR) and assign facies labels
  2. Compute Vsh from GR; effective porosity from density log
  3. Mineral modulus K0 via Voigt-Reuss-Hill mixing (quartz + shale)
  4. Gassmann inversion → dry rock parameters (Kd, Kφ)
  5. QC on Kd/K0 vs φ crossplot; fit adaptive polynomial dry-rock trend
  6. Forward Gassmann substitution: brine → oil  AND  brine → gas

Well: WELL-2, Glitne Field, Norway (Heimdal Formation, oil field)
Data: Vp (km/s), Vs (km/s), RHOB (g/cc), GR (GAPI), NPHI
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (saves without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# MINERAL AND FLUID PROPERTIES  (Simm 2007 + info_params.txt updates)
# ─────────────────────────────────────────────────────────────────────────────
# Quartz (info_params.txt)
K_QTZ   = 36.8;  G_QTZ   = 44.0;  RHO_QTZ  = 2.65   # GPa, g/cc

# Clay / Shale (info_params.txt)
K_SH    = 15.0;  G_SH    =  5.0;  RHO_SH   = 2.72

# Brine (info_params.txt)
K_BR    = 2.80;  RHO_BR  = 1.09   # GPa, g/cc

# Dry gas at reservoir conditions
K_GAS   = 0.02;  RHO_GAS = 0.12   # GPa, g/cc

# Oil – density given directly; K_OIL computed via Batzle-Wang in main()
RHO_OIL = 0.78                     # g/cc  (info_params.txt)

# ── Reservoir conditions ──────────────────────────────────────────────────────
RESERVOIR_TEMP   = 77.2    # °C
RESERVOIR_P      = 20.0    # MPa effective pressure
OIL_API          = 32
OIL_GOR          = 64      # sm³/sm³  (metric solution GOR)
OWC_DEPTH        = 2183    # m  (oil-water contact)
TOP_HEIMDAL      = 2153    # m  (top of Heimdal reservoir)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_txt(filepath):
    """Parse a well-data text file (handles both header styles in the dataset)."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if (not s
                    or s.startswith('%')
                    or s.lower().startswith('depth')
                    or s.startswith('vti')):
                continue
            try:
                vals = [float(x) for x in s.split()]
                if len(vals) == 6:
                    rows.append(vals)
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=['depth', 'Vp', 'Vs', 'rho', 'GR', 'nphi'])


def load_well(data_dir):
    """Load main well log."""
    return _parse_txt(os.path.join(data_dir, 'well_2.txt'))


def load_facies(data_dir):
    """
    Load all per-facies text files and return a dict mapping
    facies name → set of rounded depths (used for labelling well samples).
    """
    patterns = {
        'Clean Sand'   : 'well2_clnSand*.txt',
        'Cemented Sand': 'well2_cemSand*.txt',
        'Silty Sand 1' : 'well2_sltSand1*.txt',
        'Silty Sand 2' : 'well2_sltSand2*.txt',
        'Silty Shale'  : 'well2_sltShale*.txt',
        'Shale'        : 'well2_Shale*.txt',
    }
    result = {}
    for name, pat in patterns.items():
        files = glob.glob(os.path.join(data_dir, pat))
        if not files:
            print(f"  WARNING: no files matched {pat}")
            continue
        frames = [_parse_txt(f) for f in files]
        combined = pd.concat(frames, ignore_index=True)
        result[name] = set(np.round(combined['depth'].values, 1))
    return result


def assign_facies(well, facies_depths):
    """Label each depth sample by its facies (depth-matched)."""
    w = well.copy()
    w['facies'] = 'Background'
    d_r = w['depth'].round(1)
    for name, depths in facies_depths.items():
        w.loc[d_r.isin(depths), 'facies'] = name
    return w


# ─────────────────────────────────────────────────────────────────────────────
# BATZLE-WANG OIL MODULUS
# ─────────────────────────────────────────────────────────────────────────────

def batzle_wang_oil(T_c, P_mpa, API, GOR_sm3):
    """
    Batzle & Wang (1992) live-oil bulk modulus at reservoir conditions.

    Steps follow Batzle & Wang (1992) Geophysics 57(11):
      1. Surface dead-oil density from API gravity
      2. Formation volume factor B_0  (Eq. 18)
      3. Live oil density at reservoir  (Eq. 19)
      4. Dead-oil P-wave velocity at T, P  (Eq. 20)
      5. Live-oil velocity via dissolved-gas correction (Eq. 22):
         substitute live density into dead-oil velocity formula
      6. K_oil = rho_live * (Vp_live / 1000)^2  [GPa]

    Returns (K_oil GPa, rho_live_bw g/cc).
    """
    G_g = 0.6  # dissolved-gas gravity (relative to air)

    # 1. Surface dead-oil density
    rho_0 = 141.5 / (131.5 + API)

    # 2. Convert GOR: sm³/sm³ → scf/stb (B-W equations use imperial)
    R_s = GOR_sm3 * 5.615

    # 3. Formation volume factor (B-W Eq. 18); temperature in °F
    T_f = T_c * 9.0 / 5.0 + 32.0
    B_0 = 0.972 + 0.00038 * (2.4 * R_s * np.sqrt(G_g / rho_0) + T_f + 17.8)**1.175

    # 4. Live oil density at reservoir (B-W Eq. 19)
    rho_live = (rho_0 + 0.001224 * G_g * R_s) / B_0

    # 5a. Dead-oil P-wave velocity at T (°C), P (MPa)  [m/s]  (B-W Eq. 20)
    def _dead_vp(rho):
        return (2096.0 * np.sqrt(rho / (2.6 - rho))
                - 3.7 * T_c
                + 4.64 * P_mpa
                + 0.0115 * (4.12 * np.sqrt(1.08 / rho - 1.0) - 1.0) * T_c * P_mpa)

    # 5b. Live-oil velocity: apply dead-oil formula with live density (B-W Eq. 22)
    Vp_live = _dead_vp(rho_live)

    # 6. Oil bulk modulus  [GPa]
    #    Per plan step 7: use given density if B-W live density is unreliable.
    #    Return both K_oil (from B-W ρ) and Vp_live so caller can override.
    K_oil = rho_live * (Vp_live / 1000.0)**2

    return K_oil, rho_live, Vp_live


# ─────────────────────────────────────────────────────────────────────────────
# ROCK PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def vrh(f, M1, M2):
    """Voigt-Reuss-Hill average. f = volume fraction of M1."""
    Mv = f * M1 + (1 - f) * M2
    Mr = 1.0 / (f / M1 + (1 - f) / M2)
    return 0.5 * (Mv + Mr)


def gassmann_inv(Ksat, K0, Kfl, phi):
    """
    Invert Gassmann equation for normalised dry bulk modulus Kd/K0.

    Derivation of closed-form inverse (Simm 2007, Eq. 4):
        y = x + (1-x)^2 / (β - x)   where y=Ksat/K0, x=Kd/K0, β=φ·K0/Kfl+1-φ
        → x = (y·β − 1) / (y + β − 2)
    """
    beta = phi * K0 / Kfl + 1.0 - phi
    y    = Ksat / K0
    return (y * beta - 1.0) / (y + beta - 2.0)


def gassmann_fwd(Kd_K0, K0, Kfl_new, phi):
    """Forward Gassmann: return new saturated bulk modulus (GPa)."""
    x    = Kd_K0
    beta = phi * K0 / Kfl_new + 1.0 - phi
    return K0 * (x + (1.0 - x)**2 / (beta - x))


def poisson(Vp, Vs):
    return (Vp**2 - 2*Vs**2) / (2*(Vp**2 - Vs**2))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROCK PHYSICS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_rock_physics(well):
    """
    Steps 2-4 of Simm's workflow:
      - Vsh from GR (linear normalisation, data-driven end-points)
      - Effective porosity from density log
      - Mineral moduli K0, G0 via VRH
      - Saturated moduli from logs (V in km/s, ρ in g/cc → moduli in GPa)
      - Gassmann inversion for Kd, Kφ
      - Poisson's ratio and acoustic impedance
    """
    w = well.copy()

    # ── Vsh from GR ───────────────────────────────────────────────────────────
    gr_clean = np.percentile(w['GR'], 5)
    gr_shale = np.percentile(w['GR'], 95)
    w['Vsh'] = ((w['GR'] - gr_clean) / (gr_shale - gr_clean)).clip(0.0, 1.0)

    # ── Effective porosity from density ──────────────────────────────────────
    rho_min  = w['Vsh'] * RHO_SH + (1.0 - w['Vsh']) * RHO_QTZ
    w['phi'] = ((rho_min - w['rho']) / (rho_min - RHO_BR)).clip(0.01, 0.55)

    # ── Mineral moduli (VRH) ─────────────────────────────────────────────────
    w['K0'] = vrh(w['Vsh'], K_SH, K_QTZ)
    w['G0'] = vrh(w['Vsh'], G_SH, G_QTZ)

    # ── Saturated moduli from logs ────────────────────────────────────────────
    # Units: ρ [g/cc] × V² [km²/s²] = GPa  ✓
    w['mu']   = w['rho'] * w['Vs']**2
    w['Ksat'] = w['rho'] * w['Vp']**2 - (4.0/3.0) * w['mu']

    # ── Gassmann inversion (assumes brine saturation) ────────────────────────
    w['Kd_K0'] = gassmann_inv(w['Ksat'], w['K0'], K_BR, w['phi'])
    w['Kd']    = w['Kd_K0'] * w['K0']

    # Normalised pore stiffness  Kφ/K0 = (Kd/K0) / (1 − Kd/K0)
    denom = (1.0 - w['Kd_K0']).where(lambda x: x.abs() > 1e-4, other=np.nan)
    w['Kphi_K0'] = w['Kd_K0'] / denom

    # ── Derived quantities ────────────────────────────────────────────────────
    w['PR']   = poisson(w['Vp'], w['Vs'])
    w['AI']   = w['rho'] * w['Vp']
    w['VpVs'] = w['Vp'] / w['Vs']

    return w


def fit_dry_rock_trend(well):
    """
    Fit a 2nd-order polynomial Kd/K0 = aφ² + bφ + 1 to sand data,
    anchored at the mineral point (φ=0, Kd/K0=1) as in Simm 2007 Fig. 8.
    Returns array [a, b, 1.0].
    """
    sand_facies = ['Clean Sand', 'Cemented Sand', 'Silty Sand 1', 'Silty Sand 2']
    mask = (
        well['facies'].isin(sand_facies)
        & (well['Kd_K0'] > 0.05)
        & (well['Kd_K0'] < 1.00)
        & (well['phi']   > 0.05)
    )
    sand = well[mask]
    if len(sand) < 5:
        print("  WARNING: very few valid sand points for trend fitting.")
    phi = sand['phi'].values
    y   = sand['Kd_K0'].values - 1.0        # shift so anchor = 0
    # Fit y - 1 = a*phi^2 + b*phi  (no constant → enforces anchor at phi=0)
    X, _, _, _ = np.linalg.lstsq(
        np.column_stack([phi**2, phi]), y, rcond=None)
    a, b = X
    return np.array([a, b, 1.0])            # coefficients for np.polyval


def apply_fluid_substitution(well, coeffs, K_fl_new, RHO_fl_new, suffix):
    """
    Generic Gassmann forward substitution: brine → any target fluid.

    Conditions dry rock on the fitted polynomial trend, then performs
    forward Gassmann substitution.  Adds columns:
      Ksat_{suffix}, rho_{suffix}, Vp_{suffix}, Vs_{suffix},
      PR_{suffix}, AI_{suffix}
    """
    w = well.copy()

    # Conditioned dry bulk modulus (trend value, clipped to physical range)
    w['Kd_K0_c'] = np.polyval(coeffs, w['phi']).clip(0.05, 0.99)

    # Forward substitution: brine → new fluid
    w[f'Ksat_{suffix}'] = gassmann_fwd(w['Kd_K0_c'], w['K0'], K_fl_new, w['phi'])

    # Updated density (replace brine with new fluid in pore space)
    w[f'rho_{suffix}']  = w['rho'] - w['phi'] * RHO_BR + w['phi'] * RHO_fl_new

    # New velocities (shear modulus μ is fluid-independent)
    w[f'Vp_{suffix}']   = np.sqrt(
        (w[f'Ksat_{suffix}'] + (4.0/3.0)*w['mu']) / w[f'rho_{suffix}'])
    w[f'Vs_{suffix}']   = np.sqrt(w['mu'] / w[f'rho_{suffix}'])

    # Post-substitution Poisson's ratio and AI
    w[f'PR_{suffix}']   = poisson(w[f'Vp_{suffix}'], w[f'Vs_{suffix}'])
    w[f'AI_{suffix}']   = w[f'rho_{suffix}'] * w[f'Vp_{suffix}']

    return w


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

FACIES_COLORS = {
    'Clean Sand'   : '#FFD700',
    'Cemented Sand': '#FF8C00',
    'Silty Sand 1' : '#6DBF67',
    'Silty Sand 2' : '#2E8B57',
    'Silty Shale'  : '#AAAAAA',
    'Shale'        : '#606060',
    'Background'   : '#E0E0E0',
}


def _shade_facies(ax, w):
    """Add subtle facies colour bands behind a depth track."""
    depth = w['depth'].values
    fac   = w['facies'].values
    i     = 0
    while i < len(depth):
        j = i
        while j < len(depth) and fac[j] == fac[i]:
            j += 1
        color = FACIES_COLORS.get(fac[i], '#FFFFFF')
        ax.axhspan(depth[i], depth[min(j, len(depth)-1)],
                   color=color, alpha=0.18, linewidth=0)
        i = j


def _add_reservoir_markers(axes, depth_col):
    """Add Top Heimdal and OWC horizontal marker lines to all axes."""
    for ax in axes:
        ax.axhline(TOP_HEIMDAL, color='#2CA02C', ls='-.',  lw=0.9, alpha=0.8)
        ax.axhline(OWC_DEPTH,   color='#1F77B4', ls='-.',  lw=0.9, alpha=0.8)


# ── Figure 1: Multi-track log display ─────────────────────────────────────────

def fig_well_logs(w):
    fig = plt.figure(figsize=(20, 13))
    gs  = GridSpec(1, 7, figure=fig, wspace=0.06,
                   left=0.06, right=0.97, top=0.88, bottom=0.10)

    depth = w['depth']

    def new_ax(col, sharey=None):
        return fig.add_subplot(gs[0, col], sharey=sharey)

    ax0 = new_ax(0)
    ax1 = new_ax(1, ax0)
    ax2 = new_ax(2, ax0)
    ax3 = new_ax(3, ax0)
    ax4 = new_ax(4, ax0)
    ax5 = new_ax(5, ax0)
    ax6 = new_ax(6, ax0)
    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
        ax.invert_yaxis()
        ax.set_ylim(depth.max() + 5, depth.min() - 5)
        ax.tick_params(labelsize=7)
        ax.grid(axis='x', color='#CCCCCC', lw=0.4)
        _shade_facies(ax, w)

    for ax in axes[1:]:
        ax.set_yticks([])

    # ── Reservoir horizon markers ─────────────────────────────────────────────
    _add_reservoir_markers(axes, depth)
    # Annotate on the leftmost track (ax0)
    xlim = ax0.get_xlim()
    ax0.text(xlim[1] * 0.98, TOP_HEIMDAL - 3, 'Top Heimdal',
             va='bottom', ha='right', fontsize=6, color='#2CA02C', style='italic')
    ax0.text(xlim[1] * 0.98, OWC_DEPTH - 3, 'OWC',
             va='bottom', ha='right', fontsize=6, color='#1F77B4', style='italic')

    # Track 0: GR
    ax0.plot(w['GR'], depth, 'k-', lw=0.6)
    ax0.set_xlim(0, 150)
    ax0.set_xlabel('GR\n(GAPI)', fontsize=8)
    ax0.set_ylabel('Depth (m)', fontsize=9)

    # Track 1: Vsh
    ax1.fill_betweenx(depth, 0, w['Vsh'], color='sienna', alpha=0.5)
    ax1.plot(w['Vsh'], depth, color='sienna', lw=0.6)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Vsh\n(v/v)', fontsize=8)

    # Track 2: Effective porosity
    ax2.fill_betweenx(depth, 0, w['phi'], color='teal', alpha=0.4)
    ax2.plot(w['phi'], depth, color='teal', lw=0.6)
    ax2.set_xlim(0, 0.50)
    ax2.set_xlabel('φ_e\n(v/v)', fontsize=8)

    # Track 3: Density
    ax3.plot(w['rho'], depth, color='#8B0000', lw=0.8)
    ax3.set_xlim(1.8, 2.8)
    ax3.set_xlabel('ρ\n(g/cc)', fontsize=8)

    # Track 4: Vp  (brine=blue solid, oil=green dashed, gas=red dotted)
    ax4.plot(w['Vp'],     depth, 'b-',  lw=1.0,  label='Brine')
    ax4.plot(w['Vp_oil'], depth, color='#2CA02C', ls='--', lw=1.0, label='Oil',  alpha=0.85)
    ax4.plot(w['Vp_gas'], depth, 'r:',  lw=1.2,  label='Gas',  alpha=0.85)
    ax4.set_xlim(1.2, 4.5)
    ax4.set_xlabel('Vp\n(km/s)', fontsize=8)
    ax4.legend(fontsize=6, loc='lower right')

    # Track 5: Vs  (brine=blue solid, oil=green dashed, gas=red dotted)
    ax5.plot(w['Vs'],     depth, 'b-',  lw=1.0,  label='Brine')
    ax5.plot(w['Vs_oil'], depth, color='#2CA02C', ls='--', lw=1.0, label='Oil',  alpha=0.85)
    ax5.plot(w['Vs_gas'], depth, 'r:',  lw=1.2,  label='Gas',  alpha=0.85)
    ax5.set_xlim(0.4, 2.5)
    ax5.set_xlabel('Vs\n(km/s)', fontsize=8)
    ax5.legend(fontsize=6, loc='lower right')

    # Track 6: Poisson's ratio  (brine=blue solid, oil=green dashed, gas=red dotted)
    ax6.plot(w['PR'],     depth, 'b-',  lw=1.0,  label='Brine')
    ax6.plot(w['PR_oil'], depth, color='#2CA02C', ls='--', lw=1.0, label='Oil',  alpha=0.85)
    ax6.plot(w['PR_gas'], depth, 'r:',  lw=1.2,  label='Gas',  alpha=0.85)
    ax6.axvline(0.33, color='gray', ls=':', lw=0.8)
    ax6.set_xlim(0.0, 0.50)
    ax6.set_xlabel("Poisson's\nratio", fontsize=8)
    ax6.legend(fontsize=6, loc='lower right')

    # Facies legend
    patches = [mpatches.Patch(color=c, alpha=0.7, label=f)
               for f, c in FACIES_COLORS.items() if f != 'Background']
    # Reservoir horizon legend entries
    patches += [
        mpatches.Patch(color='#2CA02C', alpha=0.8, label='Top Heimdal'),
        mpatches.Patch(color='#1F77B4', alpha=0.8, label='OWC'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=7, title='Facies / Horizons', bbox_to_anchor=(0.52, 0.01))

    fig.suptitle(
        'Well 2 — Glitne Field, Norway  (Heimdal Formation)\n'
        'Multi-Track Log Display: Brine (blue solid)  ·  Oil (green dashed)'
        '  ·  Gas-Substituted (red dotted)',
        fontsize=11, fontweight='bold')
    return fig


# ── Figure 2: Kd/K0 vs porosity crossplot ─────────────────────────────────────

def fig_kd_k0(w, coeffs):
    """
    Central diagnostic template from Simm (2007).
    Horizontal dashed lines = contours of constant normalised pore stiffness
    Kφ/K0 (horizontal because Kφ/K0 = (Kd/K0) / (1−Kd/K0) depends only on Kd/K0).
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    phi_r = np.linspace(0.01, 0.50, 300)

    # ── Kφ/K0 contours ────────────────────────────────────────────────────────
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        kd_k0_c = c / (1.0 + c)               # Kd/K0 for this Kφ/K0
        ax.axhline(kd_k0_c, color='#888888', ls='--', lw=0.9, alpha=0.7)
        ax.text(0.505, kd_k0_c + 0.007,
                f'Kφ/K₀ = {c:.1f}', fontsize=8, color='#555555', va='bottom')

    # ── Scatter (facies-coloured) ─────────────────────────────────────────────
    order = ['Background', 'Silty Shale', 'Shale',
             'Silty Sand 2', 'Silty Sand 1', 'Cemented Sand', 'Clean Sand']
    for fac in order:
        grp = w[w['facies'] == fac]
        valid = grp[(grp['Kd_K0'] > -0.35) & (grp['Kd_K0'] < 1.05)]
        ax.scatter(valid['phi'], valid['Kd_K0'],
                   c=FACIES_COLORS.get(fac, '#DDDDDD'),
                   s=10, alpha=0.55, edgecolors='none',
                   label=fac, zorder=3)

    # ── Dry rock trend ────────────────────────────────────────────────────────
    kd_trend = np.polyval(coeffs, phi_r)
    ax.plot(phi_r, kd_trend, 'k-', lw=2.5, zorder=6,
            label='Conditioned dry-rock trend')

    # ── Mineral point anchor ──────────────────────────────────────────────────
    ax.scatter([0.0], [1.0], marker='*', s=200, c='black',
               zorder=7, label='Mineral point (φ=0)')

    ax.axhline(0.0, color='black', lw=0.8, ls=':')  # zero reference

    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(-0.35, 1.08)
    ax.set_xlabel('Effective Porosity  φ_e  (v/v)', fontsize=12)
    ax.set_ylabel('Normalised Dry Bulk Modulus  K_d / K₀', fontsize=12)
    ax.set_title(
        'Dry Rock Bulk Modulus Template  (Simm 2007, Figs 5 & 8)\n'
        'Horizontal dashed lines = constant K_φ/K₀ contours\n'
        f'Updated mineral moduli: K_qtz={K_QTZ} GPa, K_clay={K_SH} GPa, '
        f'G_clay={G_SH} GPa', fontsize=10)
    ax.legend(fontsize=8, markerscale=1.8, loc='upper right')
    ax.grid(True, alpha=0.25)
    return fig


# ── Figure 3: Crossplot suite ──────────────────────────────────────────────────

def fig_crossplots(w):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    labelled = w[w['facies'] != 'Background']

    kw_br  = dict(s=10, alpha=0.65, edgecolors='none', marker='o')
    kw_oil = dict(s=8,  alpha=0.45, edgecolors='none', marker='^')
    kw_gas = dict(s=8,  alpha=0.45, edgecolors='none', marker='s')

    # (a) Vp vs φ_e
    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax1.scatter(grp['phi'], grp['Vp'],     c=c, label=f'{fac}',      **kw_br)
        ax1.scatter(grp['phi'], grp['Vp_oil'], c=c, label='_nolegend_',   **kw_oil)
        ax1.scatter(grp['phi'], grp['Vp_gas'], c=c, label='_nolegend_',   **kw_gas)
    # Proxy legend entries for substituted points
    ax1.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax1.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax1.set_xlabel('Effective Porosity φ_e', fontsize=10)
    ax1.set_ylabel('Vp  (km/s)', fontsize=10)
    ax1.set_title('(a)  Vp vs Porosity', fontsize=11)
    ax1.legend(fontsize=6, markerscale=1.8, ncol=2)
    ax1.grid(True, alpha=0.25)

    # (b) Vp/Vs vs Vp  — shows lithology + fluid sensitivity
    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax2.scatter(grp['Vp'],     grp['VpVs'],                         c=c, label=fac, **kw_br)
        ax2.scatter(grp['Vp_oil'], grp['Vp_oil']/grp['Vs_oil'],         c=c, label='_nolegend_', **kw_oil)
        ax2.scatter(grp['Vp_gas'], grp['Vp_gas']/grp['Vs_gas'],         c=c, label='_nolegend_', **kw_gas)
    ax2.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax2.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax2.set_xlabel('Vp  (km/s)', fontsize=10)
    ax2.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax2.set_title('(b)  Vp/Vs vs Vp', fontsize=11)
    ax2.axhline(2.0, color='gray', ls='--', lw=0.8, alpha=0.6,
                label='Vp/Vs = 2 (σ ≈ 0.33)')
    ax2.legend(fontsize=6, markerscale=1.8, ncol=2)
    ax2.grid(True, alpha=0.25)

    # (c) AI vs Vp/Vs  — AVO proxy (class II/III gas sands appear bottom-left)
    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax3.scatter(grp['AI'],     grp['VpVs'],                         c=c, label=fac, **kw_br)
        ax3.scatter(grp['AI_oil'], grp['Vp_oil']/grp['Vs_oil'],         c=c, label='_nolegend_', **kw_oil)
        ax3.scatter(grp['AI_gas'], grp['Vp_gas']/grp['Vs_gas'],         c=c, label='_nolegend_', **kw_gas)
    ax3.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax3.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax3.set_xlabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax3.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax3.set_title('(c)  AI vs Vp/Vs  (AVO proxy)\nCircles = brine, △ = oil-sub, □ = gas-sub',
                  fontsize=10)
    ax3.legend(fontsize=6, markerscale=1.8, ncol=2)
    ax3.grid(True, alpha=0.25)

    # (d) Kd/K0 vs φ_e, coloured continuously by Vsh  (dry rock — fluid-independent)
    valid = labelled[(labelled['Kd_K0'] > -0.35) & (labelled['Kd_K0'] < 1.05)]
    sc = ax4.scatter(valid['phi'], valid['Kd_K0'],
                     c=valid['Vsh'], cmap='RdYlGn_r',
                     vmin=0, vmax=1, s=10, alpha=0.65, edgecolors='none')
    plt.colorbar(sc, ax=ax4, label='Vsh  (v/v)', shrink=0.85)
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        ax4.axhline(c/(1+c), color='gray', ls='--', lw=0.7, alpha=0.6)
    ax4.axhline(0.0, color='black', lw=0.8, ls=':')
    ax4.set_xlim(-0.01, 0.52)
    ax4.set_ylim(-0.35, 1.05)
    ax4.set_xlabel('Effective Porosity φ_e', fontsize=10)
    ax4.set_ylabel('K_d / K₀', fontsize=10)
    ax4.set_title('(d)  K_d/K₀ vs Porosity  (coloured by Vsh)\n'
                  '[Dry-rock property — fluid independent]', fontsize=10)
    ax4.grid(True, alpha=0.25)

    fig.suptitle('Rock Physics Crossplot Suite — Well 2 (Glitne Field)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Figure 4: Fluid substitution sensitivity ──────────────────────────────────

def fig_fluid_sensitivity(w):
    """
    Illustrate Gassmann fluid substitution effect on elastic properties.
    Shows brine→oil (green) and brine→gas (red) substitution vectors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    sand_facies = ['Clean Sand', 'Cemented Sand', 'Silty Sand 1', 'Silty Sand 2']
    labelled = w[w['facies'].isin(sand_facies)]

    kw_br  = dict(s=22, alpha=0.70, edgecolors='none', marker='o')
    kw_oil = dict(s=18, alpha=0.65, edgecolors='none', marker='^')
    kw_gas = dict(s=18, alpha=0.65, edgecolors='none', marker='s')

    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        # ── Panel (a): AI vs Vp/Vs ──
        ax1.scatter(grp['AI'],     grp['VpVs'],                c=c, **kw_br)
        ax1.scatter(grp['AI_oil'], grp['Vp_oil']/grp['Vs_oil'], c=c, **kw_oil)
        ax1.scatter(grp['AI_gas'], grp['Vp_gas']/grp['Vs_gas'], c=c, **kw_gas)
        # ── Panel (b): AI vs PR ──
        ax2.scatter(grp['PR'],     grp['AI'],     c=c, **kw_br)
        ax2.scatter(grp['PR_oil'], grp['AI_oil'], c=c, **kw_oil)
        ax2.scatter(grp['PR_gas'], grp['AI_gas'], c=c, **kw_gas)

    # Draw arrows: brine→oil (green) and brine→gas (red)
    subset = labelled.sample(min(100, len(labelled)), random_state=42)
    for _, r in subset.iterrows():
        # Oil arrows (green)
        ax1.annotate('', xy=(r['AI_oil'], r['Vp_oil']/r['Vs_oil']),
                     xytext=(r['AI'], r['VpVs']),
                     arrowprops=dict(arrowstyle='->', color='#2CA02C',
                                     lw=0.7, alpha=0.45))
        ax2.annotate('', xy=(r['PR_oil'], r['AI_oil']),
                     xytext=(r['PR'], r['AI']),
                     arrowprops=dict(arrowstyle='->', color='#2CA02C',
                                     lw=0.7, alpha=0.45))
        # Gas arrows (red)
        ax1.annotate('', xy=(r['AI_gas'], r['Vp_gas']/r['Vs_gas']),
                     xytext=(r['AI'], r['VpVs']),
                     arrowprops=dict(arrowstyle='->', color='#D62728',
                                     lw=0.7, alpha=0.45))
        ax2.annotate('', xy=(r['PR_gas'], r['AI_gas']),
                     xytext=(r['PR'], r['AI']),
                     arrowprops=dict(arrowstyle='->', color='#D62728',
                                     lw=0.7, alpha=0.45))

    # ── Proxy legend ─────────────────────────────────────────────────────────
    brine_pt = mpatches.Patch(color='gray', label='Brine (circle)')
    oil_pt   = mpatches.Patch(color='#2CA02C', label='Oil-sub (triangle, green arrow)')
    gas_pt   = mpatches.Patch(color='#D62728', label='Gas-sub (square, red arrow)')

    ax1.set_xlabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax1.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax1.set_title('(a)  AI vs Vp/Vs\nBrine → Oil (green) and Brine → Gas (red)', fontsize=11)
    ax1.legend(handles=[brine_pt, oil_pt, gas_pt], fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Poisson's ratio  σ", fontsize=10)
    ax2.set_ylabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax2.set_title("(b)  AI vs Poisson's ratio\nBrine → Oil (green) and Brine → Gas (red)",
                  fontsize=11)
    ax2.legend(handles=[brine_pt, oil_pt, gas_pt], fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        'Fluid Substitution Effect on Elastic Properties\n'
        'Brine → Oil  and  Brine → Gas  (Glitne Field, WELL-2)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Gassmann Fluid Substitution Demo  (Simm 2007)")
    print("  Glitne Field, WELL-2 — Brine / Oil / Gas scenarios")
    print("=" * 60)

    # ── Batzle-Wang oil modulus ────────────────────────────────────────────────
    print("\n[0] Computing oil properties (Batzle-Wang 1992) …")
    K_OIL_bw, rho_oil_bw, Vp_oil_mps = batzle_wang_oil(
        RESERVOIR_TEMP, RESERVOIR_P, OIL_API, OIL_GOR)
    # Use the given density (0.78 g/cc) for K_oil because B-W live density
    # differs significantly from the measured reservoir value (high-GOR case).
    K_OIL = RHO_OIL * (Vp_oil_mps / 1000.0)**2
    print(f"    B-W live density:     {rho_oil_bw:.3f} g/cc  "
          f"(given: {RHO_OIL:.2f} g/cc)")
    print(f"    B-W live velocity:    {Vp_oil_mps:.0f} m/s")
    print(f"    K_oil (Batzle-Wang):  {K_OIL:.3f} GPa  "
          f"[= {RHO_OIL:.2f} × ({Vp_oil_mps/1000:.3f})²]")

    # ── Load ───────────────────────────────────────────────────────────────────
    print("\n[1] Loading well data …")
    well = load_well(DATA_DIR)
    print(f"    {len(well)} samples, depth {well['depth'].min():.0f}–"
          f"{well['depth'].max():.0f} m")

    print("[2] Assigning facies …")
    facies_depths = load_facies(DATA_DIR)
    well = assign_facies(well, facies_depths)
    print(well['facies'].value_counts().to_string(header=False))

    # ── Rock physics ───────────────────────────────────────────────────────────
    print("\n[3] Computing rock physics …")
    well = compute_rock_physics(well)

    sand = well[well['facies'].isin(['Clean Sand', 'Cemented Sand',
                                      'Silty Sand 1', 'Silty Sand 2'])]
    print(f"    Porosity range (all):  {well['phi'].min():.3f} – "
          f"{well['phi'].max():.3f}")
    print(f"    Kd/K0  P5–P95:  "
          f"{well['Kd_K0'].quantile(0.05):.3f} – "
          f"{well['Kd_K0'].quantile(0.95):.3f}  "
          f"(negative values expected in shaly/low-φ zones)")

    # ── Dry rock trend ─────────────────────────────────────────────────────────
    print("\n[4] Fitting conditioned dry-rock trend …")
    coeffs = fit_dry_rock_trend(well)
    print(f"    Kd/K0 = {coeffs[0]:.3f}·φ² + {coeffs[1]:.3f}·φ + {coeffs[2]:.3f}")

    # ── Fluid substitutions ────────────────────────────────────────────────────
    print("[5] Applying fluid substitutions …")
    well = apply_fluid_substitution(well, coeffs, K_OIL,  RHO_OIL,  suffix='oil')
    well = apply_fluid_substitution(well, coeffs, K_GAS,  RHO_GAS,  suffix='gas')

    # Refresh sand mask after new columns are added
    sand = well[well['facies'].isin(['Clean Sand', 'Cemented Sand',
                                      'Silty Sand 1', 'Silty Sand 2'])]
    mean_dvp_oil = (sand['Vp_oil'] - sand['Vp']).mean()
    mean_dvp_gas = (sand['Vp_gas'] - sand['Vp']).mean()
    mean_dpr_oil = (sand['PR_oil'] - sand['PR']).mean()
    mean_dpr_gas = (sand['PR_gas'] - sand['PR']).mean()

    print(f"    K_oil (Batzle-Wang):  {K_OIL:.3f} GPa")
    print(f"    ρ_oil (given):        {RHO_OIL:.2f} g/cc")
    print(f"    Mean ΔVp oil (sand):  {mean_dvp_oil:.3f} km/s")
    print(f"    Mean ΔVp gas (sand):  {mean_dvp_gas:.3f} km/s")
    print(f"    Mean ΔPR  oil (sand): {mean_dpr_oil:.3f}")
    print(f"    Mean ΔPR  gas (sand): {mean_dpr_gas:.3f}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n[6] Generating figures …")
    figs = [
        (fig_well_logs(well),         'fig1_well_logs.png'),
        (fig_kd_k0(well, coeffs),     'fig2_kd_k0_template.png'),
        (fig_crossplots(well),        'fig3_crossplots.png'),
        (fig_fluid_sensitivity(well), 'fig4_fluid_sensitivity.png'),
    ]
    for fig, fname in figs:
        path = os.path.join(DATA_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved → {path}")

    print("\nDone.  Open the PNG files in the rock-physics directory.")


if __name__ == '__main__':
    main()
