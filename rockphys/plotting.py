import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from .constants import TOP_HEIMDAL, OWC_DEPTH, K_QTZ, K_SH, G_SH

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
    i = 0
    while i < len(depth):
        j = i
        while j < len(depth) and fac[j] == fac[i]:
            j += 1
        ax.axhspan(depth[i], depth[min(j, len(depth)-1)],
                   color=FACIES_COLORS.get(fac[i], '#FFFFFF'), alpha=0.18, linewidth=0)
        i = j


def _add_reservoir_markers(axes, depth_col):
    """Add Top Heimdal and OWC horizontal marker lines to all axes."""
    for ax in axes:
        ax.axhline(TOP_HEIMDAL, color='#2CA02C', ls='-.', lw=0.9, alpha=0.8)
        ax.axhline(OWC_DEPTH,   color='#1F77B4', ls='-.', lw=0.9, alpha=0.8)


def fig_well_logs_zoomed(w, depth_min=2120.0, depth_max=2320.0):
    """Zoomed 8-track display over the reservoir interval."""
    fig = plt.figure(figsize=(22, 13))
    gs  = GridSpec(1, 8, figure=fig, wspace=0.06,
                   left=0.04, right=0.97, top=0.88, bottom=0.10)
    depth = w['depth']
    zm    = (depth >= depth_min) & (depth <= depth_max)
    wz    = w[zm].copy()
    dz    = depth[zm]

    def new_ax(col, sharey=None):
        return fig.add_subplot(gs[0, col], sharey=sharey)

    ax_d = new_ax(0)
    ax0  = new_ax(1, ax_d); ax1 = new_ax(2, ax_d); ax2 = new_ax(3, ax_d)
    ax3  = new_ax(4, ax_d); ax4 = new_ax(5, ax_d); ax5 = new_ax(6, ax_d); ax6 = new_ax(7, ax_d)
    axes = [ax_d, ax0, ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
        ax.invert_yaxis(); ax.set_ylim(depth_max+2, depth_min-2)
        ax.tick_params(labelsize=7); ax.grid(axis='x', color='#CCCCCC', lw=0.4)
        _shade_facies(ax, wz)
    for ax in axes[1:]:
        ax.set_yticks([])

    _add_reservoir_markers(axes, depth)
    xlim = ax_d.get_xlim()
    if depth_min <= TOP_HEIMDAL <= depth_max:
        ax_d.text(xlim[1]*0.5, TOP_HEIMDAL-1.5, 'Top Heimdal',
                  va='bottom', ha='center', fontsize=6, color='#2CA02C', style='italic')
    if depth_min <= OWC_DEPTH <= depth_max:
        ax_d.text(xlim[1]*0.5, OWC_DEPTH-1.5, 'OWC',
                  va='bottom', ha='center', fontsize=6, color='#1F77B4', style='italic')

    ax_d.set_xlim(-0.5, 0.5); ax_d.set_xticks([])
    ax_d.set_ylabel('Depth (m)', fontsize=9, fontweight='bold')
    for d in np.arange(depth_min, depth_max+10, 10):
        ax_d.text(0, d, f'{d:.0f}', ha='center', va='center', fontsize=6)

    ax0.plot(wz['GR'], dz, 'k-', lw=0.6); ax0.set_xlim(0, 150)
    ax0.set_xlabel('GR\n(GAPI)', fontsize=8)

    ax1.fill_betweenx(dz, 0, wz['Vsh'], color='sienna', alpha=0.5)
    ax1.plot(wz['Vsh'], dz, color='sienna', lw=0.6); ax1.set_xlim(0, 1)
    ax1.set_xlabel('Vsh\n(v/v)', fontsize=8)

    ax2.fill_betweenx(dz, 0, wz['phi'], color='teal', alpha=0.4)
    ax2.plot(wz['phi'], dz, color='teal', lw=0.6); ax2.set_xlim(0, 0.50)
    ax2.set_xlabel('φ_e\n(v/v)', fontsize=8)

    ax3.plot(wz['rho'], dz, color='#8B0000', lw=0.8); ax3.set_xlim(1.8, 2.8)
    ax3.set_xlabel('ρ\n(g/cc)', fontsize=8)

    for ax, col, lbl, xlim in [
        (ax4, 'Vp', 'Vp\n(km/s)', (1.2, 4.5)),
        (ax5, 'Vs', 'Vs\n(km/s)', (0.4, 2.5)),
        (ax6, 'PR', "Poisson's\nratio", (0.0, 0.50)),
    ]:
        ax.plot(wz[col],                dz, 'b-',  lw=1.0, label='Brine')
        ax.plot(wz[f'{col}_oil'],       dz, color='#2CA02C', ls='--', lw=1.0, label='Oil (Simm)', alpha=0.85)
        ax.plot(wz[f'{col}_gas'],       dz, 'r:',  lw=1.2, label='Gas (Simm)', alpha=0.85)
        ax.plot(wz[f'{col}_default_oil'], dz, color='#FF4500', ls='-.', lw=1.0, label='Oil (Default)', alpha=0.7)
        ax.plot(wz[f'{col}_default_gas'], dz, color='#8B0000', ls='-.', lw=1.2, label='Gas (Default)', alpha=0.7)
        ax.set_xlim(*xlim); ax.set_xlabel(lbl, fontsize=8)
        ax.legend(fontsize=5, loc='lower right', ncol=2)
    ax6.axvline(0.33, color='gray', ls=':', lw=0.8)

    patches = [mpatches.Patch(color=c, alpha=0.7, label=f)
               for f, c in FACIES_COLORS.items() if f != 'Background']
    patches += [mpatches.Patch(color='#2CA02C', alpha=0.8, label='Top Heimdal'),
                mpatches.Patch(color='#1F77B4', alpha=0.8, label='OWC')]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=7, title='Facies / Horizons', bbox_to_anchor=(0.52, 0.01))
    fig.suptitle(
        f'Well 2 — Glitne Field  [ZOOMED: {depth_min:.0f}–{depth_max:.0f} m]\n'
        'Brine (blue) · Oil Simm (green --) · Gas Simm (red ·) · '
        'Oil Default (orange -·) · Gas Default (brown -·)',
        fontsize=11, fontweight='bold')
    return fig


def fig_well_logs(w):
    fig = plt.figure(figsize=(22, 13))
    gs  = GridSpec(1, 8, figure=fig, wspace=0.06,
                   left=0.04, right=0.97, top=0.88, bottom=0.10)
    depth = w['depth']

    def new_ax(col, sharey=None):
        return fig.add_subplot(gs[0, col], sharey=sharey)

    ax_d = new_ax(0)
    ax0  = new_ax(1, ax_d); ax1 = new_ax(2, ax_d); ax2 = new_ax(3, ax_d)
    ax3  = new_ax(4, ax_d); ax4 = new_ax(5, ax_d); ax5 = new_ax(6, ax_d); ax6 = new_ax(7, ax_d)
    axes = [ax_d, ax0, ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
        ax.invert_yaxis(); ax.set_ylim(depth.max()+5, depth.min()-5)
        ax.tick_params(labelsize=7); ax.grid(axis='x', color='#CCCCCC', lw=0.4)
        _shade_facies(ax, w)
    for ax in axes[1:]:
        ax.set_yticks([])

    _add_reservoir_markers(axes, depth)
    xlim = ax_d.get_xlim()
    ax_d.text(xlim[1]*0.5, TOP_HEIMDAL-3, 'Top Heimdal',
              va='bottom', ha='center', fontsize=6, color='#2CA02C', style='italic')
    ax_d.text(xlim[1]*0.5, OWC_DEPTH-3,   'OWC',
              va='bottom', ha='center', fontsize=6, color='#1F77B4', style='italic')

    ax_d.set_xlim(-0.5, 0.5); ax_d.set_xticks([])
    ax_d.set_ylabel('Depth (m)', fontsize=9, fontweight='bold')
    for d in np.arange(depth.min(), depth.max()+50, 50):
        ax_d.text(0, d, f'{d:.0f}', ha='center', va='center', fontsize=6)

    ax0.plot(w['GR'],  depth, 'k-', lw=0.6); ax0.set_xlim(0, 150)
    ax0.set_xlabel('GR\n(GAPI)', fontsize=8)

    ax1.fill_betweenx(depth, 0, w['Vsh'], color='sienna', alpha=0.5)
    ax1.plot(w['Vsh'], depth, color='sienna', lw=0.6); ax1.set_xlim(0, 1)
    ax1.set_xlabel('Vsh\n(v/v)', fontsize=8)

    ax2.fill_betweenx(depth, 0, w['phi'], color='teal', alpha=0.4)
    ax2.plot(w['phi'], depth, color='teal', lw=0.6); ax2.set_xlim(0, 0.50)
    ax2.set_xlabel('φ_e\n(v/v)', fontsize=8)

    ax3.plot(w['rho'], depth, color='#8B0000', lw=0.8); ax3.set_xlim(1.8, 2.8)
    ax3.set_xlabel('ρ\n(g/cc)', fontsize=8)

    for ax, col, lbl, xlim in [
        (ax4, 'Vp', 'Vp\n(km/s)', (1.2, 4.5)),
        (ax5, 'Vs', 'Vs\n(km/s)', (0.4, 2.5)),
        (ax6, 'PR', "Poisson's\nratio", (0.0, 0.50)),
    ]:
        ax.plot(w[col],                depth, 'b-',  lw=1.0, label='Brine')
        ax.plot(w[f'{col}_oil'],       depth, color='#2CA02C', ls='--', lw=1.0, label='Oil (Simm)', alpha=0.85)
        ax.plot(w[f'{col}_gas'],       depth, 'r:',  lw=1.2, label='Gas (Simm)', alpha=0.85)
        ax.plot(w[f'{col}_default_oil'], depth, color='#FF4500', ls='-.', lw=1.0, label='Oil (Default)', alpha=0.7)
        ax.plot(w[f'{col}_default_gas'], depth, color='#8B0000', ls='-.', lw=1.2, label='Gas (Default)', alpha=0.7)
        ax.set_xlim(*xlim); ax.set_xlabel(lbl, fontsize=8)
        ax.legend(fontsize=5, loc='lower right', ncol=2)
    ax6.axvline(0.33, color='gray', ls=':', lw=0.8)

    patches = [mpatches.Patch(color=c, alpha=0.7, label=f)
               for f, c in FACIES_COLORS.items() if f != 'Background']
    patches += [mpatches.Patch(color='#2CA02C', alpha=0.8, label='Top Heimdal'),
                mpatches.Patch(color='#1F77B4', alpha=0.8, label='OWC')]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=7, title='Facies / Horizons', bbox_to_anchor=(0.52, 0.01))
    fig.suptitle(
        'Well 2 — Glitne Field, Norway  (Heimdal Formation)\n'
        'Brine (blue) · Oil Simm (green --) · Gas Simm (red ·) · '
        'Oil Default (orange -·) · Gas Default (brown -·)',
        fontsize=11, fontweight='bold')
    return fig


def fig_kd_k0(w, coeffs):
    """Kd/K0 vs porosity template — central diagnostic from Simm (2007)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    phi_r = np.linspace(0.01, 0.50, 300)

    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        kd_k0_c = c / (1.0 + c)
        ax.axhline(kd_k0_c, color='#888888', ls='--', lw=0.9, alpha=0.7)
        ax.text(0.505, kd_k0_c + 0.007, f'Kφ/K₀ = {c:.1f}',
                fontsize=8, color='#555555', va='bottom')

    order = ['Background', 'Silty Shale', 'Shale',
             'Silty Sand 2', 'Silty Sand 1', 'Cemented Sand', 'Clean Sand']
    for fac in order:
        grp   = w[w['facies'] == fac]
        valid = grp[(grp['Kd_K0'] > -0.35) & (grp['Kd_K0'] < 1.05)]
        ax.scatter(valid['phi'], valid['Kd_K0'],
                   c=FACIES_COLORS.get(fac, '#DDDDDD'),
                   s=10, alpha=0.55, edgecolors='none', label=fac, zorder=3)

    ax.plot(phi_r, np.polyval(coeffs, phi_r), 'k-', lw=2.5, zorder=6,
            label='Conditioned dry-rock trend')
    ax.scatter([0.0], [1.0], marker='*', s=200, c='black',
               zorder=7, label='Mineral point (φ=0)')
    ax.axhline(0.0, color='black', lw=0.8, ls=':')
    ax.set_xlim(-0.01, 0.52); ax.set_ylim(-0.35, 1.08)
    ax.set_xlabel('Effective Porosity  φ_e  (v/v)', fontsize=12)
    ax.set_ylabel('Normalised Dry Bulk Modulus  K_d / K₀', fontsize=12)
    ax.set_title(
        'Dry Rock Bulk Modulus Template  (Simm 2007, Figs 5 & 8)\n'
        'Dashed lines = constant Kφ/K₀ contours  |  '
        f'K_qtz={K_QTZ} GPa, K_clay={K_SH} GPa, G_clay={G_SH} GPa', fontsize=10)
    ax.legend(fontsize=8, markerscale=1.8, loc='upper right')
    ax.grid(True, alpha=0.25)
    return fig


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
        ax1.scatter(grp['phi'], grp['Vp'],     c=c, label=fac,          **kw_br)
        ax1.scatter(grp['phi'], grp['Vp_oil'], c=c, label='_nolegend_', **kw_oil)
        ax1.scatter(grp['phi'], grp['Vp_gas'], c=c, label='_nolegend_', **kw_gas)
    ax1.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax1.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax1.set_xlabel('Effective Porosity φ_e', fontsize=10)
    ax1.set_ylabel('Vp  (km/s)', fontsize=10)
    ax1.set_title('(a)  Vp vs Porosity', fontsize=11)
    ax1.legend(fontsize=6, markerscale=1.8, ncol=2); ax1.grid(True, alpha=0.25)

    # (b) Vp/Vs vs Vp
    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax2.scatter(grp['Vp'],     grp['VpVs'],                 c=c, label=fac,          **kw_br)
        ax2.scatter(grp['Vp_oil'], grp['Vp_oil']/grp['Vs_oil'], c=c, label='_nolegend_', **kw_oil)
        ax2.scatter(grp['Vp_gas'], grp['Vp_gas']/grp['Vs_gas'], c=c, label='_nolegend_', **kw_gas)
    ax2.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax2.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax2.set_xlabel('Vp  (km/s)', fontsize=10); ax2.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax2.set_title('(b)  Vp/Vs vs Vp', fontsize=11)
    ax2.axhline(2.0, color='gray', ls='--', lw=0.8, alpha=0.6, label='Vp/Vs = 2')
    ax2.legend(fontsize=6, markerscale=1.8, ncol=2); ax2.grid(True, alpha=0.25)

    # (c) AI vs Vp/Vs
    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax3.scatter(grp['AI'],     grp['VpVs'],                 c=c, label=fac,          **kw_br)
        ax3.scatter(grp['AI_oil'], grp['Vp_oil']/grp['Vs_oil'], c=c, label='_nolegend_', **kw_oil)
        ax3.scatter(grp['AI_gas'], grp['Vp_gas']/grp['Vs_gas'], c=c, label='_nolegend_', **kw_gas)
    ax3.scatter([], [], marker='^', c='k', s=12, alpha=0.6, label='Oil-sub')
    ax3.scatter([], [], marker='s', c='k', s=12, alpha=0.6, label='Gas-sub')
    ax3.set_xlabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax3.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax3.set_title('(c)  AI vs Vp/Vs  (AVO proxy)\nCircles=brine  △=oil-sub  □=gas-sub', fontsize=10)
    ax3.legend(fontsize=6, markerscale=1.8, ncol=2); ax3.grid(True, alpha=0.25)

    # (d) Kd/K0 vs φ coloured by Vsh
    valid = labelled[(labelled['Kd_K0'] > -0.35) & (labelled['Kd_K0'] < 1.05)]
    sc = ax4.scatter(valid['phi'], valid['Kd_K0'],
                     c=valid['Vsh'], cmap='RdYlGn_r', vmin=0, vmax=1,
                     s=10, alpha=0.65, edgecolors='none')
    plt.colorbar(sc, ax=ax4, label='Vsh  (v/v)', shrink=0.85)
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        ax4.axhline(c/(1+c), color='gray', ls='--', lw=0.7, alpha=0.6)
    ax4.axhline(0.0, color='black', lw=0.8, ls=':')
    ax4.set_xlim(-0.01, 0.52); ax4.set_ylim(-0.35, 1.05)
    ax4.set_xlabel('Effective Porosity φ_e', fontsize=10)
    ax4.set_ylabel('K_d / K₀', fontsize=10)
    ax4.set_title('(d)  K_d/K₀ vs Porosity  (coloured by Vsh)\n[Dry-rock — fluid independent]', fontsize=10)
    ax4.grid(True, alpha=0.25)

    fig.suptitle('Rock Physics Crossplot Suite — Well 2 (Glitne Field)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_fluid_sensitivity(w):
    """Fluid substitution vectors: brine→oil (green) and brine→gas (red)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes
    sand_facies = ['Clean Sand', 'Cemented Sand', 'Silty Sand 1', 'Silty Sand 2']
    labelled    = w[w['facies'].isin(sand_facies)]
    kw_br  = dict(s=22, alpha=0.70, edgecolors='none', marker='o')
    kw_oil = dict(s=18, alpha=0.65, edgecolors='none', marker='^')
    kw_gas = dict(s=18, alpha=0.65, edgecolors='none', marker='s')

    for fac, grp in labelled.groupby('facies'):
        c = FACIES_COLORS[fac]
        ax1.scatter(grp['AI'],     grp['VpVs'],                 c=c, **kw_br)
        ax1.scatter(grp['AI_oil'], grp['Vp_oil']/grp['Vs_oil'], c=c, **kw_oil)
        ax1.scatter(grp['AI_gas'], grp['Vp_gas']/grp['Vs_gas'], c=c, **kw_gas)
        ax2.scatter(grp['PR'],     grp['AI'],     c=c, **kw_br)
        ax2.scatter(grp['PR_oil'], grp['AI_oil'], c=c, **kw_oil)
        ax2.scatter(grp['PR_gas'], grp['AI_gas'], c=c, **kw_gas)

    subset = labelled.sample(min(100, len(labelled)), random_state=42)
    for _, r in subset.iterrows():
        ax1.annotate('', xy=(r['AI_oil'], r['Vp_oil']/r['Vs_oil']),
                     xytext=(r['AI'], r['VpVs']),
                     arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=0.7, alpha=0.45))
        ax2.annotate('', xy=(r['PR_oil'], r['AI_oil']), xytext=(r['PR'], r['AI']),
                     arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=0.7, alpha=0.45))
        ax1.annotate('', xy=(r['AI_gas'], r['Vp_gas']/r['Vs_gas']),
                     xytext=(r['AI'], r['VpVs']),
                     arrowprops=dict(arrowstyle='->', color='#D62728', lw=0.7, alpha=0.45))
        ax2.annotate('', xy=(r['PR_gas'], r['AI_gas']), xytext=(r['PR'], r['AI']),
                     arrowprops=dict(arrowstyle='->', color='#D62728', lw=0.7, alpha=0.45))

    brine_pt = mpatches.Patch(color='gray',    label='Brine (circle)')
    oil_pt   = mpatches.Patch(color='#2CA02C', label='Oil-sub (triangle, green arrow)')
    gas_pt   = mpatches.Patch(color='#D62728', label='Gas-sub (square, red arrow)')
    ax1.set_xlabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax1.set_ylabel('Vp/Vs  ratio', fontsize=10)
    ax1.set_title('(a)  AI vs Vp/Vs\nBrine → Oil (green)  Brine → Gas (red)', fontsize=11)
    ax1.legend(handles=[brine_pt, oil_pt, gas_pt], fontsize=7); ax1.grid(True, alpha=0.25)
    ax2.set_xlabel("Poisson's ratio  σ", fontsize=10)
    ax2.set_ylabel('Acoustic Impedance  (g/cc · km/s)', fontsize=10)
    ax2.set_title("(b)  AI vs Poisson's ratio\nBrine → Oil (green)  Brine → Gas (red)", fontsize=11)
    ax2.legend(handles=[brine_pt, oil_pt, gas_pt], fontsize=7); ax2.grid(True, alpha=0.25)
    fig.suptitle('Fluid Substitution Effect — Sand Facies, Glitne Field WELL-2',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_facies_comparison(w):
    """Per-facies Vp depth profiles: default vs Simm conditioning."""
    facies_list = ['Clean Sand','Cemented Sand','Silty Sand 1','Silty Sand 2','Silty Shale']
    fig, axes   = plt.subplots(len(facies_list), 1, figsize=(12, 15), sharex=True)
    depth = w['depth']
    for i, fac in enumerate(facies_list):
        ax   = axes[i]
        mask = w['facies'] == fac
        if not mask.any():
            ax.text(0.5, 0.5, f'No data for {fac}', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        _shade_facies(ax, w[mask])
        ax.plot(w.loc[mask,'Vp'],             depth[mask], 'b-',  lw=1.5, label='Brine')
        ax.plot(w.loc[mask,'Vp_default_oil'], depth[mask], 'r--', lw=1.2, label='Default Oil')
        ax.plot(w.loc[mask,'Vp_oil'],         depth[mask], 'g-',  lw=1.2, label='Simm Oil')
        ax.plot(w.loc[mask,'Vp_default_gas'], depth[mask], 'm:',  lw=1.2, label='Default Gas')
        ax.plot(w.loc[mask,'Vp_gas'],         depth[mask], 'c-',  lw=1.2, label='Simm Gas')
        ax.set_ylim(depth[mask].max()+2, depth[mask].min()-2)
        ax.set_ylabel('Depth (m)', fontsize=9)
        ax.set_title(f'{fac}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    axes[-1].set_xlabel('Vp (km/s)', fontsize=10)
    fig.suptitle('Facies-Specific Fluid Substitution: Default (Raw Kd) vs Simm (Conditioned)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_default_vs_simm_differences(w):
    """Difference crossplots: (Default − Simm) for oil and gas substitutions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    labelled = w[w['facies'] != 'Background']

    for ax, xcol, ycol, title in [
        (ax1, 'phi', 'Vp_default_oil - Vp_oil',  '(a)  Oil ΔVp vs Porosity'),
        (ax2, 'phi', 'Vp_default_gas - Vp_gas',  '(b)  Gas ΔVp vs Porosity'),
        (ax3, 'Vsh', 'Vp_default_oil - Vp_oil',  '(c)  Oil ΔVp vs Shale Volume'),
        (ax4, 'phi', 'PR_default_oil - PR_oil',   '(d)  Oil ΔPR vs Porosity'),
    ]:
        for fac, grp in labelled.groupby('facies'):
            yvals = grp.eval(ycol)
            ax.scatter(grp[xcol], yvals, c=FACIES_COLORS[fac],
                       s=15, alpha=0.7, edgecolors='none', label=fac)
        ax.axhline(0, color='black', ls='--', alpha=0.5)
        xlabel = 'Effective Porosity φ_e' if xcol == 'phi' else 'Vsh  (v/v)'
        ylabel = ycol.replace(' - ', ' − ').replace('_', ' ')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(f'Δ  ({ylabel.split("−")[0].strip()})  (Default − Simm)', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, markerscale=1.5, ncol=2); ax.grid(True, alpha=0.25)

    fig.suptitle(
        'Impact of Dry Rock Conditioning: Default vs Simm (Positive Δ = Default higher)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig
