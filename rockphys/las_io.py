"""LAS loading adapter for real-data style inputs."""

from __future__ import annotations

from pathlib import Path

import lasio


CURVE_ALIASES = {
    "depth": ("DEPT", "DEPTH"),
    "Vp": ("VP", "Vp"),
    "Vs": ("VS", "Vs"),
    "rho": ("RHOB", "RHO", "DEN", "DENS"),
    "GR": ("GR", "GAMMA"),
    "nphi": ("NPHI", "PHIN", "NPOR"),
}


def _find_curve(columns, aliases):
    normalized = {column.upper(): column for column in columns}
    for alias in aliases:
        column = normalized.get(alias.upper())
        if column is not None:
            return column
    return None


def load_las_well(las_path):
    """Load a LAS well and normalize curve names for the rockphys workflow.

    The current rock-physics workflow expects velocities in km/s and density in
    g/cc. LAS files that already use those common units are passed through.
    """
    las = lasio.read(Path(las_path))
    well = las.df().reset_index()

    rename = {}
    for target, aliases in CURVE_ALIASES.items():
        source = _find_curve(well.columns, aliases)
        if source is not None:
            rename[source] = target

    well = well.rename(columns=rename)
    required = ("depth", "Vp", "Vs", "rho", "GR")
    missing = [column for column in required if column not in well.columns]
    if missing:
        raise ValueError(f"Missing required LAS curves: {', '.join(missing)}")

    optional = ["nphi"] if "nphi" in well.columns else []
    columns = [*required, *optional]
    return well.loc[:, columns].dropna(subset=required).reset_index(drop=True)
