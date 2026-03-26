import os
import glob
import numpy as np
import pandas as pd


def _parse_txt(filepath):
    """Parse a well-data text file (handles both header styles in the dataset)."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if (not s or s.startswith('%')
                    or s.lower().startswith('depth') or s.startswith('vti')):
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
    """Load all per-facies text files; return dict: facies name → set of depths."""
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
