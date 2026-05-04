"""Read-only baseline checks for the rock-physics Gassmann demo repo."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_DATA_FILES = (
    "README.md",
    "gassmann_demo.ipynb",
    "gassmann_demo.py",
    "data/info.txt",
    "data/info_params.txt",
    "data/info_cdp.txt",
    "data/well_2.txt",
    "data/well_2.las",
    "rockphys/config.py",
    "rockphys/pipeline.py",
)

FACIES_FILES = (
    "data/well2_cemSand.txt",
    "data/well2_clnSand.txt",
    "data/well2_sltSand1.txt",
    "data/well2_sltSand2.txt",
    "data/well2_sltShale.txt",
)

EXPECTED_FIGURES = (
    "fig1_well_logs.png",
    "fig1b_well_logs_zoomed.png",
    "fig2_kd_k0_template.png",
    "fig3_crossplots.png",
    "fig4_fluid_sensitivity.png",
    "fig5_facies_comparison.png",
    "fig6_differences.png",
)


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _format_result(result: CheckResult) -> str:
    return f"[{result.status}] {result.name:<28} {result.detail}"


def _check_file(path: str | Path) -> CheckResult:
    file_path = _repo_path(path)
    if not file_path.exists():
        return CheckResult("FAIL", _display(file_path), "missing")
    if file_path.stat().st_size == 0:
        return CheckResult("FAIL", _display(file_path), "empty file")
    return CheckResult("OK", _display(file_path), f"{file_path.stat().st_size:,} bytes")


def _read_numeric_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(_repo_path(path), comment="%", header=None, sep=r"\s+")
    numeric = table.apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(how="any")


def _check_depth_table(
    path: str | Path,
    min_rows: int,
    name: str,
    *,
    require_monotonic: bool = True,
) -> CheckResult:
    table_path = _repo_path(path)
    if not table_path.exists():
        return CheckResult("FAIL", name, f"missing: {_display(table_path)}")

    try:
        table = _read_numeric_table(table_path)
    except Exception as exc:
        return CheckResult("FAIL", name, f"read error: {type(exc).__name__}: {exc}")

    problems: list[str] = []
    if len(table) < min_rows:
        problems.append(f"rows={len(table)}, expected >= {min_rows}")
    if table.shape[1] < 6:
        problems.append(f"columns={table.shape[1]}, expected >= 6")
    if require_monotonic and table.shape[1] and not table.iloc[:, 0].is_monotonic_increasing:
        problems.append("depth column is not monotonic increasing")

    if problems:
        return CheckResult("FAIL", name, "; ".join(problems))

    depth = table.iloc[:, 0]
    return CheckResult(
        "OK",
        name,
        f"rows={len(table)}, columns={table.shape[1]}, depth={depth.min():.2f}-{depth.max():.2f} m",
    )


def _check_rock_physics_smoke() -> CheckResult:
    try:
        from rockphys.pipeline import SimmWorkflowConfig, run_simm_workflow

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = run_simm_workflow(SimmWorkflowConfig(data_dir=ROOT / "data"))
    except Exception as exc:
        return CheckResult("FAIL", "Rock physics smoke", f"{type(exc).__name__}: {exc}")

    required_columns = (
        "Kd_K0",
        "Kphi_K0",
        "Vp_gas",
        "AI_gas",
        "Vp_default_gas",
        "AI_default_gas",
    )
    merged = result.well
    missing = [column for column in required_columns if column not in merged.columns]
    if missing:
        return CheckResult("FAIL", "Rock physics smoke", "missing columns: " + ", ".join(missing))

    labelled = merged["facies"] != "Background"
    finite = (
        np.isfinite(merged[["Kd_K0"]]).all().all()
        and np.isfinite(merged.loc[labelled, ["Vp_gas", "AI_gas"]]).all().all()
    )
    if not finite:
        return CheckResult("FAIL", "Rock physics smoke", "non-finite values in labelled facies outputs")

    facies_count = int(labelled.sum())
    return CheckResult(
        "OK",
        "Rock physics smoke",
        f"rows={len(merged)}, facies_samples={facies_count}, coeffs={result.coeffs[0]:.3f},{result.coeffs[1]:.3f},{result.coeffs[2]:.1f}",
    )


def run_checks() -> list[CheckResult]:
    results = [_check_file(path) for path in REQUIRED_DATA_FILES]
    results.extend(_check_file(path) for path in FACIES_FILES)
    results.extend(_check_file(path) for path in EXPECTED_FIGURES)
    results.append(_check_depth_table("data/well_2.txt", 4000, "Well 2 table"))
    for path in FACIES_FILES:
        results.append(
            _check_depth_table(
                path,
                50,
                f"Facies table {Path(path).name}",
                require_monotonic=False,
            )
        )
    results.append(_check_rock_physics_smoke())
    return results


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    _parse_args()
    results = run_checks()

    for result in results:
        print(_format_result(result))

    failures = [result for result in results if result.status == "FAIL"]
    warnings = [result for result in results if result.status == "WARN"]
    print()
    print(f"{len(results)} checks: {len(failures)} FAIL, {len(warnings)} WARN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
