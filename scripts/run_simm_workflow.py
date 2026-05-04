"""Command-line wrapper for the Simm rock-physics workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rockphys.pipeline import SimmWorkflowConfig, run_simm_workflow  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing well_2.txt and facies interval files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to the repo root when --write-figures is used.",
    )
    parser.add_argument(
        "--write-figures",
        action="store_true",
        help="Write the standard Simm diagnostic figure set.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = SimmWorkflowConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        write_figures=args.write_figures,
    )
    result = run_simm_workflow(config)
    facies_samples = int((result.well["facies"] != "Background").sum())

    print(
        "Simm workflow complete: "
        f"rows={len(result.well)}, "
        f"facies_samples={facies_samples}, "
        f"coeffs={result.coeffs[0]:.3f},{result.coeffs[1]:.3f},{result.coeffs[2]:.1f}, "
        f"k_oil={result.oil_bulk_modulus_gpa:.3f} GPa, "
        f"figures={len(result.figure_paths)}"
    )
    for figure_path in result.figure_paths.values():
        print(f"Saved {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
