# shaley-gassmann Inventory

`rock-physics` is the main Simm (2007) project. `shaley-gassmann` is now best
treated as a compact prototype and reference implementation for a LAS-focused
shaley-sand fluid substitution workflow.

## Purpose

`shaley-gassmann` implements the Rob Simm shaley-sand fluid substitution idea
around one narrow problem: standard Gassmann/Avseth substitution can expose
nonphysical inverted dry bulk modulus values, especially in shaley sands, so the
workflow fits a dry-rock trend and conditions questionable samples before
resaturation.

`rock-physics` covers the same scientific area, but with the fuller notebook
context: facies files, Batzle-Wang oil properties, oil and gas substitutions,
default-vs-Simm comparisons, and the seven figure diagnostic suite.

## Structure Compared

`shaley-gassmann`:

- `fluid_sub/logs.py`: LAS loading, unit conversion, VSH, and effective porosity.
- `fluid_sub/standard.py`: Bruges Avseth substitution plus dry bulk modulus QC.
- `fluid_sub/adaptive.py`: Simm-style trend fitting, conditioning, and
  resaturation.
- `fluid_sub/simm.py`: small formula functions for dry bulk inversion,
  conditioning, density substitution, and velocity calculation.
- `fluid_sub/plots.py`: single four-panel QC plot.
- `fluid_sub/pipeline.py`: importable pipeline.
- `fluid_sub_workflow.py`: thin CLI wrapper.
- `tests/`: useful unit and workflow tests around LAS loading, negative dry bulk
  visibility, finite substituted outputs, config overrides, and plot writing.

`rock-physics`:

- `rockphys/io.py`: text-table well and facies loading for the Simm demo data.
- `rockphys/physics.py`: Batzle-Wang oil, VRH, Gassmann inverse/forward, and
  derived rock physics attributes.
- `rockphys/substitution.py`: default and Simm adaptive substitutions.
- `rockphys/plotting.py`: seven diagnostic figure builders.
- `rockphys/pipeline.py`: canonical importable Simm workflow.
- `scripts/run_simm_workflow.py`: supported CLI entry point.
- `notebooks/`: compact API notebook plus the original narrative notebook.
- `tests/`: pipeline, CLI, notebook contract, checker, and supported entrypoint
  tests.

## Unique Useful Pieces In shaley-gassmann

- LAS-first loading path using `lasio`.
- Bruges `avseth_fluidsub` baseline comparison.
- Explicit kg/m3 internal-unit prototype.
- Small formula-level tests proving negative dry bulk modulus is exposed for QC.
- Single compact QC plot, useful as a quick-look real-data view.
- Real-data style CLI accepting a direct LAS path and output PNG path.

## Already Covered By rock-physics

- Main Simm paper narrative and reference data.
- Dry-rock trend fitting and adaptive conditioning.
- Default-vs-Simm substitution comparison.
- Oil and gas substitution scenarios.
- Batzle-Wang oil calculation.
- Facies-aware data model.
- Seven richer diagnostic figures.
- Importable API, CLI, notebook contract, and baseline checks.

## Keep / Migrate / Archive

Keep in `rock-physics`:

- Simm paper notebook and main scientific workflow.
- Package modules under `rockphys/`.
- Seven diagnostic figures and pipeline notebook.
- Regression tests that pin scientific output and formula behavior.

Migrate from `shaley-gassmann` only when needed:

- LAS ingestion as an optional loader, if the next real-data step needs LAS
  input rather than the Simm text-table data.
- Bruges/Avseth comparison as an optional baseline, if you want to compare the
  current in-house default Gassmann implementation with a known library.
- The compact four-panel QC plot, if a quick-look real-data figure is useful.

Archive or leave as reference:

- `fluid_sub_workflow.py`, once `rock-physics` has any needed LAS entrypoint.
- Duplicate data/reference files.
- Duplicate formula modules once matching regression coverage exists in
  `rock-physics`.

## Recommendation

Do not merge formulas wholesale. The repos currently make different scientific
choices:

- `rock-physics` conditions selected silty facies from the labelled facies data.
- `shaley-gassmann` conditions by thresholds on VSH and dry bulk modulus.

Treat any change from facies-based conditioning to threshold-based conditioning
as a scientific decision. Before changing that behavior, pin current numeric
results and figure generation with regression tests, then compare the output
curves and diagnostic plots side by side.
