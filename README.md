# rock-physics

Canonical Simm (2007) rock-physics working repo for practical Gassmann fluid
substitution in sand/shale sequences.

This repo contains the fuller notebook and plotting workflow. The smaller
`shaley-gassmann` repo is best treated as a clean engineering prototype for
API/config/CLI patterns, not as a replacement for this scientific notebook.

## Contents

- `gassmann_demo.ipynb`: narrative notebook following the Simm paper.
- `notebooks/01_simm_workflow_pipeline.ipynb`: compact notebook using the
  importable pipeline API.
- `rockphys/`: reusable helpers for loading data, computing rock physics,
  applying default/Simm substitutions, plotting, and running the workflow.
- `data/`: well 2 logs and facies interval files.
- `reference/`: Simm paper and reference notes.
- `docs/shaley_gassmann_inventory.md`: inventory of the smaller prototype repo
  and its relationship to this main project.
- `fig*.png`: generated diagnostic figures from the demo workflow.
- `scripts/check_inputs.py`: read-only baseline checks for data, figures, and
  importable workflow behavior.
- `scripts/run_simm_workflow.py`: command-line entry point for the package
  pipeline.

## Quick Start

```bash
uv sync
uv run pytest -q
uv run python scripts/check_inputs.py
```

To run the importable workflow from the terminal:

```bash
uv run python scripts/run_simm_workflow.py --data-dir data
```

To regenerate the standard figure set into a separate directory:

```bash
uv run python scripts/run_simm_workflow.py \
  --data-dir data \
  --output-dir outputs/simm_demo \
  --write-figures
```

## Notebook/API Usage

Use the importable pipeline when you want the Simm workflow results without
executing the notebook:

```python
from rockphys import run_simm_workflow

result = run_simm_workflow()
well = result.well
well[["depth", "facies", "Kd_K0", "Vp_oil", "Vp_gas", "Vp_default_gas"]].head()
```

To write the standard figure set to another directory:

```python
from pathlib import Path
from rockphys import SimmWorkflowConfig, run_simm_workflow

config = SimmWorkflowConfig(
    data_dir=Path("data"),
    output_dir=Path("outputs/simm_demo"),
    write_figures=True,
)

result = run_simm_workflow(config)
result.figure_paths
```

You can also override the scientific configuration without editing module
constants:

```python
from rockphys import DryRockTrendConfig, FluidProperties, SimmWorkflowConfig, run_simm_workflow

config = SimmWorkflowConfig(
    gas_fluid=FluidProperties(bulk_modulus_gpa=0.10, density_g_cm3=0.30),
    dry_rock_trend=DryRockTrendConfig(
        conditioned_facies=("Silty Sand 1", "Silty Sand 2"),
    ),
)

result = run_simm_workflow(config)
```

The defaults preserve the current notebook behavior.

## Relationship To shaley-gassmann

`shaley-gassmann` overlaps with this repo around dry bulk modulus conditioning
for shaley sands. Its useful contribution is mostly engineering shape:

- smaller modules
- dataclass configuration
- importable pipeline
- CLI/readme/test conventions

This repo remains the better scientific home for the Simm paper work because it
has the notebook narrative, Batzle-Wang oil properties, oil and gas substitution,
default-vs-Simm comparisons, and richer diagnostic plots.

Do not merge formulas from `shaley-gassmann` into this repo blindly. The two
repos currently condition dry rock differently:

- `rock-physics`: conditions selected silty facies, following the notebook
  interpretation of Simm's adaptive workflow.
- `shaley-gassmann`: uses threshold-style VSH/dry-bulk conditioning from the
  smaller LAS-focused prototype.

Treat formula changes as scientific decisions and pin them with regression
tests before changing outputs.
