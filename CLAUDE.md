# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Canonical **Simm (2007) shaley-sand Gassmann fluid substitution** workflow. Notebook narrative + importable `rockphys/` package + CLI + figure regeneration. This is the scientific home for the Simm paper work.

Sibling repo `../shaley-gassmann` is a smaller engineering prototype (LAS-focused, threshold-style dry-bulk conditioning). **Do not merge formulas between the two repos blindly** — they condition dry rock differently, and any formula change is a scientific decision that needs regression tests first. See [README.md](README.md) "Relationship To shaley-gassmann" for the canonical statement.

## Environment

```bash
uv sync                                         # install deps
uv sync --extra dev                             # also install pytest
uv run pytest -q                                # 7 test files (pipeline, physics, LAS, notebook contract, CLI, etc.)
uv run python scripts/check_inputs.py           # read-only baseline checks
uv run jupyter lab                              # narrative notebook
```

## Entry points

```bash
# Run pipeline with defaults (uses included data/)
uv run python scripts/run_simm_workflow.py --data-dir data

# Regenerate the standard figure set
uv run python scripts/run_simm_workflow.py \
  --data-dir data \
  --output-dir outputs/simm_demo \
  --write-figures
```

From Python:

```python
from rockphys import run_simm_workflow
result = run_simm_workflow()
well = result.well     # DataFrame: depth, facies, Kd_K0, Vp_oil, Vp_gas, Vp_default_gas, …
```

Override scientific defaults without editing module constants — pass a `SimmWorkflowConfig`, `DryRockTrendConfig`, or `FluidProperties` (see README for examples).

## Package layout (`rockphys/`)

| Module | Role |
|---|---|
| `config.py` | Dataclass configuration: `SimmWorkflowConfig`, `DryRockTrendConfig`, `FluidProperties`, `ReservoirProperties` |
| `constants.py` | Mineral moduli, conversion factors |
| `io.py` | Load text-table well/facies inputs |
| `las_io.py` | Optional LAS adapter (`load_las_well`) for real-data experiments |
| `physics.py` | Core rock-physics calculations (Vp/Vs, Kd_K0, Castagna) |
| `substitution.py` | Default Gassmann + Simm adaptive substitution |
| `pipeline.py` | `run_simm_workflow` orchestrator, `FIGURE_BUILDERS`, `DEFAULT_CONFIG` |
| `plotting.py` | Diagnostic figures (`fig1`–`fig6`) |

## Notebooks

| Notebook | Purpose |
|---|---|
| `gassmann_demo.ipynb` | Narrative notebook following the Simm paper — primary scientific reference |
| `notebooks/01_simm_workflow_pipeline.ipynb` | Compact notebook driving the importable pipeline |

## Data

`data/` ships with well 2 logs and facies interval files (text-table format). The optional LAS adapter (`load_las_well`) enables real-data experiments without changing the textbook pipeline.

## Scientific conventions

- Velocities **km/s**, densities **g/cc**, moduli **GPa**
- Quartz: K=36.8 GPa, G=44.0 GPa (defaults in `config.py`)
- Clay: K=15.0 GPa, G=5.0 GPa (defaults)
- Conditioned facies in `DryRockTrendConfig` default to silty sand bands — Simm-style adaptive workflow
- Castagna mudrock line used for VS prediction QC

## Workflow rules

1. **Notebook narrative is the scientific source of truth.** The package mirrors notebook logic; do not let them diverge.
2. **Formula changes need regression tests first.** `tests/test_notebook_contract.py` pins notebook outputs — update it deliberately, not as a side effect.
3. **Treat `shaley-gassmann` as engineering inspiration only** for module shape, dataclass config, CLI patterns. Its scientific formulas are not interchangeable with this repo's.
