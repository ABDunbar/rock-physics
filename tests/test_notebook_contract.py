import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NOTEBOOK = ROOT / "notebooks/01_simm_workflow_pipeline.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(PIPELINE_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_pipeline_notebook_uses_importable_simm_pipeline():
    source = _notebook_source()

    assert "SimmWorkflowConfig" in source
    assert "run_simm_workflow" in source


def test_pipeline_notebook_keeps_manual_formula_warning():
    source = _notebook_source()

    assert "Treat formula changes as scientific decisions" in source
