from pathlib import Path

import gassmann_demo


ROOT = Path(__file__).resolve().parents[1]


def test_demo_script_uses_shared_pipeline():
    source = (ROOT / "gassmann_demo.py").read_text(encoding="utf-8")

    assert "run_simm_workflow" in source
    assert "SimmWorkflowConfig" in source


def test_demo_script_runs_pipeline_and_writes_figures(tmp_path, capsys):
    result = gassmann_demo.main(data_dir=ROOT / "data", output_dir=tmp_path)

    captured = capsys.readouterr()
    assert len(result.well) == 4117
    assert "figures=7" in captured.out
    assert "Done." in captured.out
    assert (tmp_path / "fig1_well_logs.png").exists()
    assert (tmp_path / "fig6_differences.png").exists()
