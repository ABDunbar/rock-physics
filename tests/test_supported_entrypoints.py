from pathlib import Path

from scripts import run_simm_workflow


ROOT = Path(__file__).resolve().parents[1]


def test_cli_entrypoint_runs_pipeline_and_writes_figures(tmp_path, capsys):
    exit_code = run_simm_workflow.main(
        [
            "--data-dir",
            str(ROOT / "data"),
            "--output-dir",
            str(tmp_path),
            "--write-figures",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rows=4117" in captured.out
    assert "figures=7" in captured.out
    assert (tmp_path / "fig1_well_logs.png").exists()
    assert (tmp_path / "fig6_differences.png").exists()
