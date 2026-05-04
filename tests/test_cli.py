from pathlib import Path

from scripts import run_simm_workflow


ROOT = Path(__file__).resolve().parents[1]


def test_parse_args_uses_safe_defaults():
    args = run_simm_workflow.parse_args([])

    assert args.data_dir == Path("data")
    assert args.output_dir is None
    assert args.write_figures is False


def test_main_runs_pipeline_without_figures(capsys):
    exit_code = run_simm_workflow.main(["--data-dir", str(ROOT / "data")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rows=4117" in captured.out
    assert "figures=0" in captured.out


def test_main_writes_figures_to_output_dir(tmp_path, capsys):
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
    assert "figures=7" in captured.out
    assert (tmp_path / "fig1_well_logs.png").exists()
    assert (tmp_path / "fig6_differences.png").exists()
