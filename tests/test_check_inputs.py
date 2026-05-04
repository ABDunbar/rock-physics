from pathlib import Path
import subprocess
import sys

from scripts import check_inputs


def test_manifest_captures_demo_data_and_outputs():
    assert "README.md" in check_inputs.REQUIRED_DATA_FILES
    assert "gassmann_demo.ipynb" in check_inputs.REQUIRED_DATA_FILES
    assert "rockphys/pipeline.py" in check_inputs.REQUIRED_DATA_FILES
    assert "data/well_2.txt" in check_inputs.REQUIRED_DATA_FILES
    assert "data/well2_clnSand.txt" in check_inputs.FACIES_FILES
    assert "fig2_kd_k0_template.png" in check_inputs.EXPECTED_FIGURES


def test_rock_physics_baseline_checks_pass_without_failures():
    results = check_inputs.run_checks()

    failures = [result for result in results if result.status == "FAIL"]

    assert failures == []
    assert any(result.name == "Well 2 table" for result in results)
    assert any(result.name == "Rock physics smoke" for result in results)


def test_cli_runs_from_repo_root():
    completed = subprocess.run(
        [sys.executable, "scripts/check_inputs.py"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "0 FAIL" in completed.stdout
