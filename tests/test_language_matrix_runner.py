"""Test run_language_matrix with mock parser."""

import csv
import subprocess
import sys
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT


def test_matrix_mock_runner(tmp_path):
    """Run matrix with mock parser, verify output files and manifest."""
    cmd = [
        sys.executable, "-m", "vlm_nsm.experiments.run_language_matrix",
        "--models", "mock",
        "--tasks", "hanoi,stacking",
        "--difficulty", "hard",
        "--prompt_variants", "full,weak",
        "--alignment_modes", "full,no_alignment",
        "--skip_existing", "false",
        "--output_dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Matrix runner failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"

    # Check manifest
    manifest_path = tmp_path / "language_matrix_manifest.csv"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # 2 tasks x 2 prompts x 2 alignments

    # Check result files exist and have content
    for row in rows:
        assert row["status"] in ("completed",), f"Entry {row['run_id']} has status {row['status']}"
        result_path = Path(row["result_path"])
        assert result_path.exists(), f"Missing {result_path}"

        import json
        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert len(data) > 0, f"Empty results in {result_path}"


def test_matrix_skip_existing(tmp_path):
    """Test skip_existing works: create one file first."""
    # Create a mock result file
    import json
    fpath = tmp_path / "language_variation_hanoi_hard_mock_full_full.json"
    fpath.write_text(json.dumps([{"sample_id": "test", "goal_success": True}]), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "vlm_nsm.experiments.run_language_matrix",
        "--models", "mock",
        "--tasks", "hanoi",
        "--difficulty", "hard",
        "--prompt_variants", "full",
        "--alignment_modes", "full",
        "--skip_existing", "true",
        "--output_dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0

    manifest_path = tmp_path / "language_matrix_manifest.csv"
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"


def test_matrix_skip_existing_false(tmp_path):
    """Test skip_existing=false overwrites existing files."""
    import json
    fpath = tmp_path / "language_variation_hanoi_hard_mock_full_full.json"
    fpath.write_text(json.dumps([{"old": "data"}]), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "vlm_nsm.experiments.run_language_matrix",
        "--models", "mock",
        "--tasks", "hanoi",
        "--difficulty", "hard",
        "--prompt_variants", "full",
        "--alignment_modes", "full",
        "--skip_existing", "false",
        "--output_dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0

    manifest_path = tmp_path / "language_matrix_manifest.csv"
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "completed"
