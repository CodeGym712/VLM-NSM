"""Tests for Stage 19-D1 training distribution audit."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.stage19_training_distribution_audit import (
    run_training_distribution_audit,
    _scan_file_for_target_stack,
    _scan_json_metadata,
)


class TestScanFileForTargetStack:
    def test_finds_cubes_reverse(self):
        content = 'target_stack = cubes[::-1]  # ["cube2", "cube1"]'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            findings = _scan_file_for_target_stack(Path(f.name))
        Path(f.name).unlink()
        assert len(findings) >= 1
        assert any("cubes[::-1]" in fi["content"] for fi in findings)

    def test_no_target_stack_no_findings(self):
        content = "print('hello world')\nx = 1 + 2"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            findings = _scan_file_for_target_stack(Path(f.name))
        Path(f.name).unlink()
        assert len(findings) == 0


class TestScanJsonMetadata:
    def test_extracts_target_stack(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"target_stack": ["cube2", "cube1"], "cubes": ["cube1", "cube2"]}, f)
            f.flush()
            meta = _scan_json_metadata(Path(f.name))
        Path(f.name).unlink()
        assert meta is not None
        assert meta["target_stack"] == ["cube2", "cube1"]

    def test_non_dict_json_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            meta = _scan_json_metadata(Path(f.name))
        Path(f.name).unlink()
        assert meta is None or meta == {}


class TestTrainingDistributionAudit:
    def test_output_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake source file with target_stack evidence
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            py_file = src_dir / "test_script.py"
            py_file.write_text('target_stack = cubes[::-1]\nprint("demo")')

            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            # Patch search dirs to include our temp
            result = run_training_distribution_audit(
                [str(tmpdir)],
                str(output_path),
                str(md_path),
            )

            assert output_path.exists()
            assert md_path.exists()

    def test_finds_cube1_on_cube2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            py_file = src_dir / "demo.py"
            py_file.write_text('cubes = ["cube1", "cube2"]\ntarget_stack = cubes[::-1]\n')

            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            result = run_training_distribution_audit(
                [str(tmpdir)],
                str(output_path),
                str(md_path),
            )

            assert result["counts"]["cube1_on_cube2"] > 0
            assert result["counts"]["cube2_on_cube1"] == 0

    def test_counts_structure(self):
        """Verify the output has the required count structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            result = run_training_distribution_audit([], str(output_path), str(md_path))

            assert "counts" in result
            for key in ["cube1_on_cube2", "cube2_on_cube1", "red_on_green", "green_on_red"]:
                assert key in result["counts"]
            assert "key_conclusion" in result
