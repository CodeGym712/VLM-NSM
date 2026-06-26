"""Test build_dataset_manifest.py using mock file fixtures.

No DashScope, Robosuite, or GPU required.
"""

import json
import pytest
from pathlib import Path

from vlm_nsm.experiments.build_dataset_manifest import build_manifest


def _make_stage15_fixture(base: Path):
    """Create mock Stage 15 result files."""
    s15 = base / "generated" / "results" / "stage_15"
    s15.mkdir(parents=True, exist_ok=True)

    # parse5 summary
    p5 = s15 / "dashscope_qwen_parse_5"
    p5.mkdir(parents=True)
    (p5 / "summary.json").write_text(json.dumps({
        "num_demos": 5, "results": [{"instruction": f"test{i}"} for i in range(5)]
    }))

    # parse50 metrics + summary
    p50 = s15 / "dashscope_qwen_parse_50"
    p50.mkdir(parents=True)
    (p50 / "metrics.json").write_text(json.dumps({
        "total": 51, "fully_successful": 51,
        "parse_ok": 51, "schema_valid": 51, "alignment_success": 51,
        "target_stack_success": 51, "planner_success": 51,
        "failure_taxonomy": {},
    }))
    (p50 / "summary.json").write_text(json.dumps({
        "total": 51, "results": [{"instruction": f"t{i}"} for i in range(51)]
    }))

    # e2e5 standardized summary
    e5 = s15 / "e2e_dashscope_qwen_summary_v2"
    e5.mkdir(parents=True)
    (e5 / "summary.json").write_text(json.dumps({
        "num_demos": 5, "results": [
            {"success": True, "instruction": "test"}
            for _ in range(5)
        ],
    }))

    # e2e10 summary
    e10 = s15 / "e2e_dashscope_qwen_parser_driven_10"
    e10.mkdir(parents=True)
    (e10 / "summary.json").write_text(json.dumps({
        "num_demos": 10, "results": [
            {"success": True, "instruction": f"demo{i}"}
            for i in range(10)
        ],
    }))

    return s15


def _make_stage6b_fixture(base: Path):
    """Create mock Stage 6B language eval."""
    s6b = base / "generated" / "results" / "stage_6B"
    s6b.mkdir(parents=True)
    (s6b / "language_variation_summary.json").write_text(json.dumps([
        {
            "task": "stacking",
            "parser": "qwen_text",
            "sample_count": 20,
            "planner_success_pct": 100.0,
            "execution_success_pct": 95.0,
        },
        {
            "task": "hanoi",
            "parser": "qwen_text",
            "sample_count": 10,
            "planner_success_pct": 90.0,
            "execution_success_pct": 85.0,
        },
    ]))


def _make_stage13_fixture(base: Path):
    """Create mock Stage 13 cached Qwen E2E demo."""
    s13 = base / "generated" / "results" / "stage_13" / "e2e_cached_qwen_demo"
    s13.mkdir(parents=True)
    (s13 / "summary.json").write_text(json.dumps({
        "num_demos": 5,
    }))


def _make_stage10_ctrl_fixture(base: Path):
    """Create mock Stage 10 control learning results."""
    s10 = base / "generated" / "results" / "stage_10"
    s10.mkdir(parents=True)
    for seed in range(3):
        (s10 / f"reach_pick_only_seed{seed}_50ep.json").write_text(json.dumps({
            "total_episodes": 50, "success_rate": 0.92,
        }))


def _make_media_fixture(base: Path):
    """Create mock demo media assets."""
    media = base / "assets" / "demo" / "v0_1_1"
    media.mkdir(parents=True)
    (media / "demo_1.mp4").write_text("mock video")
    (media / "demo_manifest.md").write_text("# Demo Media")


@pytest.fixture
def stage15_fixture(tmp_path):
    """Fixture that creates a complete mock result tree."""
    _make_stage15_fixture(tmp_path)
    _make_stage6b_fixture(tmp_path)
    _make_stage13_fixture(tmp_path)
    _make_stage10_ctrl_fixture(tmp_path)
    _make_media_fixture(tmp_path)
    return tmp_path


class TestBuildManifest:
    def test_build_manifest_components(self, stage15_fixture, tmp_path):
        """Should discover all 9 components."""
        cwd = tmp_path  # use tmp_path as project root

        # Need to temporarily chdir to tmp_path, or patch paths
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
                include_missing="ok",
            )
            assert len(manifest.components) == 9, f"Expected 9 components, got {len(manifest.components)}"
            names = [c.name for c in manifest.components]
            assert "dashscope_qwen_parse5_stage15" in names
            assert "dashscope_qwen_parse50_stage15" in names
            assert "dashscope_qwen_e2e5_stage15" in names
            assert "dashscope_qwen_e2e10_stage15" in names
            assert "language_eval_v0_1" in names
            assert "cached_qwen_e2e_v0_1_1" in names
            assert "control_learning_reach_pick_v0_1" in names
            assert "demo_media_v0_1" in names
            assert "stage15_artifacts" in names
        finally:
            os.chdir(old_cwd)

    def test_sample_counts(self, stage15_fixture, tmp_path):
        """Should read correct sample counts from mock data."""
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
            )
            counts = {c.name: c.sample_count for c in manifest.components}
            assert counts["dashscope_qwen_parse5_stage15"] == 5
            assert counts["dashscope_qwen_parse50_stage15"] == 51
            assert counts["dashscope_qwen_e2e5_stage15"] == 5
            assert counts["dashscope_qwen_e2e10_stage15"] == 10
        finally:
            os.chdir(old_cwd)

    def test_stage15_metrics(self, stage15_fixture, tmp_path):
        """Parse50 should have full metrics."""
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
            )
            parse50 = [c for c in manifest.components if c.name == "dashscope_qwen_parse50_stage15"][0]
            assert parse50.metrics["total"] == 51
            assert parse50.metrics["fully_successful"] == 51
            assert parse50.metrics["failure_taxonomy"] == {}
        finally:
            os.chdir(old_cwd)

    def test_global_metrics(self, stage15_fixture, tmp_path):
        """Global metrics should include component count and sample total."""
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
            )
            gm = manifest.global_metrics
            assert gm["total_components"] == 9
            assert gm["total_tested_samples"] >= 0
        finally:
            os.chdir(old_cwd)

    def test_limitations_present(self, stage15_fixture, tmp_path):
        """Manifest should include known limitations."""
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
            )
            assert len(manifest.limitations) >= 5
            has_2block = any("2-block" in l for l in manifest.limitations)
            has_no_vision = any("vision" in l.lower() for l in manifest.limitations)
            assert has_2block, "Should mention 2-block limitation"
            assert has_no_vision, "Should mention no vision grounding"
        finally:
            os.chdir(old_cwd)

    def test_manifest_writes_json(self, stage15_fixture, tmp_path):
        """to_dict should produce valid JSON-serializable output."""
        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            manifest = build_manifest(
                stage15_root=str(tmp_path / "generated" / "results" / "stage_15"),
            )
            d = manifest.to_dict()
            json_str = json.dumps(d, indent=2)
            parsed = json.loads(json_str)
            assert parsed["dataset_name"] == "VLM-NSM Dataset Manifest"
            assert len(parsed["components"]) == 9
        finally:
            os.chdir(old_cwd)
