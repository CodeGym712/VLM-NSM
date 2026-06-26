"""Test the RGB dataset label analysis script.

Uses temporary files — no GPU or Robosuite required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_nsm.experiments.analyze_rgb_dataset_labels import analyze_labels


def _make_yolo_label(label_dir: Path, filename: str, lines: list[str]) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    p = label_dir / filename
    p.write_text("\n".join(lines) + "\n")
    return p


def _make_annotation(anno_dir: Path, objects: list[dict]) -> Path:
    anno_dir.mkdir(parents=True, exist_ok=True)
    p = anno_dir / "annotations.jsonl"
    entry = {"image_path": "dummy.png", "objects": objects}
    p.write_text(json.dumps(entry) + "\n")
    return p


class TestAnalyzeRGBDatasetLabels:
    def test_empty_directory(self, tmp_path: Path):
        """No label files → empty stats."""
        stats = analyze_labels(tmp_path)
        assert stats["num_images"] == 0
        assert stats["num_label_files"] == 0
        assert "error" in stats

    def test_single_label(self, tmp_path: Path):
        """Single YOLO label with one object."""
        _make_yolo_label(tmp_path / "labels" / "train", "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        stats = analyze_labels(tmp_path)
        assert stats["num_images"] == 1
        assert stats["num_objects"] == 1
        assert stats["class_distribution"]["0"]["count"] == 1

    def test_multi_class_distribution(self, tmp_path: Path):
        """Multiple classes across multiple label files."""
        label_dir = tmp_path / "labels" / "train"
        _make_yolo_label(label_dir, "img001.txt",
                         ["0 0.5 0.5 0.3 0.4", "1 0.3 0.3 0.2 0.2"])
        _make_yolo_label(label_dir, "img002.txt", ["2 0.7 0.5 0.2 0.3"])
        stats = analyze_labels(tmp_path)
        assert stats["num_images"] == 2
        assert stats["num_objects"] == 3
        assert stats["class_distribution"]["0"]["count"] == 1
        assert stats["class_distribution"]["1"]["count"] == 1
        assert stats["class_distribution"]["2"]["count"] == 1

    def test_objects_per_image_stats(self, tmp_path: Path):
        """Objects per image mean/min/max."""
        label_dir = tmp_path / "labels" / "train"
        _make_yolo_label(label_dir, "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        _make_yolo_label(label_dir, "img002.txt",
                         ["0 0.5 0.5 0.3 0.4", "1 0.3 0.3 0.2 0.2"])
        stats = analyze_labels(tmp_path)
        opi = stats["objects_per_image"]
        assert opi["mean"] == 1.5
        assert opi["min"] == 1
        assert opi["max"] == 2

    def test_bbox_statistics_normalized(self, tmp_path: Path):
        """Bbox width/height/area stats in normalized coords."""
        label_dir = tmp_path / "labels" / "train"
        _make_yolo_label(label_dir, "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        stats = analyze_labels(tmp_path)
        bs = stats["bbox_stats_normalized"]
        assert bs["width_mean"] == pytest.approx(0.3)
        assert bs["height_mean"] == pytest.approx(0.4)
        assert bs["area_mean"] == pytest.approx(0.12)

    def test_yolo_coord_out_of_range(self, tmp_path: Path):
        """Detect coords outside [0, 1]."""
        label_dir = tmp_path / "labels" / "train"
        _make_yolo_label(label_dir, "bad.txt", ["0 1.5 0.5 0.3 0.4"])
        stats = analyze_labels(tmp_path)
        assert stats["yolo_coord_out_of_range_count"] == 1

    def test_missing_bbox_from_annotations(self, tmp_path: Path):
        """Count missing bboxes in annotations.jsonl."""
        _make_yolo_label(tmp_path / "labels" / "train", "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        _make_annotation(tmp_path, [
            {"object_id": "cube1", "color": "red", "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}},
            {"object_id": "table", "color": None, "bbox": None},
        ])
        stats = analyze_labels(tmp_path)
        assert stats["missing_bbox_count"] == 1

    def test_invalid_bbox_zero_area(self, tmp_path: Path):
        """Count zero-area bboxes."""
        _make_yolo_label(tmp_path / "labels" / "train", "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        _make_annotation(tmp_path, [
            {"object_id": "cube1", "color": "red", "bbox": {"x1": 10, "y1": 20, "x2": 10, "y2": 20}},
        ])
        stats = analyze_labels(tmp_path)
        assert stats["invalid_bbox_count"] == 1

    def test_output_file_written(self, tmp_path: Path):
        """Output JSON file is written correctly."""
        _make_yolo_label(tmp_path / "labels" / "train", "img001.txt", ["0 0.5 0.5 0.3 0.4"])
        out_path = tmp_path / "stats.json"
        analyze_labels(tmp_path, output=str(out_path))
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["num_images"] == 1

    def test_invalid_line_skipped(self, tmp_path: Path):
        """Lines with wrong number of parts are skipped."""
        label_dir = tmp_path / "labels" / "train"
        _make_yolo_label(label_dir, "img001.txt", ["0 0.5 0.5 0.3"])  # missing h
        _make_yolo_label(label_dir, "img002.txt", ["0 0.5 0.5 0.3 0.4"])
        stats = analyze_labels(tmp_path)
        assert stats["num_objects"] == 1  # only valid line counted

    def test_all_split_dirs_scanned(self, tmp_path: Path):
        """Labels from all split subdirs are included."""
        _make_yolo_label(tmp_path / "labels" / "train", "a.txt", ["0 0.5 0.5 0.3 0.4"])
        _make_yolo_label(tmp_path / "labels" / "val", "b.txt", ["1 0.5 0.5 0.2 0.2"])
        stats = analyze_labels(tmp_path)
        assert stats["num_images"] == 2
