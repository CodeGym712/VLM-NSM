from pathlib import Path

import pytest

from local_profile import load_local_test_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_manifest_has_unit_and_symbolic_groups():
    manifest = load_local_test_manifest(PROJECT_ROOT)
    assert set(manifest) == {"unit", "symbolic"}
    assert manifest["unit"]
    assert manifest["symbolic"]


def test_local_manifest_paths_are_unique_existing_test_modules():
    manifest = load_local_test_manifest(PROJECT_ROOT)
    paths = [path for group in manifest.values() for path in group]
    assert len(paths) == len(set(paths))
    for relative_path in paths:
        assert relative_path.startswith("tests/test_")
        assert relative_path.endswith(".py")
        assert (PROJECT_ROOT / relative_path).is_file()


def test_local_manifest_rejects_duplicate_paths(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "local_test_manifest.yaml").write_text(
        "unit:\n  - tests/test_example.py\n"
        "symbolic:\n  - tests/test_example.py\n",
        encoding="utf-8",
    )
    (tests_dir / "test_example.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_local_test_manifest(tmp_path)
