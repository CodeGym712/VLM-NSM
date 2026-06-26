from __future__ import annotations

from pathlib import Path, PurePosixPath

import yaml


MANIFEST_PATH = Path("tests/local_test_manifest.yaml")
MARKERS = ("unit", "symbolic")


def load_local_test_manifest(project_root: Path) -> dict[str, list[str]]:
    project_root = Path(project_root).resolve()
    manifest_path = project_root / MANIFEST_PATH
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != set(MARKERS):
        raise ValueError(f"local test manifest must contain exactly: {', '.join(MARKERS)}")

    manifest: dict[str, list[str]] = {}
    seen: set[str] = set()
    for marker in MARKERS:
        entries = data[marker]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"local test manifest group {marker!r} must be a non-empty list")
        manifest[marker] = []
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError(f"local test path must be a string: {entry!r}")
            normalized = PurePosixPath(entry).as_posix()
            if normalized in seen:
                raise ValueError(f"duplicate local test path: {normalized}")
            if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
                raise ValueError(f"local test path must stay inside the repository: {normalized}")
            if not (project_root / normalized).is_file():
                raise ValueError(f"local test path does not exist: {normalized}")
            seen.add(normalized)
            manifest[marker].append(normalized)
    return manifest


def marker_by_path(project_root: Path) -> dict[str, str]:
    manifest = load_local_test_manifest(project_root)
    return {
        relative_path: marker
        for marker, relative_paths in manifest.items()
        for relative_path in relative_paths
    }
