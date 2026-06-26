from pathlib import Path

import pytest

from local_profile import marker_by_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MARKERS = marker_by_path(PROJECT_ROOT)


def pytest_collection_modifyitems(items):
    for item in items:
        try:
            relative_path = item.path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            continue
        marker = LOCAL_MARKERS.get(relative_path)
        if marker:
            item.add_marker(getattr(pytest.mark, marker))
