from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"


def _restart_in_utf8_mode(argv: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.utf8_mode:
        return _restart_in_utf8_mode(argv)

    sys.path.insert(0, str(TESTS_DIR))
    from local_profile import load_local_test_manifest

    import pytest

    manifest = load_local_test_manifest(PROJECT_ROOT)
    test_paths = [path for group in manifest.values() for path in group]
    pytest_args = [*test_paths, "-m", "unit or symbolic", "--tb=short", *argv]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
