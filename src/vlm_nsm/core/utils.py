from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def predicate_from_list(items: list[str]) -> tuple[str, ...]:
    if not items:
        raise ValueError("predicate list cannot be empty")
    return tuple(str(x).lower() for x in items)
