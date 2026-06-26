"""Dataset manifest dataclasses for standardized experiment artifact tracking.

See docs/dataset_card_v0_2.md for the full dataset description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class DatasetComponent:
    """A single component within a dataset manifest.

    Each component represents a logical collection of experimental
    artifacts (e.g., parse-only evaluation, E2E demos, control policy).
    """

    name: str
    version: str
    component_type: str
    source_paths: list[str]
    sample_count: int
    task_types: list[str]
    parser_modes: list[str] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


@dataclass
class DatasetManifest:
    """Top-level dataset manifest aggregating multiple components.

    Serialises to JSON for paper reproducibility and downstream tooling.
    """

    dataset_name: str
    dataset_version: str
    created_at: str
    project_stage: str
    components: list[DatasetComponent]
    global_metrics: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        stage: str,
        components: list[DatasetComponent],
        global_metrics: dict[str, Any] | None = None,
        limitations: list[str] | None = None,
    ) -> DatasetManifest:
        return cls(
            dataset_name=name,
            dataset_version=version,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            project_stage=stage,
            components=components,
            global_metrics=global_metrics or {},
            limitations=limitations or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "project_stage": self.project_stage,
            "components": [c.to_dict() for c in self.components],
            "global_metrics": self.global_metrics or {},
            "limitations": self.limitations or [],
        }

    def write_json(self, path: str) -> None:
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load_json(cls, path: str) -> DatasetManifest:
        import pathlib
        raw = json.loads(pathlib.Path(path).read_text())
        components = [DatasetComponent(**c) for c in raw.pop("components", [])]
        return cls(components=components, **raw)
