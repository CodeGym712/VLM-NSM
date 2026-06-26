from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from vlm_nsm.core.data_types import Action, PlanResult
from vlm_nsm.core.utils import PROJECT_ROOT


class MetricFFAdapter:
    def __init__(self, metric_ff_binary: str | None = None, search_mode: int = 0):
        self.metric_ff_binary = metric_ff_binary or self._resolve_binary()
        self.search_mode = search_mode

    def solve(self, domain_path, problem_path) -> PlanResult:
        if not self.metric_ff_binary:
            return PlanResult(False, [], "", "Metric-FF binary not found; set METRIC_FF_BIN or metric_ff_binary")
        cmd = [self.metric_ff_binary, "-o", str(domain_path), "-f", str(problem_path), "-s", str(self.search_mode)]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output = proc.stdout or ""
        if proc.returncode != 0 and "found legal plan" not in output.lower() and "empty plan solves it" not in output.lower():
            return PlanResult(False, [], output, f"Metric-FF exited with code {proc.returncode}")
        low = output.lower()
        failure_markers = ["not found", "unknown constant", "goal can be simplified to false", "unsolvable", "error"]
        if any(marker in low for marker in failure_markers):
            return PlanResult(False, [], output, "planner failed")
        if "empty plan solves it" in low:
            return PlanResult(True, [], output, None)
        actions = self._parse_actions(output)
        if not actions:
            return PlanResult(False, [], output, "no plan actions parsed")
        return PlanResult(True, actions, output, None)

    def _resolve_binary(self) -> str:
        candidates = [
            os.environ.get("METRIC_FF_BIN"),
            str(PROJECT_ROOT / "Metric-FF-v2.1" / "ff"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return str(Path(candidate))
        return ""

    @staticmethod
    def _parse_actions(output: str) -> list[Action]:
        try:
            section = output.split("ff: found legal plan as follows", 1)[1]
            section = section.split("time spent:", 1)[0]
        except IndexError:
            return []
        actions: list[Action] = []
        for line in section.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            _, action_text = line.split(":", 1)
            tokens = re.findall(r"[A-Za-z0-9_-]+", action_text.lower())
            if tokens:
                actions.append(Action(tokens[0], tokens[1:]))
        return actions
