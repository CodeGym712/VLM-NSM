from __future__ import annotations
#Language->Goal
import json
import re
import time
from pathlib import Path
from typing import Any

from vlm_nsm.core.data_types import GoalParseResult


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from text.

    Supports:
    - Pure JSON string
    - ```json ... ``` code blocks (with or without json/JSON marker)
    - JSON with explanatory text before/after
    - Extra whitespace and newlines
    """
    candidate = text.strip()
    # Try stripping markdown code block markers first
    match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", candidate, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()

    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object found in text")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = candidate[start : idx + 1]
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("extracted JSON is not an object")
                return parsed

    raise ValueError("incomplete JSON object (unmatched braces)")


def _strip_markdown_code_block(text: str) -> str:
    match = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    return text


def parse_json_from_text(text: str) -> dict[str, Any]:
    """Legacy wrapper around extract_json_object."""
    return extract_json_object(text)


class MockGoalParser:
    def parse(self, instruction: str, task_type: str, objects: list[str], n_blocks: int, source_peg: str = "peg1", target_peg: str = "peg3") -> GoalParseResult:
        if task_type == "hanoi":
            return self._parse_hanoi(instruction, objects, n_blocks, source_peg, target_peg)
        if task_type == "stacking":
            return self._parse_stacking(instruction, objects, n_blocks)
        return GoalParseResult(False, None, "", f"mock parser does not support task_type={task_type}")

    def _parse_hanoi(self, instruction: str, objects: list[str], n_blocks: int, source_peg: str, target_peg: str) -> GoalParseResult:
        disks = [f"cube{i}" for i in range(1, n_blocks + 1)]
        missing = [d for d in disks if d not in objects]
        if missing:
            return GoalParseResult(False, None, "", f"missing objects: {missing}")
        goals = [["on", disks[i], disks[i + 1]] for i in range(len(disks) - 1)]
        goals.append(["on", disks[-1], target_peg])
        goal_json = {
            "task_type": "hanoi",
            "object_bindings": {
                "smallest disk": disks[0],
                "largest disk": disks[-1],
                "source peg": source_peg,
                "target peg": target_peg,
            },
            "goal_predicates": goals,
        }
        return GoalParseResult(True, goal_json, str(goal_json), None)

    @staticmethod
    def _parse_stacking(instruction: str, objects: list[str], n_blocks: int) -> GoalParseResult:
        cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
        missing = [c for c in cubes if c not in objects]
        if missing:
            return GoalParseResult(False, None, "", f"missing objects: {missing}")
        goals = [["on", cubes[i], cubes[i + 1]] for i in range(len(cubes) - 1)]
        goals.append(["on", cubes[-1], "table"])
        goal_json = {
            "task_type": "stacking",
            "object_bindings": {
                "top block": cubes[0],
                "bottom block": cubes[-1],
            },
            "goal_predicates": goals,
        }
        return GoalParseResult(True, goal_json, str(goal_json), None)


class QwenTextGoalParser:
    def __init__(
        self,
        model_path: str,
        prompt_path: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        retry: int = 1,
    ):
        self.model_path = model_path
        self.prompt_path = Path(prompt_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.retry = retry
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("QwenTextGoalParser requires transformers and torch") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        # Clear generation config defaults that may not be supported
        # by all model architectures, avoiding harmless warnings.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def parse(
        self,
        instruction: str,
        objects: list[str],
        valid_predicates: list[str],
        type_table: dict,
        task_type: str = "hanoi",
        n_blocks: int | None = None,
        target_peg: str | None = None,
    ) -> GoalParseResult:
        raw_output = ""
        inference_time = 0.0
        input_tokens = 0
        output_tokens = 0

        for attempt in range(self.retry + 1):
            prompt = self._build_prompt(
                instruction=instruction,
                objects=objects,
                valid_predicates=valid_predicates,
                type_table=type_table,
                n_blocks=n_blocks,
                target_peg=target_peg,
                retry_prompt=attempt > 0,
            )
            started = time.perf_counter()
            raw_output, current_input_tokens, current_output_tokens = self._generate(prompt)
            input_tokens += current_input_tokens
            output_tokens += current_output_tokens
            inference_time += time.perf_counter() - started
            try:
                goal_json = extract_json_object(raw_output)
                return GoalParseResult(
                    True,
                    goal_json,
                    raw_output,
                    None,
                    inference_time=inference_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except (json.JSONDecodeError, ValueError):
                if attempt >= self.retry:
                    break

        return GoalParseResult(
            False,
            None,
            raw_output,
            "json_parse_error",
            inference_time=inference_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_prompt(
        self,
        instruction: str,
        objects: list[str],
        valid_predicates: list[str],
        type_table: dict,
        n_blocks: int | None,
        target_peg: str | None,
        retry_prompt: bool,
    ) -> str:
        prompt = self.prompt_template.format(
            instruction=instruction,
            objects=json.dumps(objects, ensure_ascii=False),
            valid_predicates=json.dumps(valid_predicates, ensure_ascii=False),
            type_table=json.dumps(type_table, ensure_ascii=False, sort_keys=True),
            n_blocks="" if n_blocks is None else n_blocks,
            target_peg="" if target_peg is None else target_peg,
        )
        if retry_prompt:
            prompt += "\n\nThe previous output was not valid JSON. Only output valid JSON. No explanation."
        return prompt

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{prompt}\n"

        inputs = self.tokenizer([text], return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])
        target_device = self._input_device()
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature and self.temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)
        output_ids = generated[0][input_tokens:]
        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return raw_output, input_tokens, int(output_ids.shape[-1])

    def _input_device(self):
        if self.device != "auto":
            return self.device
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return "cpu"
