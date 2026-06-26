"""Online Qwen goal parser via OpenAI-compatible chat completion API.

Parses natural language instructions into structured goal JSON
by calling a Qwen-compatible API. Designed for the E2E language-to-simulation
pipeline as a drop-in replacement for MockGoalParser / CachedQwenGoalParser.

Prompt design notes (see also S14-B):
- The table predicate (on(bottom, table)) is included for alignment-layer
  completeness checking. Physical E2E success verification still uses
  on(top, bottom) only — this is handled downstream in build_stacking_goal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from vlm_nsm.core.goal_parser import extract_json_object
from vlm_nsm.core.data_types import GoalParseResult

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _block_ids(objects: list[str]) -> list[str]:
    return [obj for obj in objects if obj != "table"]


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ": "))


STACKING_PROMPT = """You are a robot instruction parser. Convert natural language instructions into symbolic goal JSON.

Task: stacking blocks
Available objects: ["cube1", "cube2"]
Valid predicates: on(object, object_or_table)
Valid object IDs: ["cube1", "cube2", "table"]

Output JSON schema:
{
  "task_type": "stacking",
  "object_bindings": {
    "top block": "<top_object>",
    "bottom block": "<bottom_object>"
  },
  "goal_predicates": [
    ["on", "<top_object>", "<bottom_object>"],
    ["on", "<bottom_object>", "table"]
  ]
}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- Do NOT invent object IDs beyond those listed above.
- Include the bottom support predicate ["on", bottom, "table"] for symbolic completeness.

Examples:

Instruction: "Put the red cube on the green cube."
{"task_type": "stacking", "object_bindings": {"top block": "cube1", "bottom block": "cube2"}, "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]]}
"""

COARSE_DSL_PROMPT = """You are a robot task planner. Convert natural language instructions into a coarse option program using ONLY semantic-level options.

Available coarse options (USE ONLY THESE):
  - pick(object)          — grasp and lift the object from its current location
  - place(object, target) — put the held object on a target (block or table)
  - move(object, source, target) — pick from source and place on target
  - ensure(predicate)     — verify a predicate holds before proceeding
  - achieve(predicate)    — make a predicate true (decomposed by planner)

FORBIDDEN options (phase-level — DO NOT OUTPUT THESE):
  reach, grasp, lift, reach_place, release_place, retreat

Valid predicates: on(obj, support), ontable(obj), clear(obj), holding(obj), handempty, graspable(obj)

Output JSON schema:
{
  "version": "1.0",
  "steps": [
    {"op": "<pick|place|move|ensure|achieve>", "object": "<obj>", "target": "<tgt>"}
  ]
}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- Use ONLY the object IDs listed in the scene.
- pick requires the object to be reachable and clear.
- place requires you to be holding the object, and target to be clear.
- place on "table" is always allowed.
- ensure(predicate) is a verification gate — use before pick/place to check preconditions.
- achieve(predicate) is for final goals — the executor will plan how to reach them.
- No self-place: never place(obj, obj).
- Think step by step about what preconditions each step needs (e.g., clear target before place).

Examples:

Scene: cube1 on table, cube2 on table, hand empty
Instruction: "Stack cube1 on cube2"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}]}

Scene: cube1 on table, cube2 on cube1, hand empty
Instruction: "Move cube2 from cube1 to the table"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube2"}, {"op": "place", "object": "cube2", "target": "table"}]}

Scene: cube1 on cube2, cube2 on table, hand empty
Instruction: "Unstack cube1 and put it on the table"
{"version": "1.0", "steps": [{"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "table"}]}

Scene: cube1 on table, cube2 on table, cube3 on table, hand empty
Instruction: "Stack cube1 on cube2, then put cube3 on cube1"
{"version": "1.0", "steps": [{"op": "ensure", "predicate": ["clear", "cube2"]}, {"op": "pick", "object": "cube1"}, {"op": "place", "object": "cube1", "target": "cube2"}, {"op": "pick", "object": "cube3"}, {"op": "place", "object": "cube3", "target": "cube1"}]}
"""


class OnlineQwenGoalParser:
    """Qwen goal parser using an OpenAI-compatible chat completion API.

    This works with Alibaba Cloud DashScope compatible mode and with local
    vLLM/SGLang servers that expose ``/v1/chat/completions``.

    Args:
        model: Qwen model name (e.g. "qwen-plus", "qwen3.6-plus").
        api_base: API base URL. If None, uses default OpenAI-compatible endpoint.
        api_key: API key. If None, read from env var (see api_key_env).
        api_key_env: Environment variable name for API key (default: QWEN_API_KEY).
        temperature: Generation temperature (default 0.0 for deterministic).
        max_tokens: Maximum output tokens.
        timeout: API call timeout in seconds.
        retries: Number of retries on parse failure.
        save_raw_outputs: Whether to save raw model outputs to disk.
        raw_output_dir: Directory for raw outputs (required if save_raw_outputs).
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_base: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "QWEN_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
        retries: int = 1,
        save_raw_outputs: bool = False,
        raw_output_dir: str | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or os.environ.get(api_key_env)
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.save_raw_outputs = save_raw_outputs
        self.raw_output_dir = Path(raw_output_dir) if raw_output_dir else None

        if save_raw_outputs and raw_output_dir:
            self.raw_output_dir.mkdir(parents=True, exist_ok=True)

    def parse(
        self,
        instruction: str,
        available_objects: list[str] | None = None,
        objects: list[str] | None = None,
        task_type: str = "stacking",
        n_blocks: int | None = None,
        scene_description: str | None = None,
        **kwargs: Any,
    ) -> GoalParseResult:
        """Parse a natural language instruction into a structured goal or coarse program.

        Args:
            instruction: Natural language instruction.
            available_objects: Optional list of valid object IDs.
            objects: E2E-compatible alias for available object IDs.
            task_type: "stacking" for goal-only, "coarse_stacking" for coarse DSL.
            n_blocks: Optional number of blocks for prompt examples.
            scene_description: Optional scene inventory text.

        Returns:
            GoalParseResult with parsed goal JSON or error details.
        """
        if available_objects is None and objects is not None:
            available_objects = objects
        if available_objects is None:
            available_objects = ["cube1", "cube2"]
        available_objects = [str(obj) for obj in available_objects if str(obj) != "table"]
        if n_blocks is None:
            n_blocks = len(available_objects)

        is_coarse = (task_type == "coarse_stacking")

        prompt = self._build_prompt(
            instruction=instruction,
            available_objects=available_objects,
            task_type=task_type,
            n_blocks=n_blocks,
            scene_description=scene_description,
            is_coarse=is_coarse,
        )
        raw_output = ""
        inference_time = 0.0
        raw_output_path: str | None = None

        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                raw_output = self._call_api(prompt)
            except Exception as exc:
                return GoalParseResult(
                    success=False,
                    goal_json=None,
                    raw_output=raw_output,
                    error=f"api_error: {exc}",
                    inference_time=inference_time,
                )
            inference_time += time.perf_counter() - started

            try:
                goal_json = extract_json_object(raw_output)
            except (json.JSONDecodeError, ValueError):
                if attempt < self.retries:
                    prompt += "\n\nThe previous output was not valid JSON. Output ONLY valid JSON. No explanation."
                continue

            # Save raw output if requested
            if self.save_raw_outputs and self.raw_output_dir:
                safe_name = instruction.strip()[:40].replace(" ", "_").replace("/", "_")
                out_path = self.raw_output_dir / f"raw_{safe_name}.txt"
                out_path.write_text(raw_output)
                raw_output_path = str(out_path)

            return GoalParseResult(
                success=True,
                goal_json=goal_json,
                raw_output=raw_output,
                error=None,
                inference_time=inference_time,
                raw_output_path=raw_output_path,
            )

        return GoalParseResult(
            success=False,
            goal_json=None,
            raw_output=raw_output,
            error="json_parse_error",
            inference_time=inference_time,
            raw_output_path=raw_output_path,
        )

    def _build_prompt(
        self,
        instruction: str,
        available_objects: list[str] | None = None,
        task_type: str = "stacking",
        n_blocks: int | None = None,
        scene_description: str | None = None,
        is_coarse: bool = False,
    ) -> str:
        """Build a schema-constrained parser prompt.

        If is_coarse=True, uses the COARSE_DSL_PROMPT to generate pick/place/move
        programs instead of goal-only predicates.
        """
        objects = available_objects or ["cube1", "cube2"]
        objects = _block_ids([str(obj) for obj in objects])
        if not objects:
            objects = ["cube1", "cube2"]
        if n_blocks is not None:
            objects = objects[:n_blocks]
        valid_ids = [*objects, "table"]

        if is_coarse:
            return self._build_coarse_prompt(
                instruction=instruction,
                objects=objects,
                valid_ids=valid_ids,
                scene_description=scene_description,
            )

        if len(objects) >= 3:
            top, middle, bottom = objects[0], objects[1], objects[2]
            example_goal = {
                "task_type": task_type,
                "object_bindings": {
                    "top block": top,
                    "middle block": middle,
                    "bottom block": bottom,
                },
                "goal_predicates": [
                    ["on", top, middle],
                    ["on", middle, bottom],
                    ["on", bottom, "table"],
                ],
            }
            example_instruction = f"Stack {top} on {middle}, and {middle} on {bottom}."
        else:
            top = objects[0]
            bottom = objects[1] if len(objects) > 1 else objects[0]
            example_goal = {
                "task_type": task_type,
                "object_bindings": {
                    "top block": top,
                    "bottom block": bottom,
                },
                "goal_predicates": [
                    ["on", top, bottom],
                    ["on", bottom, "table"],
                ],
            }
            example_instruction = f"Put {top} on {bottom}."

        lines = [
            "You are a robot instruction parser. Convert natural language instructions into symbolic goal JSON.",
            "",
            f"Task: {task_type} blocks",
            f"Available objects: {json.dumps(objects, ensure_ascii=False)}",
            f"Valid object IDs: {json.dumps(valid_ids, ensure_ascii=False)}",
            "Valid predicates: on(block, block_or_table)",
        ]
        if scene_description:
            lines += ["", "Scene inventory:", scene_description]
        lines += [
            "",
            "Output JSON schema:",
            "{",
            '  "task_type": "stacking",',
            '  "object_bindings": {"top block": "<top_object>", "bottom block": "<bottom_object>"},',
            '  "goal_predicates": [["on", "<top_object>", "<bottom_object>"], ["on", "<bottom_object>", "table"]]',
            "}",
        ]
        if scene_description:
            lines += [
                "",
                'If the instruction refers to a color or object that is NOT present in the scene inventory, output:',
                "{",
                '  "task_type": "stacking",',
                '  "object_bindings": {},',
                '  "goal_predicates": [],',
                '  "status": "ungroundable",',
                '  "reason": "<explain what is missing>"',
                "}",
            ]
        lines += [
            "",
            "Rules:",
            "- Output ONLY valid JSON. No explanation, no markdown, no code fences.",
            "- Do NOT invent object IDs beyond the valid object IDs listed above.",
            "- Use canonical object IDs such as cube1/cube2, not color names.",
            "- Include on(bottom, table) for symbolic alignment completeness.",
            "- Physical execution success may check only on(top, bottom); still include table support in parser output.",
        ]
        if scene_description:
            lines += [
                "- If an instruction mentions a color or object NOT in the scene, output status=ungroundable.",
                "- Do NOT map a missing color name to an existing object (e.g. 'yellow' when only red/green exist).",
                "- Spatial attributes (front/back, near/far, left/right, closer/farther) CAN be inferred",
                "  from the object pose coordinates listed in the scene inventory.",
            ]
        lines += [
            "",
            "Example:",
            f"Instruction: {example_instruction}",
            _json_line(example_goal),
        ]
        if scene_description:
            lines += [
                "",
                'Instruction: "Stack the yellow cube on the red cube."',
                _json_line({
                    "task_type": "stacking",
                    "object_bindings": {},
                    "goal_predicates": [],
                    "status": "ungroundable",
                    "reason": "yellow cube is not present in the scene inventory",
                }),
            ]
        lines += [
            "",
            f"Instruction: {instruction}",
        ]
        return "\n".join(lines)

    def _build_coarse_prompt(
        self,
        instruction: str,
        objects: list[str],
        valid_ids: list[str],
        scene_description: str | None = None,
    ) -> str:
        """Build a coarse DSL prompt for pick/place/move program generation."""
        lines = [
            "You are a robot task planner. Convert natural language instructions into a coarse option program using ONLY semantic-level options.",
            "",
            "Available coarse options (USE ONLY THESE):",
            "  - pick(object)          — grasp and lift the object from its current location",
            "  - place(object, target) — put the held object on a target (block or table)",
            "  - move(object, source, target) — pick from source and place on target",
            "  - ensure(predicate)     — verify a predicate holds before proceeding",
            "  - achieve(predicate)    — make a predicate true (decomposed by planner)",
            "",
            "FORBIDDEN options (phase-level — DO NOT OUTPUT THESE):",
            "  reach, grasp, lift, reach_place, release_place, retreat",
            "",
            "Valid predicates: on(obj, support), ontable(obj), clear(obj), holding(obj), handempty, graspable(obj)",
            "",
            f"Available objects: {json.dumps(objects, ensure_ascii=False)}",
            f"Valid object IDs: {json.dumps(valid_ids, ensure_ascii=False)}",
        ]
        if scene_description:
            lines += ["", "Scene inventory:", scene_description]
        lines += [
            "",
            "Output JSON schema:",
            "{",
            '  "version": "1.0",',
            '  "steps": [',
            '    {"op": "<pick|place|move|ensure|achieve>", "object": "<obj>", "target": "<tgt>"}',
            "  ]",
            "}",
            "",
            "Rules:",
            "- Output ONLY valid JSON. No explanation, no markdown, no code fences.",
            "- Use ONLY the object IDs listed above.",
            "- pick requires the object to be reachable and clear.",
            "- place requires you to be holding the object, and target to be clear.",
            '- place on "table" is always allowed.',
            "- ensure(predicate) is a verification gate — use before pick/place to check preconditions.",
            "- achieve(predicate) is for final goals — the executor will plan how to reach them.",
            "- No self-place: never place(obj, obj).",
            "- Think step by step about what preconditions each step needs.",
            "",
            "Examples:",
            "",
            f'Scene: {", ".join(objects)} on table, hand empty',
            f'Instruction: "Stack {objects[0]} on {objects[1]}"',
            json.dumps({
                "version": "1.0",
                "steps": [
                    {"op": "ensure", "predicate": ["clear", objects[1]]},
                    {"op": "pick", "object": objects[0]},
                    {"op": "place", "object": objects[0], "target": objects[1]},
                ],
            }, ensure_ascii=False),
            "",
        ]
        if len(objects) >= 3:
            lines += [
                f'Scene: {", ".join(objects)} on table, hand empty',
                f'Instruction: "Stack {objects[0]} on {objects[1]}, then put {objects[2]} on {objects[0]}"',
                json.dumps({
                    "version": "1.0",
                    "steps": [
                        {"op": "ensure", "predicate": ["clear", objects[1]]},
                        {"op": "pick", "object": objects[0]},
                        {"op": "place", "object": objects[0], "target": objects[1]},
                        {"op": "pick", "object": objects[2]},
                        {"op": "place", "object": objects[2], "target": objects[0]},
                    ],
                }, ensure_ascii=False),
                "",
            ]
        lines += [
            'Scene: cube1 on table, cube2 on cube1, hand empty',
            'Instruction: "Unstack cube2 and put it on the table"',
            json.dumps({
                "version": "1.0",
                "steps": [
                    {"op": "ensure", "predicate": ["clear", "cube2"]},
                    {"op": "pick", "object": "cube2"},
                    {"op": "place", "object": "cube2", "target": "table"},
                ],
            }, ensure_ascii=False),
            "",
            f"Instruction: {instruction}",
        ]
        return "\n".join(lines)

    def _call_api(self, prompt: str) -> str:
        """Call Qwen-compatible chat completion API.

        Uses the OpenAI client library. Raises on transport/auth errors;
        returns the model output text on success.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OnlineQwenGoalParser requires 'openai' package") from exc

        if not self.api_key:
            raise ValueError(
                f"API key not found. Set the {self.api_key_env} environment variable "
                f"or pass api_key directly."
            )

        client_kwargs: dict[str, Any] = {}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        client = OpenAI(api_key=self.api_key, **client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""
