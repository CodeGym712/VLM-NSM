# OptionContract Audit Report

> 日期：2026-06-24

## 1. No-effect Options

reach, lift, reach_place 三个 option 的 `expected_effects` 为空（只移动 EEF，不改变谓词）。

**语义决策**：空 expected_effects → `effect_satisfied = True`（vacuously satisfied，真空真）。

**验证**：option trace sample 显示 reach/lift/reach_place 的 effect_satisfied=True ✅。

## 2. ReleasePlace Expected Effects

当前 contract：
```json
{
  "add": [["on","{target_object}","{target_location}"], ["clear","{target_object}"], ["handempty"]],
  "del": [["holding","{target_object}"]]
}
```

**问题**：当 target_location 是 "table" 时，不应该生成 `on(obj, table)`——应该生成 `ontable(obj)`。当前 contract 不区分对象目标和桌面目标。

**建议**：在 check_effects 中动态判断 target_location 是否为 "table"，并替换 expected effects 中的 on → ontable。

## 3. Grasp Expected Effects Mismatch

Contract 期望 `holding(cube1)` 但 `physical_state_to_predicates` 不生成 holding 谓词（它基于物理接触检测，Robosuite 的 has_object 字段不被 physical extractor 使用）。

**状态**：grasp 的 effect_satisfied=False 是 physical extractor 的限制，不是 contract 语义问题。

## 4. FailureType Transition Status

已将 FailureType enum 作为共享定义。以下文件仍使用旧字符串匹配：

| File | Line(s) | String Used |
|------|------|------|
| grasp_state_machine.py | 418 | "failure_subtype" → "nocontact"/"slip"/"prereach"/"horizon" |
| place_state_machine.py | 540 | "final_failure_subtype" → "preplace"/"verify_failed"/"place_failed" |
| operator_monitor.py | classify_failure() | `ft.lower().strip()` in string set |
| task_runner.py | 219 | `result.failure_type or f"{skill_name}_failed"` |
| act_skill_executor.py | — | "horizon" |

仍未迁移到 FailureType enum。这些是正常功能代码，迁移需要逐个文件替换——不是本 Phase 目标。

## 5. Trace Sample

`generated/results/upper_ns/option_trace_sample.json` — 1 episode, 5 skills, pre/post predicates + effect_satisfied per skill.
