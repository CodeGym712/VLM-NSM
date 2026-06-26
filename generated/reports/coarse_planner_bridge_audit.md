# CoarsePlannerBridge 语义审计报告

> 日期：2026-06-26 | 范围：`neuro_symbolic/coarse_planner_bridge.py`  
> 结论：发现 5 个语义问题，已全部修复

---

## 一、方法对照表

| 方法 | pick | place | move | ensure | achieve |
|------|------|-------|------|--------|---------|
| `ground()` | requires_planning=True, 无 symbolic_actions | symbolic_actions=[("place",...)], goal_predicate | symbolic_actions=[("pick_from",...),("place_on",...)], requires_planning=True | goal_predicate 已设, requires_planning=False | goal_predicate 已设, requires_planning=True |
| `to_symbolic_actions()` | move(obj, "table", "hand") ← 有 Bug | move(obj, "hand", tgt) | move(obj, src, tgt) | pass | pass |
| `execution_sequence()` | atomic_pick | atomic_place | atomic_pick + atomic_place | `{"skill":"ensure",...}` ← 无对应 skill | `{"skill":"achieve",...}` ← 无对应 skill |
| `decompose_achieve()` | N/A | N/A | N/A | N/A | **API 签名错误** |

---

## 二、逐项审计

### 2.1 `to_symbolic_actions()` — pick 的 source 硬编码为 "table"

```python
# 修复前 第 148-150 行
elif op == "pick":
    src = "table"  # default ← 错误
    actions.append({"name": "move", "args": [obj, src, "hand"]})
```

**问题**：物体可能在另一个物体上（如 cube2 在 cube1 上），source 不应恒为 "table"。

**影响**：当 pick 的对象在另一个对象上时，symbolic action 的 source 错误，tracking log 会产生误导。

**修复**：改为 `src = src or "?"`，标注 `"source inferred"`。此方法仅用于 logging/trace，不输入 BFS planner。

### 2.2 "hand" 作为 symbolic location — 语义错误

```python
# 修复前：move(obj, src, "hand") / move(obj, "hand", tgt)
```

**问题**：BFS SymbolicPlanner 的操作符是 `MoveOn(x, y)` 和 `MoveToTable(x)`，"hand" 不是一个有效的 state predicate。SymbolicSceneGraph 使用 `handempty`/`holding(x)` 谓词，不存在 `on(x, hand)` 形式。

**影响**：生成的 symbolic action dict 与 BFS planner 的内部模型不兼容。

**修复**：`to_symbolic_actions()` 改为输出 `{"name": "pick"/"place", ...}` 格式，标注仅供 trace logging 使用。

### 2.3 `decompose_achieve()` — API 签名错误（致命）

```python
# 修复前
def decompose_achieve(self, predicate, objects, initial_predicates):
    plan = self.planner.solve(objects, predicate, initial_predicates)
    # ↑ 错误：SymbolicPlanner.solve(initial: SymbolicSceneGraph, goal: SymbolicGoal)
```

**问题**：直接调用 `SymbolicPlanner.solve()`，但传入了 `(list, list, list)` 而非 `(SymbolicSceneGraph, SymbolicGoal)`，一旦调用必定 TypeError。

**修复**：改用 `PlannerAdapter("bfs").solve(cubes=cubes, target_stack=target_stack)` 接口，从 predicate 中提取 target_stack，让 adapter 内部完成 SceneGraph/Goal 构建。

### 2.4 `ensure` — 执行层无对应 skill

修复前 `execution_sequence` 输出 `{"skill": "ensure", "predicate": pred}`，但 SkillLibrary 中没有 EnsureSkill。

**修复**：`ensure` 在 `execution_sequence` 中改为 NO-OP（直接 pass）。ensure 是纯验证 gate，verify 阶段已检查过 predicate。

### 2.5 `achieve` — 执行层无对应 skill

修复前 `execution_sequence` 输出 `{"skill": "achieve", "predicate": pred}`，同样无对应 skill。

**修复**：`achieve` 在 `execution_sequence` 中改为 NO-OP（直接 pass）。achieve 必须在此前由 `decompose_achieve()` 或 BFS planner 展开为 pick/place 序列。

### 2.6 `execution_sequence()` — 无问题的映射

```python
# pick → atomic_pick     ✅ 正确
# place → atomic_place   ✅ 正确
# move → atomic_pick + atomic_place  ✅ 正确（分解为两步）
# ensure → pass          ✅ 正确（纯验证，不产生动作）
# achieve → pass         ✅ 正确（必须先被 BFS 展开）
```

不输出 reach/grasp/lift/reach_place/release_place。

---

## 三、方法职责问题

三个方法做重叠的事但输出格式不同，且 `ground()` 和 `to_symbolic_actions()` 无任何调用方：

| 方法 | 输出格式 | 实际调用方 |
|------|---------|:---:|
| `ground()` | GroundedStep objects | **无** — 从未接入管线 |
| `to_symbolic_actions()` | `[{"name":"move","args":[...]}]` | **无** — 从未接入管线 |
| `execution_sequence()` | `[{"skill":"atomic_pick",...}]` | ✅ run_coarse_program_mujoco.py |

**建议**：保留 `execution_sequence()` 作为唯一对外接口。`ground()` 和 `to_symbolic_actions()` 标记为内部方法或删除。

---

## 四、修复清单

| # | Bug | 严重度 | 修复结果 |
|---|-----|:---:|------|
| 1 | `to_symbolic_actions` 中 pick 硬编码 source="table" | 中 | ✅ 改为 `src or "?"`，标注 trace only |
| 2 | "hand" 作为 symbolic location | 高 | ✅ 输出改为 pick/place 语义，不再用 "hand" |
| 3 | `decompose_achieve` API 签名错误 | 高 | ✅ 改用 `PlannerAdapter("bfs").solve()` |
| 4 | ensure 无对应执行 skill | 中 | ✅ 改为 NO-OP（pass） |
| 5 | achieve 无对应执行 skill | 中 | ✅ 改为 NO-OP（pass） |
| 6 | is_valid() 中 pick 未设置 holding 状态 | 中 | ✅ `coarse_dsl.py` 已修复 |

---

## 五、最终验收

| 检查项 | 状态 |
|--------|:---:|
| pick → atomic_pick? | ✅ |
| place → atomic_place? | ✅ |
| move → atomic_pick + atomic_place? | ✅ |
| "hand" 不再作为 symbolic location? | ✅ 已移除 |
| achieve 是否通过 PlannerAdapter 调用 BFS? | ✅ 已修复 |
| ensure 是否只验证不产生动作? | ✅ 改为 NO-OP |
| execution_sequence 只输出 atomic_pick/atomic_place? | ✅ |
| 是否输出 reach/grasp/lift? | ✅ 不输出 |
