# VLM-NSM 上层神经符号改造审计报告

> 日期：2026-06-24 | 审计范围：全代码库 | 方法：3 并行 Agent 扫描  
> 证据优先级：代码实现 > 实验日志 > 文档表述

---

## 1. 当前上层真实链路

```
自然语言 "把红色方块放到绿色方块上面"
  │
  ▼
OnlineQwenGoalParser.parse(instruction, objects, task_type)
  ├─ _build_prompt() → 拼接 prompt 字符串
  ├─ _call_api() → OpenAI client → DashScope Qwen API
  └─ → GoalParseResult {success, goal_json}
  │
  ▼
AlignmentLayer.align_and_validate(goal_json, valid_objects, valid_predicates, type_table, predicate_signature)
  ├─ 6 层检查: schema → predicate name → object → type → stacking完整性 → 可选 hanoi 补充
  └─ → AlignmentResult {success, goal_predicates: list[Predicate]}
  │
  ▼
target_stack_from_predicates() → list[str] (bottom-to-top)
  │
  ▼
PlannerAdapter("bfs").solve(cubes, target_stack)
  ├─ BFSStackingAdapter → 构建 SceneGraph + StackOrderGoal
  ├─ SymbolicPlanner.solve(initial, goal) → BFS 搜索
  └─ → PlanResult {success, actions: list[Action]}
  │
  ▼
run_task(env, task, pid, skill_library)
  ├─ for action in plan.actions:
  │    for skill in task.skills:
  │      SkillLibrary.get(skill) → skill.execute_with_pid(env, pid, params)
  └─ → dict {goal_success, failure_type, ...}
```

**关键边界**：
- `AlignmentLayer` 在 parser pipeline 中使用（处理 raw JSON）
- `GoalValidator` 在 execution monitoring 中使用（处理 SymbolicGoal）——两个 validator 是**独立的、不互通**
- BFS 仅产出 `MoveToTable` / `MoveOn` 两种算子（硬编码在 `_generate_actions` 中）
- TaskRunner 是线性序列执行，无 BT 控制流（条件/回退/并行）

---

## 2. 已有可复用模块

| 模块 | 文件 | 可复用原因 |
|------|------|------|
| **SymbolicSceneGraph** | `symbolic/predicates.py` | 完整的状态表示：objects + predicates + clone + to_dict |
| **SymbolicGoal / StackOrderGoal** | `symbolic/goals.py` | 目标表示，支持 AND 语义 |
| **SymbolicAction / MoveOn / MoveToTable** | `symbolic/operators.py` | STRIPS 风格的算子骨架：preconditions + add/del effects + apply() |
| **AlignmentLayer** | `core/alignment.py` | 6 层 pipeline 校验：参数化，可扩展 |
| **GoalValidator** | `symbolic/goal_validator.py` | 跨谓词校验：多支撑/环检测 |
| **SkillResult** | `embodied/skills/base_skill.py` | **已包含 OptionResult 的全部字段**（pre_predicates, post_predicates, expected_effect, effect_satisfied）——仅需填充 |
| **OperatorMonitor** | `neuro_symbolic/operator_monitor.py` | 故障分类 + 恢复提示，已定义 12 种 OperatorFailureType |
| **PID Controller** | `embodied/controllers/operational_space_pid.py` | 稳定的低层控制器，无需修改 |
| **SkillLibrary** | `embodied/skills/base_skill.py` | 可扩展的技能注册表 |
| **PlannerAdapter** | `core/planner_adapter.py` | 统一规划接口，支持 bfs/hanoi/internal_stacking/metric_ff |

---

## 3. 缺失模块

### 3.1 PredicateSceneGraph

**状态：❌ 不存在**

`SymbolicSceneGraph` 是存在的，但它缺少以下内容：
- `robot` 状态（gripper_empty, holding, eef_pose）
- `task_context`（available_skills, known_supports, forbidden_relations）
- `relations` 枚举：仅 `on`/`ontable`/`clear`；缺少 `above`, `blocked_by`, `near`
- 从当前环境/oracle pose 构建的场景图
- 序列化为 LLM prompt-friendly 文本

**建议**：基于 `SymbolicSceneGraph` 扩展，而非替换。

### 3.2 TaskComposer

**状态：❌ 不存在**

### 3.3 Verifier Bank

**状态：❌ 不存在**

### 3.4 Option DSL

**状态：❌ 不存在**

### 3.5 Behavior Tree Executor

**状态：❌ 不存在**

### 3.6 Trace-to-Repair Loop

**状态：⚠️ 部分存在**

- `OperatorMonitor` 提供故障分类和恢复提示（查找表）
- `task_runner` 有 replan 钩子但仅检测 stack collapse
- `SkillResult` 有 pre/post predicates 字段但未填充

### 3.7 Unified Option Contract

**状态：⚠️ 半成品**

- `SkillResult` 已有字段定义（pre_predicates, post_predicates, expected_effect, effect_satisfied）
- `SkillParams` 已有参数定义
- 但**没有 `OptionContract`** 定义 initiation/termination 条件

### 3.8 OptionResult Logger

**状态：⚠️ 部分存在**

- `summary.json` 已有 per-episode 聚合
- `SkillResult.diagnostics` 可扩展
- 但没有 option-level trace

---

## 4. 不建议动的稳定模块

| 模块 | 原因 |
|------|------|
| `operational_space_pid.py` | 已验证 91% (2-block), 93% (3-block) |
| `grasp_state_machine.py` | 闭环抓取，已验证 |
| `place_state_machine.py` | 闭环放置，已验证 |
| `symbolic/symbolic_planner.py` | BFS 算法正确，2-5 block 验证通过 |
| `symbolic/predicates.py` — SymbolicSceneGraph 核心 | 状态表示稳定，扩展而非替换 |
| `online_qwen_parser.py` | API 调用稳定，仅需加验证层 |
| `skill_plan_from_symbolic.py` | 格式转换已修复，不增加复杂度 |
| `task_spec.py` | 好的抽象，扩展字段即可 |

---

## 5. 最小改造路径

基于审计发现，**6 个 Commit 级别的最小改造步骤**：

### Commit 1: PredicateSceneGraph 扩展

- 在 `symbolic/predicates.py` 中扩展 `SymbolicSceneGraph` 添加 `robot` 和 `task_context`
- 在 `core/predicate_scene_graph.py` 中新建 builder（从 oracle pose + env 构建）
- 添加 `to_prompt_text()` 序列化方法
- 单元测试：2-block, 3-block, Hanoi 初始状态

### Commit 2: OptionContract 定义 + SkillResult 填充

- 在 `core/option_contract.py` 中定义 `OptionContract` 数据类
- 为 6 个核心技能各创建 contract
- 在 `task_runner.py` 中每次技能执行前后调用 predicate 提取并填充 `SkillResult`

### Commit 3: Verifier Bank

- 在 `neuro_symbolic/verifier_bank.py` 中实现 5 个 Verifier
- GoalVerifier 复用 `AlignmentLayer` + `GoalValidator`
- 用构造的 bad goals/plans 做单元测试

### Commit 4: Option DSL + TaskComposer experimental path

- 在 `language/option_dsl.py` 中定义受限 DSL（不是自由 Python）
- 在 `language/task_composer.py` 中实现 TaskComposer（输入 NL + SceneGraph → TaskProgram JSON）
- 不修改 Qwen parser 主路径——新增 `--composer experimental` flag

### Commit 5: Behavior Tree Executor

- 在 `embodied/behavior_tree.py` 中实现 Sequence/Fallback/Condition/Action/Retry/Verify 节点
- 2-block stacking 跑通 BT 执行路径
- 记录 BT execution trace

### Commit 6: Trace-to-Repair + 实验

- 在 `neuro_symbolic/failure_summarizer.py` 中实现 FailureSummarizer
- 在 `neuro_symbolic/repair_proposer.py` 中实现 RepairProposer（hand-coded baseline + LLM experimental）
- Offline NS benchmark (50 tasks × SceneGraphs)
- 生成 summary.json

---

## 6. 关键 Gap 优先级

| 优先级 | Gap | 影响 |
|:---:|------|------|
| **P0** | PredicateSceneGraph 缺失 | 所有上层模块（Verifier/Composer/BT）都需要场景图输入 |
| **P0** | SkillResult 字段未填充 | OptionResult 已有字段但从未被使用 |
| **P1** | failure_type 未统一 | 10+ 文件使用不一致的字符串，OperatorMonitor 依赖 `in` 匹配 |
| **P1** | Verifier Bank 不存在 | LLM 输出无法被验证，无法拒绝 bad plans |
| **P2** | BT Executor 不存在 | 线性序列无法表达条件/回退/重试 |

---

## 7. 当前代码与文档的不一致

| 文档声称 | 代码实际 |
|------|------|
| "BFS 是默认规划器" | ✅ 正确（PlannerAdapter 默认 `"bfs"`） |
| "PhaseDefinition.pid_gain_overrides 定义增益" | ❌ 已被删除（`gain_overrides` 字段和值全删，`get_pid_overrides` 返回 `{}`） |
| "Skip_approach=False" (grasp_skill) | ⚠️ 已被 linter 改为 `skip_approach=False`（6/20 修改），磁盘代码与审计时的代码相同 |
| "SkillResult 有 pre_predicates 等字段" | ✅ 字段存在但**从未被填充**——是占位符 |
