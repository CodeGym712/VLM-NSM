# P3 符号事件归纳 Day 1 执行日志

> 日期：2026-07-09  
> 阶段：P3 — Effect Event 与 Operator Candidate 初版  
> 输入：`generated/libero_audit/p3_start_package/`  
> 输出：`generated/libero_audit/p3_symbolic_induction/`

---

## 一、今日决策

P2v4.2 作为 `G2_CONDITIONAL_PASS` 进入 P3，但 P3 必须使用可信谓词掩码：

| 谓词 | P3 使用方式 |
|---|---|
| `in`, `open`, `activated`, `handempty` | 无限制 |
| `holding` | 要求 >=3 连续帧才作为可信 effect |
| `on` | 有 specific support 时屏蔽 `on(table)` |
| `at_region` | 暂排除 |

---

## 二、完成内容

新增代码：

```text
src/vlm_nsm/research/p3_symbolic_induction.py
src/vlm_nsm/experiments/run_p3_symbolic_induction.py
tests/test_p3_symbolic_induction.py
```

实现能力：

1. 读取 P3 start package；
2. 生成 P3 input contract；
3. 应用 trusted predicate mask；
4. 规范化 8,170 帧 PredicateTrace；
5. 抽取 effect events；
6. 初步分类 operator candidates；
7. 生成 plan skeleton replay 审计；
8. 对 `relocate_effect + release` 做轻量 fusion，补充 `place(obj,target)` skeleton。

---

## 三、验证

单元测试：

```text
tests/test_p3_symbolic_induction.py
5 passed
```

覆盖内容：

- `_joint0` 规范化但保留对象编号 `_1`；
- `holding` 必须满足 >=3 连续帧；
- `on(table)` 在存在 specific support 时被 mask；
- `at_region` 被排除；
- `relocate_effect + release` 可融合为 `place` skeleton。

---

## 四、P3 产物

```text
generated/libero_audit/p3_symbolic_induction/
├── p3_input_contract.md
├── p3_trusted_predicate_mask.json
├── p3_normalized_traces.jsonl
├── p3_trace_schema_report.json
├── p3_effect_events.jsonl
├── p3_effect_event_summary.json
├── p3_effect_event_report.md
├── p3_classified_events.jsonl
├── p3_operator_candidates.json
├── p3_operator_candidate_report.md
├── p3_operator_induction_summary.json
├── p3_plan_replay_eval.json
├── p3_plan_replay_report.md
├── p3_plan_skeletons.jsonl
├── p3_plan_skeleton_summary.json
└── p3_pipeline_summary.json
```

---

## 五、初步结果

### 1. Trace 规范化

```text
Frames: 8,170
Demos: 60
Tasks: 20
```

### 2. Effect Events

```text
Effect events: 261
Event-level covered demos: 57/60
```

三条未覆盖 demo 全部来自 `push_the_plate_to_the_front_of_the_stove`，原因是该任务主要依赖 `at_region`，而 `at_region` 当前按 P3 准入规则被排除。

### 3. 初版 operator candidates

```text
pick: 31
place: 7
release: 20
relocate_effect: 82
open: 6
turn_on: 3
unclassified: 112
```

解释：

- `open` 和 `turn_on` 非常干净；
- `pick` 有足够支持，但仍带 holding/on 风险标记；
- `place` 被低估，因为 `on/in` 关系常常早于 gripper release 出现；
- `relocate_effect` 数量很高，说明目标关系变化可见，但和 release 的时间绑定还需要更强规则；
- `unclassified` 主要来自支撑关系的中间变化和 risk-mask 后的残余事件。

### 4. Fused skeleton

```text
Fused skeleton covered demos: 42/60
Fused skeleton operator counts:
- open: 6
- pick: 31
- place: 27
- turn_on: 3
Unresolved release: 0
```

解释：

- 通过 `relocate_effect + release` fusion，`place` 从 7 提升到 27；
- 仍有部分 pick/place 任务缺完整 skeleton，说明下一步需要做更稳的 event window 与目标绑定；
- push plate 任务仍无法覆盖，因为缺少可用的 `at_region`。

---

## 六、主要发现

1. P3 可以在本地推进，不需要 MuJoCo/GPU。
2. P2v4.2 PredicateTrace 足以抽取一批可解释事件，但还不足以直接形成高质量完整 operator domain。
3. `open`、`turn_on` 是最干净的 operator。
4. `pick` 的核心证据可用，但受 `holding >=3帧` 风险掩码影响，召回偏保守。
5. `place` 的主要难点不是没有目标关系，而是目标关系与 release 的时间不同步。
6. `push` 类任务暂时被 `at_region` 排除阻断。

---

## 七、下一步任务

### P3-T5：Event-to-Operator 绑定增强

目标：把 `relocate_effect`、`release`、`holding` 变化在更大窗口内绑定成稳定 operator。

计划：

1. 对每个 demo 建立 object-centric event timeline；
2. 对每个 held object 追踪：
   - pick time；
   - last trusted target relation；
   - release time；
3. 生成 object-centric `pick/place` episodes；
4. 用 goal predicate 验证最终 target 是否合理。

### P3-T6：Operator Clause 初版

目标：从 fused skeleton 中统计 precondition/effect。

需要输出：

```text
p3_operator_clauses_v1.json
p3_operator_clause_report.md
```

### P3-T7：at_region replacement 并行需求

服务器端应继续补充 `at_region` replacement queries，否则 `push_the_plate_to_the_front_of_the_stove` 无法进入 P3 主流程。

---

## 八、当前状态

```text
P3 已启动；
P3-T0/T1/T2/T3/T4 初版完成；
可以继续本地推进 P3-T5；
服务器端仅需补 at_region replacement 与后续重型验证。
```

---

## 九、追加进展：P3-T5/T6

### T5：Object-centric episodes

新增 object-centric 绑定逻辑：

```text
pick(obj)
recent relocate_effect(obj, target)
release(obj)
→ pick_place(obj, target)
```

产物：

```text
p3_object_centric_episodes.jsonl
p3_object_centric_summary.json
```

结果：

```text
episode_count: 44
pick_place: 21
pick_without_place: 8
place_without_pick: 6
open: 6
turn_on: 3
trusted: 12
risky: 32
```

解释：

- `pick_place` 已能覆盖 21 个完整对象搬运片段；
- `pick_without_place` 与 `place_without_pick` 说明事件绑定仍不完整，后续需要结合 goal predicate 和更宽的 object timeline；
- `risky` 数量较高，主要继承自 holding/on 的 P2v4.2 条件通过风险。

### T6：Operator clauses v1

新增基于 episodes 的 clause 模板归纳。

产物：

```text
p3_operator_clauses_v1.json
p3_operator_clause_report.md
```

初版 clauses：

| Operator | Support | Confidence | 说明 |
|---|---:|---|---|
| `pick(?obj)` | 29 | conditional | 来自 holding transition，缺 clear/support precondition |
| `place(?obj, ?target)` | 27 | conditional | 目标关系来自 fused skeleton，仍受 on(table) 风险影响 |
| `open(?drawer)` | 6 | trusted | 干净 |
| `turn_on(?device)` | 3 | trusted | 干净 |

### 新验证

```text
tests/test_p3_symbolic_induction.py
6 passed
```

新增测试：

- `relocate_effect + release` 可融合为 `place`；
- skeleton 中的 `pick/place` 可形成 `pick_place` episode；
- operator clauses 能从 episodes 归纳出 `pick/place`。

### 当前 P3 判断

P3 已从“事件抽取”推进到“operator clause v1”。但 v1 仍不是最终 PDDL domain：

1. `pick/place` 仍是 conditional；
2. `place` target 绑定需要 goal-aware 校验；
3. `push` 类任务仍被 `at_region` 排除；
4. 下一步应做 counterexample-driven replay，检查 operator clause 是否能解释每个 task goal。
