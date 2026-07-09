# P3 Effect Event 与 Operator Candidate 归纳计划

> 日期：2026-07-09  
> 阶段：P3 — PredicateTrace → Effect Event → Operator Candidate  
> 输入：`generated/libero_audit/p3_start_package/`  
> 决策：P2v4.2 为 `G2_CONDITIONAL_PASS`，P3 在可信谓词掩码下启动。

---

## 一、阶段目标

P3 的目标不是训练策略，也不是重新调 MuJoCo 谓词，而是把 P2v4.2 生成的逐帧一阶谓词序列转换成可规划的符号算子候选：

```text
PredicateTrace
→ trusted / conditional / excluded predicate mask
→ stable predicate changes
→ effect events
→ operator candidates
→ plan skeleton replay audit
```

本阶段要回答：

1. 现有 8,170 帧 PredicateTrace 中能否稳定抽取出 pick/place/open/turn_on 等 effect event？
2. 这些 event 能否聚合成带参数、前置条件和效果的 typed operator candidates？
3. 哪些任务可以被 learned operator skeleton 覆盖，哪些失败来自谓词风险或缺失 operator？

---

## 二、P3 输入契约

本地已导入：

```text
generated/libero_audit/p3_start_package/
├── p2_predicate_frames_v4.jsonl
├── p2_goal_consistency_v4.json
├── p2_jitter_report_v4.json
├── p2_transition_predicate_probe_v4.json
├── p2v4_2_human_eval_metrics.json
├── p2v4_2_human_eval_error_cases.json
├── p2v4_2_rule_and_eval_change_log.md
├── p2v4_2_conditional_pass_addendum.md
├── p2v4_rule_change_log.md
└── metadata/
    ├── episode_manifest.jsonl
    ├── task_list.json
    ├── entity_inventory.json
    └── bddl_task_mapping.json
```

规模：

- 60 条 demo；
- 20 个任务；
- 8,170 帧 predicate trace；
- P2v4.2 Macro-F1 = 0.9473；
- Goal consistency = 60/60。

---

## 三、可信谓词掩码

P3 不能无差别使用所有谓词。准入规则如下：

| 谓词 | P3 使用方式 | 原因 |
|---|---|---|
| `in` | 无限制 | F1=1.00 |
| `open` | 无限制 | F1=1.00 |
| `activated` | 无限制 | F1=1.00 |
| `handempty` | 无限制，但记录 drawer-handle edge case | F1=0.91 |
| `holding` | risk mask：作为 effect 时要求 >=3 连续帧 | F1=0.94，仍有短暂抓取漏检 |
| `on` | risk mask：有 specific support 时排除 `on(table)` | F1=0.83，主要错误来自 cookies-table 双支撑 |
| `at_region` | 暂排除 | 11 条 invalid query，replacement 未验证 |

输出：

```text
generated/libero_audit/p3_symbolic_induction/p3_input_contract.md
generated/libero_audit/p3_symbolic_induction/p3_trusted_predicate_mask.json
```

---

## 四、任务拆分

### P3-T0：输入契约与产物目录

**目标**：冻结 P3 输入版本，明确风险控制。

**产物**：

- `p3_input_contract.md`
- `p3_trusted_predicate_mask.json`

**审核**：

- 记录所有输入文件 SHA-256；
- 明确 `G2_CONDITIONAL_PASS`；
- 明确 `at_region` 不进入主流程。

### P3-T1：PredicateTrace 规范化

**目标**：将原始 JSONL 中不同结构的谓词统一成原子列表。

原始形式包括：

```json
{"holding": "akita_black_bowl_1_joint0"}
{"on": [{"top": "obj", "bot": "table"}]}
{"in": [{"obj": "obj", "container": "drawer"}]}
{"open": ["drawer"]}
```

规范化后：

```json
["holding", "akita_black_bowl_1"]
["on", "akita_black_bowl_1", "plate_1"]
["in", "akita_black_bowl_1", "wooden_cabinet_1_top_level"]
["open", "wooden_cabinet_1_top_level"]
```

**规则**：

- 去除末尾 `_joint0`；
- 只对 `_joint0` 做规范化，不随意删除对象编号 `_1`；
- `holding` 需要 3 帧连续稳定后才进入 trusted effect；
- `on(table)` 在存在 specific support 时放入 conditional/masked；
- `at_region` 放入 excluded，不进入 event diff。

**产物**：

- `p3_normalized_traces.jsonl`
- `p3_trace_schema_report.json`

### P3-T2：Effect Event 抽取

**目标**：对每条 demo 的稳定谓词集合做差分，抽取事件。

事件包含：

```json
{
  "file": "",
  "demo": "",
  "task": "",
  "t_start": 0,
  "t_end": 0,
  "before": [],
  "after": [],
  "added": [],
  "deleted": [],
  "confidence": "trusted",
  "risk_notes": []
}
```

**事件过滤**：

- 单帧变化不直接成事件；
- 相邻变化在 3 帧内合并成同一 event window；
- 仅含 `closed/deactivated` 的初始化状态变化不作为主要 operator event；
- 由 conditional `on(table)` 触发的事件标记为 risky，不作为核心 operator effect。

**产物**：

- `p3_effect_events.jsonl`
- `p3_effect_event_summary.json`
- `p3_effect_event_report.md`

### P3-T3：Operator Candidate 初版归纳

**目标**：基于 effect signatures 聚类，归纳候选 operator。

初版 operator 类型：

- `pick(?obj)`：`+holding(?obj), -handempty`
- `place(?obj, ?target)`：`-holding(?obj), +handempty, +on/in(?obj, ?target)`
- `open(?drawer)`：`+open(?drawer), -closed(?drawer)`
- `turn_on(?device)`：`+activated(?device), -deactivated(?device)`
- `move_or_push(?obj, ?support)`：暂仅记录，不使用 `at_region`

**产物**：

- `p3_operator_candidates.json`
- `p3_operator_candidate_report.md`

**审核**：

- operator 名称可以人工命名；
- effects 必须来自 trace 证据；
- 每个 operator 记录 support_count、example_events、risk_notes。

### P3-T4：Plan Skeleton Replay 审计

**目标**：用抽取到的 event/operator skeleton 统计每个任务是否具有可解释的高层动作序列。

**产物**：

- `p3_plan_replay_eval.json`
- `p3_plan_replay_report.md`

**指标**：

- demo event coverage；
- operator type coverage；
- per-task covered demo 数；
- failed demos；
- ambiguous events。

---

## 五、本地 / 服务器分工

本地可执行：

- trace 解析；
- risk mask；
- event extraction；
- operator candidate induction；
- replay 报告；
- Obsidian / note 日志整理。

服务器保留：

- 重新跑 LIBERO / MuJoCo；
- 重新生成 PredicateTrace；
- replacement `at_region` 验证；
- 后续 VLA / diffusion policy 训练。

---

## 六、阶段验收标准

P3 第一轮成功标准：

1. `p3_normalized_traces.jsonl` 覆盖 60/60 demos；
2. effect events 数量合理，无单帧抖动爆炸；
3. 至少抽取出 `pick/place/open/turn_on` 中 3 类；
4. 每类 operator 有 example events 和 support_count；
5. `at_region` 未进入主流程；
6. `on(table)` 风险被记录或过滤；
7. 生成可读报告，能说明 P3 下一步是 operator clause revision 还是回补谓词。

---

## 七、今日执行顺序

1. 写入本计划；
2. 新增 P3 离线归纳脚本；
3. 运行脚本生成 T0-T3 产物；
4. 检查 summary 与报告；
5. 若事件抽取稳定，再推进 plan skeleton replay。

