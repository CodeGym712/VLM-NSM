# P3 符号算子归纳完成报告

> 日期：2026-07-09  
> 阶段：P3 — PredicateTrace → Effect Events → Operator Candidates → Goal Replay  
> 当前状态：P3_PASS，60/60 goal replay coverage

## 1. 本地 P3 基线结果

本地 P3 产物位于：

```text
generated/libero_audit/p3_symbolic_induction/
```

本地 pipeline 已完成以下内容：

- 读取 P2v4 Certified PredicateTrace，共 8,170 帧、60 条 demo；
- 从谓词轨迹中抽取 effect events；
- 构建 object-centric episodes；
- 归纳初版 operator clauses；
- 执行 goal replay 与 counterexample-driven skeleton repair。

本地 v2 回放结果：

```text
covered: 57/60
failed: 0/60
blocked: 3/60
```

剩余 3 条 blocked 均来自：

```text
push_the_plate_to_the_front_of_the_stove
```

阻塞原因不是 operator induction 主流程失败，而是 P2 风险控制阶段临时排除了 `at_region`，导致 push 任务目标无法在本地 replay 中闭合。

## 2. 服务器端补充验证

服务器端补充验证产物位于：

```text
generated/libero_audit/p3_server_validation/
```

核心验证结果：

```text
demo_count: 60
covered: 60
failed: 0
blocked_excluded_predicate: 0
```

服务器端对 push 任务中的 `at_region` 进行了 replacement validation。关键结论是：正确对象应为 `plate_1`，不是此前盲审 query 中错误使用的 `cookies_1`。

验证明细：

| demo | T | at_region 帧数 | 末帧满足 |
|---|---:|---:|:---:|
| demo_27 | 220 | 53 | ✅ |
| demo_32 | 115 | 32 | ✅ |
| demo_39 | 146 | 45 | ✅ |

新增候选算子：

```text
push_to_region(?obj, ?region)
preconditions_pos: []
effects_add: at_region(?obj, ?region)
support_count: 3
confidence: conditional
risk_notes: at_region_validated_server
```

## 3. P3 最终判断

P3 达成阶段目标：

- PredicateTrace 可以支撑 effect event 抽取；
- effect events 可以归纳出可解释的 operator candidates；
- object-centric episodes 可用于绑定对象、目标和效果；
- goal replay 从 57/60 提升到 60/60；
- push 任务缺口已由服务器端 `at_region(plate_1, stove_front)` 验证闭合。

因此当前阶段状态为：

```text
P3_PASS
goal replay coverage: 60/60
```

## 4. 需要谨慎表述的边界

P3 通过不等于已经完成“无监督 PDDL 学习”。目前更准确的表述是：

> 基于 certified predicate traces，从官方 LIBERO 示范中抽取效果事件，并归纳可用于规划回放的候选符号算子。

仍需保留的风险说明：

- `push_to_region` support 只有 3 条 demo，可信度为 conditional；
- `place` 中部分支持来自 goal-implied final-state repair，不能宣称为完全干净的因果发现；
- 当前 P3 验证的是 symbolic goal replay coverage，不是机器人闭环执行成功率；
- 下一阶段必须把 operator clauses 导出为规范 domain，并用 planner / ablation 验证其规划一致性。

## 5. 下一阶段入口

建议进入 P4：类型化关系 Operator 归纳与 Domain 导出。

P4 最小任务：

1. 合并 P3 本地 operator clauses 与服务器端 `push_to_region`；
2. 输出 typed operator schema：
   - `pick(?obj)`
   - `place(?obj, ?target)`
   - `open(?drawer)`
   - `turn_on(?device)`
   - `push_to_region(?obj, ?region)`
3. 生成 PDDL-like / JSON domain；
4. 对 20 个 LIBERO task 执行 planner replay；
5. 做 baseline 对比：
   - local-only effect clustering；
   - no-goal-repair；
   - no-risk-mask；
   - oracle-domain upper bound。

P4 的审核目标不再是“是否有谓词”，而是：

- schema 是否类型一致；
- precondition/effect 是否有 supporting episodes；
- 是否能解释完整任务目标；
- 是否能在组合 held-out split 上优于局部规则 baseline。
