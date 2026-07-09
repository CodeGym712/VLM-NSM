# P5-C Leave-one-task-template-out 符号规划审计日志

> 日期：2026-07-10  
> 阶段：P5-C — Leave-one-task-template-out Planning Audit  
> 当前状态：P5_LOTO_AUDIT_COMPLETE

## 1. 目标

P5-B 的组合矩阵证明：

```text
compatibility-filtered cross-task planning: 1110/1110
invalid_plan: 0
```

但 P5-B 仍然使用 full domain。P5-C 进一步做严格证据消融：

```text
每次 held out 一个 task template；
从 operator evidence 中删除该 task 的所有支持；
若某 operator 在剩余 task 中 support=0，则从该 split domain 中移除；
再测试 held-out task 的 3 条 demo 是否能从初始状态规划到目标。
```

这一步用于判断：

> 当前 typed domain 的哪些 operator 具有跨任务证据，哪些只是单任务模板支撑。

## 2. 新增代码

```text
src/vlm_nsm/experiments/run_p5_loto_planning.py
```

并在：

```text
src/vlm_nsm/research/p5_symbolic_planning.py
```

新增：

- `operator_support_by_task()`；
- `support_excluding_task()`；
- `filter_domain_by_support()`；
- `expected_goal_operator_families()`；
- `run_p5_loto_planning_audit()`。

## 3. 运行命令

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.run_p5_loto_planning `
  --p3-dir generated/libero_audit/p3_symbolic_induction `
  --p4-dir generated/libero_audit/p4_domain_export `
  --package-dir generated/libero_audit/p3_start_package `
  --output-dir generated/libero_audit/p5_loto_planning `
  --max-depth 4 `
  --max-expansions 20000
```

输出：

```text
P5-C OK: status=P5_LOTO_AUDIT_COMPLETE splits=17/20 demos=51/60 invalid=0
```

## 4. 输出产物

```text
generated/libero_audit/p5_loto_planning/
├── p5_loto_planning_summary.json
├── p5_loto_planning_results.jsonl
├── p5_loto_split_summary.jsonl
└── p5_loto_planning_report.md
```

## 5. 结果

```json
{
  "split_count": 20,
  "problem_count": 60,
  "solved": 51,
  "failed": 9,
  "invalid_plan": 0,
  "solved_splits": 17,
  "failed_splits": 3,
  "heldout_task_success_rate": 0.85,
  "heldout_demo_success_rate": 0.85
}
```

通过的 split：17/20。

失败的 split：

| Held-out task | 失败原因 |
|---|---|
| `open_the_top_drawer_and_put_the_bowl_inside` | `place_in` 在 train tasks 中 support=0 |
| `push_the_plate_to_the_front_of_the_stove` | `push_to_region` 在 train tasks 中 support=0 |
| `turn_on_the_stove` | `turn_on` 在 train tasks 中 support=0 |

## 6. 解释

这不是坏结果，反而是 P5-C 的核心价值：

1. `pick` / `place_on` 有多任务证据，因此大部分 held-out 放置任务可规划；
2. `open` 有 middle drawer 与 top drawer 相关证据，因此 held-out middle drawer 可规划；
3. `place_in` 只由 `open_the_top_drawer_and_put_the_bowl_inside` 支撑，held-out 后被移除；
4. `push_to_region` 只由 push task 支撑，held-out 后被移除；
5. `turn_on` 只由 turn_on task 支撑，held-out 后被移除。

也就是说，P5-C 揭示了当前系统的泛化边界：

```text
可泛化：pick / place_on / open
证据不足：place_in / push_to_region / turn_on
```

## 7. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_p5_symbolic_planning.py tests/test_p4_domain_export.py tests/test_p3_symbolic_induction.py
```

结果：

```text
27 passed
```

## 8. 阶段意义

P5-C 给出了一个可以用于论文的更严格结论：

> 在 leave-one-task-template-out 的证据消融设置下，当前 typed domain 对 17/20 个 task template 保持可规划；失败均由 operator family 在训练任务中零支持导致。

这比直接汇报 full-domain 60/60 更可信。

## 9. 下一步

后续有两条路线：

1. 数据路线：
   - 增加 place_in / push_to_region / turn_on 的多模板证据；
   - 或从 LIBERO 其他 task 中引入相同 operator 的支持。

2. 方法路线：
   - 引入 operator family prior；
   - 允许少样本 operator schema transfer；
   - 将 zero-support operator 标为 prior-enabled，而不是 learned-from-train。

论文表述必须区分：

```text
learned from train evidence
vs
prior-enabled operator family
```
