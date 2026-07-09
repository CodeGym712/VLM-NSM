# P5-A 初始状态符号规划审计日志

> 日期：2026-07-09  
> 阶段：P5-A — Initial PredicateTrace State + Typed Domain + BFS Planner  
> 当前状态：P5_INITIAL_PLANNING_PASS

## 1. 目标

P4 已经完成 typed domain replay：

```text
P4_PASS
covered: 60/60
```

但 P4 replay 仍然使用 P3/P4 skeleton 作为计划结构，不能证明 domain 能从初始状态独立规划。

P5-A 的目标是构建真正的符号规划问题：

```text
initial PredicateTrace state + BDDL goal + P4 typed domain
  -> BFS symbolic planner
  -> plan found / failed
```

## 2. 新增代码

```text
src/vlm_nsm/research/p5_symbolic_planning.py
src/vlm_nsm/experiments/run_p5_symbolic_planning.py
tests/test_p5_symbolic_planning.py
```

核心功能：

1. 从 `p3_normalized_traces.jsonl` 提取每条 demo 的 `t=0` trusted predicates；
2. 从 `p4_domain.json` 读取 typed operators；
3. 从 BDDL mapping 读取目标；
4. 从初始状态、目标和 skeleton evidence 推断对象类型 universe；
5. ground typed actions；
6. 用 BFS 在最大深度 4 内搜索计划；
7. 输出规划结果和报告。

## 3. P5 暴露并修复的问题

### 问题 1：`place_in` 前置条件过弱

初版 P5 发现：

```text
open_the_top_drawer_and_put_the_bowl_inside
```

可以被规划成：

```text
pick -> place_in
```

这说明 P4 的 `place_in` 缺少 `open(container)` 前置条件。已修复为：

```text
place_in(?obj, ?container)
preconditions:
  holding(?obj)
  open(?container)
effects:
  handempty()
  in(?obj, ?container)
  not holding(?obj)
```

修复后 top drawer 任务规划为：

```text
open(wooden_cabinet_1_top_level)
-> pick(akita_black_bowl_1)
-> place_in(akita_black_bowl_1, wooden_cabinet_1_top_level)
```

### 问题 2：P3/P4 replay 的宽松 alias 不适合 planner audit

严格匹配后，避免了：

```text
bottom_level 错误匹配 top_level
```

但设备类目标需要受控别名：

```text
activated(flat_stove_1_button)
≈ activated(flat_stove_1)
```

因此 P5 使用更严格的 goal matching：

- drawer level：exact / region-level bridge；
- device：允许 `flat_stove_1` 与 `flat_stove_1_button/main`；
- object body：只允许 exact / `_main` bridge。

## 4. 运行命令

先重新导出 P4 domain：

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.run_p4_domain_export `
  --p3-dir generated/libero_audit/p3_symbolic_induction `
  --package-dir generated/libero_audit/p3_start_package `
  --server-validation-dir generated/libero_audit/p3_server_validation `
  --output-dir generated/libero_audit/p4_domain_export
```

再运行 P5：

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.run_p5_symbolic_planning `
  --p3-dir generated/libero_audit/p3_symbolic_induction `
  --p4-dir generated/libero_audit/p4_domain_export `
  --package-dir generated/libero_audit/p3_start_package `
  --output-dir generated/libero_audit/p5_symbolic_planning `
  --max-depth 4 `
  --max-expansions 20000
```

输出：

```text
P5 OK: status=P5_INITIAL_PLANNING_PASS solved=60/60 failed=0 avg_len=1.90
```

## 5. 输出产物

```text
generated/libero_audit/p5_symbolic_planning/
├── p5_initial_planning_summary.json
├── p5_initial_planning_results.jsonl
└── p5_initial_planning_report.md
```

## 6. 结果

```json
{
  "status": "P5_INITIAL_PLANNING_PASS",
  "demo_count": 60,
  "solved": 60,
  "failed": 0,
  "max_depth": 4,
  "plan_length_counts": {
    "1": 9,
    "2": 48,
    "3": 3
  },
  "avg_plan_length": 1.90
}
```

解释：

- 1-step：open / turn_on / push_to_region；
- 2-step：pick + place_on；
- 3-step：open + pick + place_in。

## 7. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_p5_symbolic_planning.py tests/test_p4_domain_export.py tests/test_p3_symbolic_induction.py
```

结果：

```text
22 passed
```

## 8. 阶段意义

P5-A 比 P4 replay 更进一步：

```text
P4: skeleton replay
P5-A: initial-state planner search
```

当前可以汇报：

> typed domain 不仅能解释演示 skeleton，还能从每条 demo 的初始 PredicateTrace state 出发，通过符号搜索找到达成 BDDL goal 的计划。

边界：

- 仍不是 held-out composition 泛化；
- 仍不是物理执行；
- 仍不是 VLA policy；
- 当前 precondition 仍偏保守，需要 P5-B/P6 继续做 split 和真实执行验证。

## 9. 下一步

P5-B 应进入 held-out / composition planning：

1. leave-one-task-template-out；
2. new goal composition；
3. invalid-plan counterexample generation；
4. 与 local-only / no-goal-repair baseline 对比。
