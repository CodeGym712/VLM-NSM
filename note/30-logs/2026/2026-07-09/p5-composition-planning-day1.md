# P5-B 组合符号规划审计日志

> 日期：2026-07-09  
> 阶段：P5-B — Cross-task Composition Planning Audit  
> 当前状态：P5_COMPOSITION_AUDIT_PASS

## 1. 目标

P5-A 已经验证：

```text
每条 demo 的初始 PredicateTrace state + 本 task goal
  -> typed domain BFS planner
  -> 60/60 solved
```

P5-B 继续向组合泛化推进：把每条 demo 的初始状态与所有 task goal 做交叉组合，形成一个 source-initial-state × target-goal 的组合规划矩阵。

目标不是宣称 held-out 泛化完成，而是先回答：

> 在所需可动物体确实存在于初始场景的前提下，当前 typed domain 是否能组合规划其他任务目标？

## 2. 新增代码

```text
src/vlm_nsm/experiments/run_p5_composition_planning.py
```

并在：

```text
src/vlm_nsm/research/p5_symbolic_planning.py
```

中新增：

- movable object presence filter；
- required movable goal symbols；
- plan grounding validation；
- cross-task composition audit；
- composition report renderer。

## 3. 安全过滤

为了避免假泛化，P5-B 不允许 planner 凭空使用场景中不存在的可动物体。

若目标所需 movable object 不在初始状态中，则该组合问题被标记为：

```text
skipped_absent_object
```

例如目标需要 `wine_bottle_1`，而某个 source initial state 里没有 wine bottle，则不尝试规划。

同时，规划完成后会检查：

```text
pick / place / push 是否使用了初始状态不存在的 movable object
```

若出现，则标记为：

```text
invalid_plan
```

## 4. 运行命令

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.run_p5_composition_planning `
  --p3-dir generated/libero_audit/p3_symbolic_induction `
  --p4-dir generated/libero_audit/p4_domain_export `
  --package-dir generated/libero_audit/p3_start_package `
  --output-dir generated/libero_audit/p5_composition_planning `
  --max-depth 4 `
  --max-expansions 20000
```

输出：

```text
P5-B OK: status=P5_COMPOSITION_AUDIT_PASS attempted=1110/1200 solved=1110 failed=0 invalid=0 skipped=90
```

## 5. 输出产物

```text
generated/libero_audit/p5_composition_planning/
├── p5_composition_planning_summary.json
├── p5_composition_planning_results.jsonl
└── p5_composition_planning_report.md
```

## 6. 结果

```json
{
  "problem_count": 1200,
  "attempted": 1110,
  "solved": 1110,
  "failed": 0,
  "invalid_plan": 0,
  "skipped_absent_object": 90,
  "attempted_solve_rate": 1.0
}
```

分类：

| Category | solved | failed | invalid | skipped |
|---|---:|---:|---:|---:|
| same_task | 60 | 0 | 0 | 0 |
| cross_task | 1050 | 0 | 0 | 90 |

## 7. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_p5_symbolic_planning.py tests/test_p4_domain_export.py tests/test_p3_symbolic_induction.py
```

结果：

```text
24 passed
```

## 8. 解释

P5-B 说明：

1. 当前 typed domain 在对象存在的兼容场景下，可以组合规划其他 task goal；
2. 组合矩阵中没有发现 invalid plan；
3. 90 个问题被跳过，原因是目标所需可动物体不在 source initial state 中；
4. 这比 P5-A 的 same-task planning 更强，但仍不是严格 held-out retraining。

## 9. 边界

不能把 P5-B 说成：

- leave-one-task-template-out 训练；
- 真实物理执行泛化；
- VLA policy 泛化；
- 完整组合泛化 benchmark。

更准确的表述：

> compatibility-filtered cross-task symbolic planning audit。

## 10. 下一步

P5-C 建议进入真正 held-out 设置：

1. 构造 leave-one-task-template-out split；
2. 每次从其余 task 归纳/导出 domain；
3. 在 held-out task 的 initial state + goal 上规划；
4. 与 local-only / no-goal-repair baseline 比较；
5. 服务器端复算 P5-A/P5-B。
