# P4 类型化 Domain 导出与符号回放日志

> 日期：2026-07-09  
> 阶段：P4 — Typed Operator Schema + Domain Export + Planner Replay  
> 当前状态：P4_PASS，60/60 symbolic goal replay covered

## 1. 目标

P4 的目标是把 P3 的候选 operator 变成更适合规划与论文描述的 typed symbolic domain。

P3 结束时已经有：

```text
pick
place
open
turn_on
push_to_region
```

但其中 `place` 在 P3 中是 fused operator，效果写作 `on-or-in`，不适合直接导出为 PDDL。P4 因此做了显式拆分：

```text
place_on(?obj, ?target)
place_in(?obj, ?container)
```

并合并服务器端验证的：

```text
push_to_region(?obj, ?region)
```

## 2. 新增代码

```text
src/vlm_nsm/research/p4_domain_export.py
src/vlm_nsm/experiments/run_p4_domain_export.py
tests/test_p4_domain_export.py
```

核心功能：

1. 读取 P3 `p3_operator_clauses_v2.json`；
2. 读取 P3 `p3_plan_skeletons_v2.jsonl`；
3. 读取服务器端 `p3_push_operator_candidate.json` 与 `at_region_replacement_eval.json`；
4. 构建 typed domain；
5. 导出 PDDL-like domain；
6. 对 60 条 demo 执行 symbolic goal replay；
7. 输出 P4 summary 与 report。

## 3. 输出产物

```text
generated/libero_audit/p4_domain_export/
├── p4_domain.json
├── p4_domain.pddl
├── p4_plan_skeletons.jsonl
├── p4_domain_replay_eval.json
├── p4_domain_replay_report.md
└── p4_pipeline_summary.json
```

## 4. 当前 Domain

P4 v1 domain 包含 6 个 operator：

| Operator | Support | Confidence |
|---|---:|---|
| pick | 29 | conditional |
| place_on | 50 | conditional |
| place_in | 3 | conditional |
| open | 6 | trusted |
| turn_on | 3 | trusted |
| push_to_region | 3 | conditional |

谓词集合：

```text
handempty()
holding(obj)
on(obj, target)
in(obj, container)
open(drawer)
closed(drawer)
activated(device)
deactivated(device)
at_region(obj, region)
```

类型集合：

```text
object
support
container
openable
device
region
```

## 5. 真实运行命令

```powershell
.\.venv\Scripts\python.exe -m vlm_nsm.experiments.run_p4_domain_export `
  --p3-dir generated/libero_audit/p3_symbolic_induction `
  --package-dir generated/libero_audit/p3_start_package `
  --server-validation-dir generated/libero_audit/p3_server_validation `
  --output-dir generated/libero_audit/p4_domain_export
```

输出：

```text
P4 OK: status=P4_PASS operators=6 covered=60/60 failed=0 invalid=0
Artifacts saved to generated\libero_audit\p4_domain_export
```

## 6. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_p4_domain_export.py tests/test_p3_symbolic_induction.py
```

结果：

```text
17 passed
```

## 7. 阶段解释

当前 P4 通过的含义是：

> P3/P4 归纳出的 typed symbolic domain 可以解释当前 60 条 LIBERO demo 的目标达成。

不能夸大为：

- 不是完整无监督 PDDL 学习；
- 不是物理闭环执行成功；
- 不是 VLA policy 训练完成；
- 不是证明全部 precondition 都已被强因果识别。

更准确的研究表述是：

> Certified PredicateTrace 支撑的 typed operator schema 导出与 planning-consistency replay audit。

## 8. 下一步

P4 后续建议继续做 baseline / ablation：

1. `no_push_patch`：不使用服务器 at_region patch，应回退到 57/60；
2. `no_goal_repair`：不使用 P3 final-state repair，观察 place 覆盖下降；
3. `fused_place`：保留 `place(on-or-in)`，与拆分 `place_on/place_in` 对比；
4. `local_only_effect`：只用局部 effect clustering，不做 goal replay 修订；
5. `oracle_domain`：人工 oracle domain 上界。

P4 的论文价值应落在：

```text
可审计的符号算子归纳 + typed domain 导出 + 规划一致性反例修订
```

而不是单纯宣称“从示范学 PDDL”。
