# P4-B Baseline / Ablation 日志

> 日期：2026-07-09  
> 阶段：P4-B — Typed Domain Baseline and Ablation  
> 当前状态：已完成第一轮本地消融

## 1. 目标

P4 第一轮已经得到：

```text
P4_PASS
covered: 60/60
operators: 6
```

P4-B 的目标是回答：

> 60/60 覆盖到底来自哪些机制？

本轮不重新抽谓词、不跑 MuJoCo，只在已有 P3/P4 artifacts 上做符号层消融。

## 2. 新增能力

在 `src/vlm_nsm/research/p4_domain_export.py` 中新增：

- `split_place` domain mode；
- `fused_place_baseline`；
- `run_p4_ablation_suite()`；
- `p4_ablation_summary.json`；
- `p4_ablation_report.md`。

P4 CLI 不变，运行同一个命令会额外输出 ablation 报告。

## 3. 运行命令

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
```

## 4. 消融结果

| Variant | Covered | Failed | Invalid | 相对 full |
|---|---:|---:|---:|---:|
| full_typed_split | 60/60 | 0 | 0 | 0 |
| no_push_patch | 57/60 | 3 | 0 | -3 |
| no_goal_repair | 34/60 | 26 | 0 | -26 |
| local_only_effect | 31/60 | 29 | 0 | -29 |
| fused_place_full | 60/60 | 0 | 0 | 0 |

解释：

- `no_push_patch` 掉 3 条，正好对应 push task 的 3 条 demo；
- `no_goal_repair` 掉 26 条，说明 final-state goal repair 是 P3/P4 覆盖提升的关键；
- `local_only_effect` 只有 31/60，说明只做局部 effect event clustering 不足以解释长程任务；
- `fused_place_full` 覆盖仍为 60/60，但它保留 `place(on-or-in)`，表达上不如 `place_on/place_in` 适合 PDDL。

## 5. 当前结论

这轮消融支持一个比较重要的论文叙事：

> 本系统的提升不是来自单一规则补丁，而是来自三部分组合：可靠 PredicateTrace、goal replay 驱动的反例修订、以及服务器验证的 at_region/push operator 补全。

其中：

```text
local_only_effect -> full_typed_split: +29 demos
goal_repair gain: +26 demos
server at_region patch gain: +3 demos
typed split gain in coverage: 0 demos, but improves PDDL schema clarity
```

## 6. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_p4_domain_export.py tests/test_p3_symbolic_induction.py
```

结果：

```text
17 passed
```

## 7. 下一步

P4-C 建议做两件事：

1. 服务器端独立复算 P4 artifacts：
   - 接收代码；
   - 运行 P4 CLI；
   - 返回 `p4_pipeline_summary.json`、`p4_ablation_summary.json` 和 hash。
2. 开始 P5 设计：
   - 不再只做 demonstration replay；
   - 引入 held-out task / held-out composition；
   - 检查 typed domain 是否能进行组合规划。
