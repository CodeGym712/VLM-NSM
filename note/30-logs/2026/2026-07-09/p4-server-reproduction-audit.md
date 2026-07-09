# P4 服务器端独立复算审计

> 日期：2026-07-09  
> 阶段：P4 / P4-B  
> 状态：SERVER_REPRO_PASS

## 1. 服务器端输入

服务器端在项目目录下执行 P4 domain export 后，产物位于：

```text
generated/libero_audit/p4_domain_export/
```

本次审计基于服务器端打印的两个核心文件：

```text
p4_pipeline_summary.json
p4_ablation_summary.json
```

## 2. Pipeline 复算结果

服务器端结果：

```json
{
  "ok": true,
  "stage": "P4",
  "domain_operator_count": 6,
  "demo_count": 60,
  "covered": 60,
  "failed": 0,
  "invalid_plan": 0,
  "status": "P4_PASS"
}
```

Operator 使用计数：

```json
{
  "open": 6,
  "place_in": 3,
  "pick": 31,
  "place_on": 50,
  "push_to_region": 3,
  "turn_on": 3
}
```

结论：

```text
服务器端 P4 pipeline 与本地结果一致。
P4 typed domain replay = 60/60 covered。
```

## 3. Ablation 复算结果

服务器端 P4-B 消融：

| Variant | Covered | Failed | Invalid | 结论 |
|---|---:|---:|---:|---|
| full_typed_split | 60/60 | 0 | 0 | 完整系统 |
| no_push_patch | 57/60 | 3 | 0 | 去掉 at_region/push 后损失 3 条 |
| no_goal_repair | 34/60 | 26 | 0 | 去掉 goal repair 后损失 26 条 |
| local_only_effect | 31/60 | 29 | 0 | 仅局部 effect 远不足够 |
| fused_place_full | 60/60 | 0 | 0 | 覆盖相同，但表达不如 typed split clean |

服务器端 interpretation：

```json
{
  "server_at_region_patch_gain": 3,
  "goal_repair_gain": 26,
  "local_to_full_gain": 29,
  "typed_split_coverage_delta": 0
}
```

## 4. 审计判断

本次服务器端独立复算通过，当前可以记录为：

```text
P4_SERVER_REPRO_PASS
```

可以用于汇报的实验证据：

1. 完整 typed domain 可以解释当前 60 条 LIBERO demo；
2. 去掉服务器 `at_region/push` patch 后回退到 57/60；
3. 去掉 goal repair 后回退到 34/60；
4. 仅局部 effect event 只有 31/60；
5. `place_on/place_in` 拆分不提升 replay coverage，但提供更干净的 PDDL schema。

## 5. 边界

该复算仍是 symbolic replay / planning-consistency replay，不是：

- 物理闭环执行；
- VLA policy success；
- held-out task 泛化；
- 完全无监督 PDDL 学习。

下一阶段必须进入 P5：

```text
held-out / composition planning
```

否则当前 60/60 只能证明示范回放一致性，不能证明组合泛化。

## 6. 下一步

建议服务器端补充：

```bash
find generated/libero_audit/p4_domain_export -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > generated/libero_audit/p4_domain_export/p4_manifest.sha256
```

然后将以下文件传回本地归档：

```text
p4_pipeline_summary.json
p4_ablation_summary.json
p4_ablation_report.md
p4_domain_replay_report.md
p4_manifest.sha256
```
