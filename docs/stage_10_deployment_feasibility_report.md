# Stage 10: 仿真部署策略定稿与端到端可行性演示

> **日期**: 2026-05-17
> **目的**: 确认 canonical deployment policy，运行端到端语言→仿真演示，更新协议

## 1. Stage 9 回顾

Stage 9 验证了规范共享 runner 环境并确认了关键发现：
- DAgger canonical 3×50ep avg **83.3%** — Stage 7B-5 的 93.3% 是小样本偏差
- Scripted 基线 **88.0%** — 仿真固有失败率 ~12%
- preplace 是 DAgger reach_pick+reach_place 的主要失败模式（52.5%）

## 2. 为什么 reach_pick+reach_place 不够稳健

DAgger reach_pick+reach_place 3×50ep 分析：

| 指标 | 值 |
|:----|:---:|
| 平均成功率 | 83.3%（82/82/86%） |
| 平均 preplace 失败 | 43.7/150ep |
| preplace 占比 | 29.1% 的 episode |
| BC 平均步数 | 496 |

**根因**: reach_place 阶段的 BC 策略末端精度不足。BC 输出的动作噪声导致末端执行器无法精确定位到目标位置上方，使 preplace 验证反复触发直至超时。即使在成功 episode 中，80% 的 `failure_type` 也被标记为 `preplace`。

## 3. reach_pick-only 验证

### 100ep（S10-A）

| 指标 | 值 |
|:----|:---:|
| 成功率 | **92.0%** (92/100) |
| BC 步数 | 113（仅 reach_pick） |
| preplace 失败 | **0** |
| horizon | 6 |
| verify_failed | 2 |

**关键**: preplace=0 — 去除 reach_place 阶段的 BC 控制完全消除了 preplace 失败模式。

### 3×50ep（S10-B）

| 指标 | seed=0 | seed=1 | seed=2 | 平均 |
|:---|:-----:|:-----:|:-----:|:----:|
| 成功率 | **96.0%** | **90.0%** | **94.0%** | **93.3%** |
| 成功数 | 48/50 | 45/50 | 47/50 | 46.7/50 |
| horizon | 0 | 1 | 1 | 0.7 |
| preplace | 1 | 1 | 0 | 0.7 |
| verify_failed | 1 | 3 | 1 | 1.7 |

**93.3% avg — consistent with 92.0% on 100ep, confirming reach_pick-only stability across seeds.**

## 4. 部署策略对比

| Policy | N | Success | Preplace | 推荐 |
|:-------|:-:|:-------:|:--------:|:----:|
| Scripted-only | 100 | 88.0% | 0 | 基线参考 |
| DAgger reach+place | 150 | 83.3% | 43.7 | ⚠️ 研究用 |
| **DAgger reach_pick-only (100ep)** | **100** | **92.0%** | **0** | **✅ 默认部署** |
| **DAgger reach_pick-only (3×50ep)** | **150** | **93.3% avg** | **0.7** | **✅ 确认稳定** |

## 5. Canonical Deployment Policy

**Policy**: `DAgger reach_pick-only`
**bc_phases**: `reach_pick`
**成功率**: 92.0%（100ep 验证） / 93.3% avg（3×50ep 多 seed 验证）
**理由**:
1. preplace 完全消除（0/100ep, 0.7/150ep avg）
2. 相比 reach+place 增益 +9~10pp
3. 相比 scripted 基线增益 +4~5pp（BC 在 reach_pick 阶段确实优于 scripted）
4. 脚本控制 grasp/lift/release/retreat 保持稳定
5. 多 seed 一致性高（96/90/94%，方差 ±3%）

## 6. 端到端语言→仿真 Demo

- [x] ✅ 已完成 — **5/5 (100%) 全部成功**

### 结果

| Demo | 指令 | 结果 | 步数 | BC 步数 |
|:-----|:-----|:----:|:----:|:-------:|
| 1 | Put the red cube on the green cube | ✅ | 798 | 100 |
| 2 | Stack cube A on cube B | ✅ | 784 | 77 |
| 3 | Place the cube on top of the other cube | ✅ | 792 | 98 |
| 4 | Move the block onto the target block | ✅ | 788 | 103 |
| 5 | Stack the red block on top of the green one | ✅ | 774 | 100 |

### 关键修复：Symbolic-Physical Goal Semantics Fix

E2E demo 长期 0-1/5 的根因是 `build_stacking_goal` 添加了 `on(bottom, table)` 谓词。物理谓词提取器 `extract_on_predicates` 用桌子中心 (0,0) 做 xy 距离检查（阈值 0.05），堆叠操作后底层块被推离原位，xy 偏移经常超过 0.05，导致谓词检测失败。

**修复**：移除 `on(bottom, table)` 谓词。`run_embodied_stacking_robosuite.build_stacking_goal` 从未包含此谓词，正是为了避免该问题。

其他修复包括：恢复 `InternalStackingPlanner` 的 `[obj, "table", target]` 语义（恢复 bottom-to-top 约定）、修正 `build_stacking_goal` 迭代顺序（逆序生成 `on(顶层, 底层)`）。

## 7. 剩余限制

1. **preplace 仍然是 reach+place 的瓶颈** — 如果未来需要 BC 控制 reach_place，需要改进末端精度
2. **仿真固有失败率 ~12%** — 非策略问题，Robosuite 随机初始化导致
3. **策略崩溃** — ~5% 的 episode 在 reach_pick 早期产出无效动作
4. **确定性不可保证** — Robosuite 1.5.2 不支持 `env.seed()`
5. **仅验证了 stacking 任务** — 其他任务类型（如 pick-place）尚未评估

## 8. 下一步

1. **Qwen 解析器集成** — 替换 MockGoalParser 为 QwenTextGoalParser
2. **针对性的 DAgger Round 2** — 仅当 reach_pick 失败占主导时才进行
3. **更细粒度的仿真日志** — 记录每个 phase 的末端执行器位置
4. **多任务泛化** — 扩展到 pick-place 等场景

## 9. 文件清单

| 文件 | 说明 |
|:----|:----|
| `docs/simulation_protocol.md` | 更新后的规范仿真协议（reach_pick 为 default） |
| `docs/stage_10_deployment_feasibility_report.md` | 本报告 |
| `generated/report_tables/stage_10/reach_pick_only_100ep_summary.md` | reach_pick-only 100ep 汇总 |
| `generated/report_tables/stage_10/deployment_policy_comparison.md` | 部署策略对比 |
| `generated/report_tables/stage_10/e2e_demo_summary.md` | E2E demo 汇总 |
| `generated/results/stage_10/reach_pick_only_seed{0,1,2}_50ep.json` | 3×50ep 验证结果（96/90/94%） |
| `generated/results/stage_10/e2e_sim_demo/*.json` | 端到端 demo 结果（5/5 ✅） |
| `src/vlm_nsm/experiments/export_stage10_deployment_report.py` | 报告导出脚本 |
| `tests/test_stage10_deployment_report.py` | Stage 10 测试 |
| `tests/test_e2e_language_to_sim_outputs.py` | E2E demo 输出测试 |
