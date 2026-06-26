# VLM-NSM 规范仿真协议

> 更新时间：2026-05-17（Stage 10）

## Canonical Checkpoint

```
generated/results/stage_7B-4/dagger_reach_bc/best.pt
```

DAgger Reach BC，MLP(17→256→256→7)，72K params，phase-conditioned。训练数据：Expert v2 strict_only（100ep, 61,618 steps）+ DAgger round1（30ep, 18,820 steps）= 80,438 steps 聚合数据集。

## Canonical Config

```
generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json
```

## Canonical Eval Command

```bash
PYTHONUNBUFFERED=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<GPU> \
python3 -u -m vlm_nsm.learning.eval_bc_hybrid \
  --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
  --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
  --num_episodes <N> \
  --seed <SEED> \
  --condition_on_phase \
  --bc_phases <reach_pick reach_place | reach_pick | none> \
  --fixed_zero_action_dims 3 4 5 \
  --device cuda \
  --output <OUTPUT_PATH>
```

### Flags 说明

| Flag | 值 | 说明 |
|------|:--:|------|
| `PYTHONUNBUFFERED=1` | — | Python 无缓冲输出 |
| `MUJOCO_GL=egl` | — | 无头渲染（服务器环境） |
| `CUDA_VISIBLE_DEVICES` | N | 指定 GPU |
| `--device cuda` | — | 使用 GPU 推理（默认 CPU 导致极慢） |
| `--condition_on_phase` | — | 在观察中添加 phase one-hot |
| `--bc_phases` | `reach_pick reach_place` | DAgger hybrid BC 模式 |
| `--bc_phases` | `reach_pick` | 仅 BC 控制 pick reaching |
| `--bc_phases` | `none` | 纯 scripted 基线 |
| `--fixed_zero_action_dims` | `3 4 5` | 固定 orientation dims 为零 |

## Canonical bc_phases

规范部署策略经过 Stage 9-10 系统性评估确认。

### Deployment Default

- **`reach_pick`** — Stage 10 确认 **92.0% (92/100)**，3×50ep avg **93.3%**，preplace=0。推荐作为 canonical deployment policy。

### Research Variants

- **DAgger Hybrid `reach_pick reach_place`**: Stage 9 3×50ep avg **83.3%**，preplace 为主要失败模式（avg 43.7/150ep）。因 BC 末端精度不足导致放置验证失败，不推荐作为默认部署策略。
- **Scripted baseline `none`**: 纯 scripted 执行 **88.0% (88/100)**，提供仿真固有成功率上限参考。

### 评估汇总

| Policy | N | Success | Preplace | Verdict |
|:-------|:-:|:-------:|:--------:|:--------|
| Scripted-only | 100ep | 88.0% | 0 | 仿真上限参考 |
| DAgger reach+place | 3×50ep | 83.3% avg | 43.7 avg | ⚠️ preplace 瓶颈 |
| **DAgger reach_pick-only (100ep)** | **100ep** | **92.0%** | **0** | **✅ 推荐默认** |
| **DAgger reach_pick-only (3×50ep)** | **150ep** | **93.3% avg** | **0.7 avg** | **✅ 多 seed 确认** |

## 评估指标

| 指标 | 说明 |
|------|------|
| success_rate | goal_success 比例 |
| mean_steps | 平均 episode 步数 |
| p95_steps | 95 百分位步数（应 < 1000） |
| mean_bc_steps / mean_scripted_steps | BC 与 scripted 步数分布 |
| failure_type_counts | 失败类型及发生次数 |

## 失败类型

| 类型 | 说明 |
|:----:|------|
| horizon | 超时（1000 steps） |
| preplace | 放置验证失败 |
| verify_failed | 目标谓词验证失败 |

## 废弃路径（禁止使用）

| 废弃项 | 替代 |
|--------|------|
| `eval_bc.py` 手动 env loop（horizon//6 phase 分割、非 BC phase 零动作） | `eval_bc_hybrid.py` + `run_episode()` |
| 合成 phase_id（基于 step budget 推断） | 真实 phase_id（从 RobosuiteEnvWrapper 状态机获取） |
| `strict_success/` 混合训练目录 | `strict_only/`（100 个严格验证的 demo） |
| Offline MSE 作为主要成功指标 | 闭环 success rate |
| BC 控制 grasp 或 reach_place phase | BC 限于 reach_pick（canonical default） |

## 已知限制

1. **Robosuite 1.5.2 不支持 `env.seed()`**: `--seed` 仅作为元数据记录，环境随机初始化在不同运行间可能不同
2. **Offline MSE 不预测闭环性能**: DAgger-only 模型 val loss 0.000627 但闭环 0%
3. **DAgger-only 数据不足**: 纯 on-policy 数据（18,820 样本）需要结合 expert 数据（61,618 样本）
4. **Stage 8 100ep DAgger 80.0%**: 从 Stage 7B-5 的 92-93% 回落，确认小样本偏差
5. **reach_place BC 引入 preplace 失败**: DAgger reach_pick+reach_place avg 83.3%，preplace 43.7/150ep；reach_pick-only 92.0%，preplace=0。reach_place 末端精度不足是策略当前最核心瓶颈
6. **仿真固有失败率**: Scripted 基线 88.0% 表明 Robosuite 仿真本身有 ~12% 的随机失败率
