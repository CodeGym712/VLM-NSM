# VLM-NSM 已知问题清单

> 更新：2026-06-16 | 来源：代码审查 (review.md) + 数据质量审计 + ID 控制器设计需求分析
> 关联文档：[[controller.md]], [[review.md]], [[progress.md]], [[results.md]]

---

## 一、HDF5 数据质量缺陷（影响 ID 模型训练）

### 1.1 Phase 4 (release) 和 Phase 5 (retreat) 系统性缺失

**严重度**：CRITICAL（对 ID 控制器训练）

**影响范围**：全部数据集 — d1_small_200ep, d1_500ep, control_audit_200ep, v0, v0_phase_fixed, best_config_50ep 等 ~1,500 episodes

**现象**：
```
所有 episode 的 HDF5 phase_id 分布:
  phase_id=0: reach_pick   ✅ ~170步/ep
  phase_id=1: grasp        ✅ ~240步/ep
  phase_id=2: lift         ✅ ~180步/ep
  phase_id=3: reach_place  ✅ ~400步/ep
  phase_id=4: release      ❌ 0步 (仅 episode 末尾 3-5 步误标为 phase_3)
  phase_id=5: retreat      ❌ 0步 (完全不存在)
```

**根因**：`place_state_machine` 内部做完 release → hold → retreat → verify，但从未调用 `_rec_set_phase()` 更新阶段标记。`run_embodied_stacking_robosuite.py` 主循环在 place_state_machine 完成后设置 `place_done=True`，跳过 release/retreat 步骤。

**后果**：
- 无法从现有数据训练 release/retreat 阶段的 ID 模型
- Phase 3 数据实际包含 reach_place (transport + descend) 但不含 release/retreat
- 状态转移 ID 模型只能覆盖 4/6 阶段

**修复方向**：
- 在 place_state_machine 内部添加 phase callback，标记 release/hold/retreat 步骤
- 或重新采集包含完整 6 阶段标签的数据

### 1.2 Phase 名称词汇表不匹配（C2，已修复代码）

**严重度**：HIGH（已修复代码，但已有数据未重新采集）

**现象**：
```
Encoder _PHASE_NAMES (10个):  reach_pick, descend_align, micro_align, grasp_close,
                                lift_verify, transport, descend_place, micro_place,
                                release_verify, retreat

Dataset _phase_map (6个):     0→"reach_pick", 1→"grasp", 2→"lift",
                               3→"reach_place", 4→"release", 5→"retreat"
```

"grasp" ≠ "grasp_close", "lift" ≠ "lift_verify", "reach_place" ≠ "transport", "release" ≠ "release_verify"。不匹配时静默 fallback 到 "retreat"。

**修复状态**：代码已修复 (2026-06-15)，已有 HDF5 数据存的是旧标签（6 阶段名称）。

---

## 二、代码缺陷（review.md 已记录）

### CRITICAL（已修复）

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| C1 | `--zero_symbolic` 训练标志 NO-OP | `vla_dataset.py` | B4 模型训练条件无法验证 |
| C2 | Phase 名称词汇表不匹配 | `symbolic_condition.py` | 4/6 阶段被错误编码 |
| C3 | BFS Planner 接口不兼容 | `skill_plan_from_symbolic.py` | 规划器无法接入下游 |
| C5 | Gate 绕过安全检查 | `eval_s1_phase_override.py` | NaN/workspace 检查被跳过 |
| C6 | S0 脚本 NameError | `eval_vla_live_shadow.py` | 代码无法运行 |
| C7 | MoveOn effects_del 语义错误 | `operators.py` | 下游拿到空列表 |

### HIGH（待修复）

| # | 问题 | 影响 ID 控制器? |
|---|------|:---:|
| H1 | obs_history 读 2 帧只用 1 帧 | ❌ (ID 模型不需要视觉) |
| H2 | Phase-aware 采样权重未使用 | ⚠️ (ID 训练需 phase 均衡采样) |
| H4 | VERIFY 注释与实现不符 | ⚠️ (place_state_machine 逻辑) |
| H5 | Holding check 盲目继续 | ⚠️ (ID 训练数据标签可靠性) |
| H8 | 符号条件维度 31 vs 42 混淆 | ❌ (ID 模型不用符号条件) |
| H11 | 端到端整合未完成 | ❌ |

---

## 三、控制器架构限制（P-controller 具体问题）

### 3.1 需要显式目标位置

每个阶段硬编码 `target = object_pose + offset`，依赖 Oracle 感知提供精确 object_pose。

### 3.2 参数对任务敏感

| 物体 | kp | max_delta | 原因 |
|------|:---:|:---:|------|
| red cube (4cm) | 0.05 | 0.02 | 标准参数 |
| green cube (5cm) | 0.03 | 0.01 | 需要更慢速，否则滑脱 |

### 3.3 Transport 方向一致性差

cos=0.549，VLA 无法控制 reach_place。P-controller 沿直线移动，没有轨迹优化。

### 3.4 3-cube 100% 倒塌

放置路径碰撞已堆叠结构。P-controller 无碰撞感知。

### 3.5 无速度/加速度建模

纯位置误差比例控制，无速度前馈、无积分消除稳态误差、无阻尼抑制震荡。

---

## 四、方法论问题（影响论文）

| # | 问题 | 状态 |
|---|------|:---:|
| C4 | Gate 循环论证 — 度量 teacher-alignment 而非 task-success | 📝 叙事层面 |
| S1 | b4_clean 70% 是异常值（其他 B4 为 48-64%） | 📝 |
| S2 | B4.5 cos_sim=0.434 无法在 report.json 中定位 | 📝 |
| S3 | 所有 21 个 report.json 无置信区间 | 📝 |
| S4 | Teacher 56%→70% 声称与数据不符 | 📝 |
| S5 | VLA-caused failures metric 与 gate 配置耦合 | 📝 |
