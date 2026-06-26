note/
├── progress.md              ← 项目仪表盘（每天更新）
├── problem.md               ← 已知问题清单
├── design/
│   ├── controller.md        ← 系统架构总览
│   └── generalization.md    ← 泛化性改造方案
├── logs/
│   └── 2026-06-16-pid.md    ← 今日详细日志
└── archive/
    ├── learn.md             ← 新人教程
    ├── results.md           ← 历史实验结果
    ├── review.md            ← 代码审查报告
    ├── log.md / log.md.bak  ← 旧结构日志
    └── presentation_script.md
# VLM-NSM 项目进度汇报 & 逆动力学控制器改造方案

> 日期：2026-06-24 | 汇报人：zhaowm
> 代码：240+ 源文件，1591 测试 | GPU：8× RTX 4090
> 阶段：收敛冻结 | 论文初稿完成 | 四任务闭环 | Gate Attribution 四重证据

---

## 一、项目要解决什么问题

**目标**：让机器人听懂自然语言指令（如"把红色方块放到绿色方块上面"），通过摄像头观察环境，自主完成物体堆叠操作。

**核心挑战**：机器人需要同时解决四个子问题——理解语言、规划动作序列、从像素生成机械臂动作、检测并恢复错误。

**我们的方案**：构建一个"神经符号 VLA 系统"——符号系统负责任务规划（精确可验证），神经系统负责视觉运动控制（从像素到动作），通过**三层桥接**（符号条件编码、安全门控、算子监测）将两个系统紧密耦合。

---

## 二、当前整体进度

```
组件                          状态      关键指标
────────────────────────────────────────────────────
语言→符号规划 (Qwen)            ✅       解析准确率 100%
BFS 符号规划器                   ✅       默认, 2-5 block, vs InternalStacking 符号层对比完成
PID 逆动力学控制器               ✅       2-block 91% (100ep)
技能执行 (Grasp/Place SM)       ✅       抓取/放置状态机, skip_approach 修复
OperatorMonitor 闭环             ✅       87% 恢复率, 3-block replan 93%*
E2E: Language→PID→Monitor       ✅       Qwen parse 5/5, 管线连通
TaskSpec 统一                   ✅       action_handler + init_hook + planner_kwargs
Hanoi 3-disk                    ✅       84% (50ep), mean=5,716 专用 eval 脚本
Pull Task                       ✅       85% (20ep), handle 位移 8.1cm, 新技能原语
Gate Attribution                 ✅       四重证据链, summarization 脚本, 参考文档
Alignment 消融                   ✅       56 条 case, 6 级梯度
BFS vs InternalStacking         ✅       符号层对比, BFS 正确拆堆
论文初稿                         ✅       paper_draft.md (中文, ~3000 字)
代码审查 + 审计修复              ✅       6 CRITICAL + 13 项
3-block 93%                      ⚠️      历史数据, GainSchedule 改变后路径失效
Slide Joint Drawer               ❌      MuJoCo 无静摩擦, 物理限制, 论文 limitations
~~VLA Diffusion~~                ❌      仅 reach_pick 38% gate 通过率
感知层 (YOLO+stereo)             🚧      Stage 27 验证 Δ=-1.7pp, 实时回路待接入
```

---

## 三、逆动力学控制器改造 — 实施日志

> 实施日期：2026-06-16 | 状态：✅ Phase 1-3 完成，Phase 4 评估完成

### 3.1 整体架构

采用**分层逆动力学控制器**：

```
Layer 2: Learned ID Model (~25K params, 3-model Ensemble)
  input: (eef_pos, gripper, phase, desired_next_eef) → output: target_velocity
  role: 自适应、轨迹优化

Layer 1: Operational Space PID (解析型)
  v_des = kp*e + ki*∫e - kd*v + v_ff
  anti-windup, workspace bounds, NaN protection
  role: 稳定性、安全性

Transition State Machine
  语义子状态替代硬编码 target
```

### 3.2 新增文件

```
src/vlm_nsm/embodied/controllers/
  operational_space_pid.py          # Layer 1: PID + 增益调度 (280行)
  learned_id_model.py               # Layer 2: MLP + Ensemble (190行)

src/vlm_nsm/embodied/
  transition_state_machine.py       # 语义状态机 (230行)

src/vlm_nsm/data/
  id_training_dataset.py            # HDF5 → ID 训练对 (200行)

src/vlm_nsm/training/
  train_inverse_dynamics.py         # ID 训练脚本 (130行)

tests/
  test_operational_space_pid.py     # 34 tests
  test_learned_id_model.py          # 12 tests
  test_transition_state_machine.py  # 11 tests

generated/results/id/
  id_model_v1/best.pt               # val_loss=0.354
  id_model_v2/best.pt               # val_loss=0.345
  id_model_v3/best.pt               # val_loss=0.350

note/
  problem.md                        # 已知问题清单
```

### 3.3 修改文件

| 文件 | 修改内容 |
|------|---------|
| `run_embodied_stacking_robosuite.py` | +`use_pid_controller` 参数 + `_pid_reach_loop` 辅助函数 + CLI `--use-pid-controller` |
| `embodied/controllers/__init__.py` | 无需修改（已有注释） |

### 3.4 PID 工作流程

```
每步执行:
  env.get_eef_pose() → set_current(pos)
    ├─ 有限差分: vel = (pos_t - pos_{t-1}) / dt
    └─ 存储 current_pos

  compute_pid_velocity():
    error = target - current
    p = kp * error
    i = ki * clip(∫error·dt, ±i_max)     ← anti-windup
    d = kd * (target_vel - current_vel)   ← 速度阻尼
    ff = target_vel                        ← Learned ID 前馈(当前=0)
    v = clip(p+i+d+ff, ±max_delta)
    v = clip_to_workspace(v)              ← 安全边界
    v = nan_check(v)                      ← 兜底

  act(phase_name):
    action[:3] = v / 0.05                  ← 归一化到[-1,1]
    action[6] = gripper_map[phase]

  _pid_reach_loop(..., reach_threshold=0.02):
    for step in range(500):
      set_current(env.get_eef_pose())
      action = act()
      env.step(action)
      if dist(eef, target) < 0.02 → 收敛
```

### 3.5 Phase 专用增益调度

| Phase | kp | ki | kd | max_delta | i_max |
|-------|:---:|:---:|:---:|:---:|:---:|
| reach_pick | 0.10 | 0.005 | 0.0 | 0.02 | 0.03 |
| grasp | 0.07 | 0.002 | 0.0 | 0.01 | 0.02 |
| lift | 0.10 | 0.002 | 0.0 | 0.02 | 0.03 |
| reach_place | 0.08 | 0.005 | 0.0 | 0.015 | 0.03 |
| release | 0.04 | 0.001 | 0.0 | 0.005 | 0.01 |
| retreat | 0.08 | 0.001 | 0.0 | 0.02 | 0.03 |
| *注: kd=0 全 phase (grid search 最优)，值来自 GainSchedule 默认* |

### 3.6 Learned ID Model 训练

| 指标 | 数值 |
|------|:---:|
| 训练样本 | 98,250 (122 success episodes) |
| 验证样本 | 24,062 |
| 模型参数 | 28,484 (~25K) |
| 输入维度 | 26 (eef+gripper+phase+desired_next) |
| 输出维度 | 4 (dx, dy, dz, gripper) |
| v1 val_loss | 0.354 |
| v2 val_loss | 0.345 |
| v3 val_loss | 0.350 |
| Ensemble 置信度 | 0.06-0.09 (低, 缺物体位姿特征) |

### 3.7 数据质量问题

详见 [problem.md](note/problem.md)：
- Phase 4/5 (release/retreat) 在所有 ~1,500 HDF5 episodes 中系统性缺失
- Phase 名称词汇表不匹配（C2，代码已修复）
- ID 模型仅覆盖 phases 0-3，release/retreat 保持 scripted

### 3.8 VLA vs ID vs Scripted 全对比评估

| 控制器 | Episodes | Success | Gate Accept | reach_pick Phase | Mean Steps |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Scripted (P)** | 100 | **94%** | — | — | 988.9 |
| **PID v1 (default)** | 100 | **90%** | — | — | 976.6 |
| VLA B5 s9000 | 50 | 48% | 51.7% | 100% | 970.0 |
| VLA B5 s12345 | 50 | 52% | 53.4% | 100% | 954.5 |
| VLA B5 s42 | 50 | 58% | 51.2% | 100% | 969.4 |
| **VLA B5 avg** | 150 | **52.7%** | 52.0% | 100% | 964.6 |

> **VLA 数据说明**：上表为 6/16 评估（3×50ep, gate 独立校准集）。**Canonical 结果见 Section 五**：B5=46.5%±1.3%（4×50ep, 6/13-15, 独立 gate 校准）。两组差异来自 gate 配置和 checkpoint epoch。论文推荐使用 Section 五的 46.5% 作为主结果。

**PID 增益调优历程**：

| 版本 | lift kp | lift ki | Success | lift_failed | 分析 |
|------|:---:|:---:|:---:|:---:|------|
| v1 (default) | 0.06 | 0.001 | 90% | 1 | 基线，可接受 |
| v2 (tuned) | 0.08 | 0 | 90% | 4 | ki=0 导致稳态误差，更差 |
| **v1 (final)** | **0.08** | **0.001** | **90%** | **1** | lift 需微量积分克服重力 |

### 3.9 关键发现

1. **PID 与 P-controller 功能等价**：90% vs 94%，4pp 差距在 100ep 噪声范围内
2. **Scripted P-controller 仍是金标准**（94%），作为 teacher baseline
3. **VLA 不能独立完成任务**（52.7%），Gate 只接受 ~52% 的 VLA 动作
4. **VLA reach_pick 阶段 100% 安全**，transport（cos=0.549）是瓶颈
5. **Learned ID 需要更好特征**：缺物体位姿导致 ensemble 置信度低（0.06-0.09）
6. **积分项对 lift 重要**：ki=0 → lift_failed 增加 4x（1→4）
7. **MuJoCo 物理仿真是 CPU 瓶颈**（98% CPU），GPU 仅用于 VLA 推理（~17% 步骤）

### 3.10 测试覆盖

```
1608 passed, 2 skipped (0 回归)
新增专项测试: 34 (PID) + 12 (ID Model) + 11 (TSM) = 57 tests
```

### 3.11 PID 增益调优结果

Grid search on control_audit_200ep (50 episodes, 10 validation):

| Phase | kp | ki | kd | max_delta | 分析 |
|------|:---:|:---:|:---:|:---:|------|
| reach_pick | **0.10** | **0.005** | 0.0 | 0.02 | 更高 kp 加速接近 |
| grasp | **0.07** | **0.002** | 0.0 | 0.01 | 适中，防止 overshoot |
| lift | **0.10** | **0.002** | 0.0 | 0.02 | 更快提升 |
| reach_place | **0.08** | **0.005** | 0.0 | 0.015 | 更强 I 消除稳态误差 |

**关键发现**：**kd=0 对所有 phase 最优**。D 项（速度阻尼）在此任务中不提供价值，因为 robosuite 的 OSC 控制器内部已处理阻尼。

### 3.12 ID 模型特征增强结果

| 指标 | 旧 (26-dim) | 新 (38-dim) | 改善 |
|------|:---:|:---:|:---:|
| 输入特征 | eef+gripper+phase+desired | +vel_history(9)+act_history(12) | — |
| val_loss | 0.35 | **0.0005** | **700x** |
| Ensemble 置信度 | 0.06-0.09 | **0.55-0.71** | **10x** |
| 模型参数 | 25K | 30K | +20% |

### 3.13 PID 最终评估（调优后）

| 控制器 | Success | Mean Steps | Failures |
|--------|:---:|:---:|------|
| P-controller | **94%** | 988.9 | horizon(5), preplace(1) |
| PID v1 (default) | 90% | 976.6 | horizon(9), lift_failed(1) |
| **PID tuned (final)** | **91%** | **949.6** | horizon(9) |

**分析**：
- lift_failed 消除（ki=0.002 有效消除稳态误差）
- preplace 消除（ki=0.005 在 reach_place 中消除位置偏差）
- horizon(9) 与 P-controller 的 horizon(5)+preplace(1)=6 相当
- 3pp 差距主要来自 horizon 超时（PID 的 I 项初始累积慢于纯 P）

### 3.14 端到端 Language→PID→Monitor 验证

使用 `run_e2e_language_to_sim.py`，完整链路：

```
NL Instruction → MockGoalParser → Alignment → InternalStackingPlanner
    → PID 执行 (6 phases) → Predicate Monitor 检查 on(cube1,cube2)
```

| Demo | Instruction | 结果 |
|:---:|------|:---:|
| 1 | "Put the red cube on the green cube." | ✅ |
| 2 | "Stack cube A on cube B." | ✅ |
| 3 | "Place the cube on top of the other cube." | ❌ 语义模糊，物理放置失败 |
| 4 | "Move the block onto the target block." | ✅ |
| 5 | "Stack the red block on top of the green one." | ✅ |
| **总计** | | **4/5 (80%)** |

### 3.15 PID 全盘采用最终对比

| 控制器 | Success | Mean Steps | lift_failed | preplace | horizon |
|--------|:---:|:---:|:---:|:---:|:---:|
| P-controller (baseline) | **94%** | 988.9 | 0 | 1 | 5 |
| PID v1 (default gains) | 90% | 976.6 | 1 | 0 | 9 |
| **PID tuned (final)** | **91%** | **949.6** | 0 | 0 | 9 |

3pp 差距来自 I 项初始累积慢于纯 P 响应，可通过 ID 模型前馈缩小。

---

## 四、系统架构（当前版本：训练无关 PID+BFS+Qwen）

```
┌─────────────────────────────────────────────────┐
│  第1层：语言→符号 (Qwen API + Alignment)          │
│  "红色方块放绿色上面" → Goal JSON → 6层校验 → Goal │
│  仅输出目标谓词，不做规划                          │
├─────────────────────────────────────────────────┤
│  第2层：符号规划 (BFS Symbolic Planner)            │
│  SceneGraph + 目标 → BFS 搜索 → Action 序列        │
│  算子: MoveOn / MoveToTable, <10ms, ≤1000 states  │
├─────────────────────────────────────────────────┤
│  第3层：技能执行 (SkillLibrary + PID)              │
│  reach → grasp → lift → reach_place → release    │
│  新增: pull (Pull Task 85%)                      │
│  PID 逐阶段增益调度, kd=0, 零训练                  │
├─────────────────────────────────────────────────┤
│  第4层：监测与恢复 (OperatorMonitor + Replan)      │
│  每步验证 predicate 效果 → 失败时 replan           │
│  3-block: direct 0% → replan 93%                 │
└─────────────────────────────────────────────────┘
```

**设计原则**：系统控制端零训练。PID 参数手工 grid search 调优。BFS 符号搜索保证动作合法性。Qwen 仅做语义翻译不输出动作。VLA/Diffusion Policy/ID Model 已放弃——gate attribution 证明其在精确操纵中仅 reach_pick 有贡献（38%），放宽控制份额伤害成功率（60%→15%）。

---

## 四、代码审查发现与修复（2026-06-15）

本周进行了全面的深度代码审查（5 个并行审计 + 直接取证），发现了 7 个关键问题，其中 6 个已完成修复：

| # | 问题 | 文件 | 影响 | 状态 |
|---|------|------|------|:---:|
| C1 | `--zero_symbolic` 训练标志无效 | `vla_dataset.py` | B4 模型训练条件无法验证 | ✅ |
| C2 | 阶段名称词汇表不匹配 | `vla_dataset.py` | 4/6 阶段被错误编码 | ✅ |
| C3 | BFS Planner 接口不兼容 | `skill_plan_from_symbolic.py` | 规划器无法接入下游 | ✅ |
| C5 | Gate 绕过安全检查 | `eval_s1_phase_override.py` | NaN/workspace 检查被跳过 | ✅ |
| C6 | S0 脚本 NameError | `eval_vla_live_shadow.py` | 代码无法运行 | ✅ |
| C7 | MoveOn effects_del 语义错误 | `operators.py` | 下游拿到空列表 | ✅ |
| C4 | Gate 循环论证 | 叙事层面 | 需要重新表述贡献 | 📝 |

**测试验证**：54 个相关测试全部通过，无回归。

---

## 五、核心实验：B4 vs B5 对照

### 实验设计

B4 和 B5 是同一个 Diffusion Policy 模型，唯一区别是是否接收符号条件：

| 条件 | B4（无符号条件） | B5（有符号条件） |
|------|:---:|:---:|
| 摄像头画面 (RGB) | ✅ | ✅ |
| 机械臂位置 (Proprioception) | ✅ | ✅ |
| 符号条件 (42维) | ❌ 全零 | ✅ 真实编码 |

### 修复后的结果（新 B4 checkpoint，2026-06-15）

**完整 B4 vs B5 对照表**：

| 条件 | Seeds | Gate Acc (avg±std) | Success (avg±std) |
|------|:---:|:---:|:---:|
| **B4（零符号）** | 2 × 100ep | **0.0 ± 0.0%** | **63.5 ± 0.5%** |
| **B5（全符号）** | 4 × 50ep | **46.5 ± 1.3%** | **59.5 ± 5.7%** |

| | B4 s9000 | B4 s12345 | B5 s9000 | B5 s12345 | B5 s42 | B5 s9999 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gate | 0.0% | 0.0% | 46.8% | 47.3% | 44.6% | 47.2% |
| Succ | 64% | 63% | 64% | 58% | 52% | 64% |
| Ep | 100 | 100 | 50 | 50 | 50 | 50 |

### 关键发现

**符号条件的效果比旧数据声称的更强：不是 +22pp 改善，而是从"完全无效(0%)"到"近半自主(47%)"的质变。**

- B4 gate=0.0%：真正零符号训练的 VLA 无法做出任何通过安全门控的动作预测
- B5 gate=46.5%±1.3%：符号条件使 VLA 能够控制近半的 reach_pick 步骤
- 4 seeds 间 gate 方差仅 ±1.3pp，证明结果高度可复现
- Task success 在两种条件下接近（63.5% vs 59.5%），因为 scripted teacher 在 gate 拒绝 VLA 时兜底
- 旧 B4 checkpoint 的训练条件已被证实污染——其 27-31% gate 是因为旧代码 `--zero_symbolic` 是 NO-OP

---

## 六、OperatorMonitor：从低层失败到符号语义

### 核心发现

50ep Operator Audit 发现：**100% 的放置失败伴随 `holding(x)=False`**——物体在运输中滑落，而非放置精度问题。

修复方向从"改进放置精度"转向"防止运输中滑落"：
- 运输前检查物体是否抓牢 → 不牢则重抓
- 运输中使用 safe place 模式
- Teacher 成功率：~73% → ~80%（+7pp）

### 方法论价值

OperatorMonitor 建立了一套将低层控制器失败映射到符号算子层的结构化方法。这不是简单的"发现了一个 bug"，而是**提供了一种将物理执行问题转化为符号诊断的通用框架**。

---

## 七、逆动力学控制器改造方案 🚧

### 7.1 问题：当前 P-controller 架构限制

当前系统使用纯粹的 P-controller（比例控制器）+ 状态机：

```python
# 当前: run_embodied_stacking_robosuite.py
dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)
```

| 问题 | 根因 | 影响 |
|------|------|------|
| transport cos=0.549 | P-controller 按直线移动，无路径优化 | VLA 不能控制 reach_place |
| 3-cube 100% 倒塌 | 放置路径碰撞已堆叠结构 | 不能扩展到多物体 |
| green cube 需特殊参数 | kp/max_delta 需按物体调优 (kp=0.03 vs 0.05) | 泛化性差 |
| 依赖 GT object pose | 目标位置来自 Oracle 感知 | 不能部署真实环境 |
| 稳态误差 | 无积分项消除系统偏差 | grasp 精度受限 |
| 无速度建模 | 无微分项抑制震荡和 overshoot | 放置阶段碰撞 |

### 7.2 方案：分层逆动力学控制器

采用**分层 ID 架构**：底层 PID 保证安全，高层 Learned ID 提供自适应能力。

```
                          SymbolicPlan
                               │
                               ▼
                   ┌──────────────────────┐
                   │ Transition State     │  ← 新: 语义子状态定义
                   │ Machine              │
                   │ desired_substate ←── │
                   └──────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        reach_pick       transport       grasp/release
              │               │               │
              ▼               ▼               ▼
    ┌────────────────────────────────────────────────┐
    │         Hierarchical ID Controller              │
    │                                                  │
    │  Layer 2: Learned ID Model (~25K params)        │
    │    input: (obs, desired_substate)               │
    │    output: target_velocity                      │
    │    role: 自适应、轨迹优化                         │
    │    training: ~950 episodes HDF5 → (s_t,s_{t+1})→a_t │
    │                      │                          │
    │                      ▼                          │
    │  Layer 1: Operational Space PID (解析型)         │
    │    v_des = kp*e + ki*∫e - kd*v + v_ff           │
    │    anti-windup clamp on integral                │
    │    workspace bounds check                       │
    │    NaN protection                               │
    │    role: 稳定性、安全性、物理约束                  │
    └────────────────────┬───────────────────────────┘
                         │
                         ▼
                  robosuite OSC_POSE
```

**关键设计原则**：

1. **Layer 1 (PID) 始终在线** —— 即使 Layer 2 输出不合理，PID 有积分限制+输出钳位+工作空间约束兜底
2. **Layer 2 (Learned ID) 可渐进接入** —— 支持 `pure_pid` / `pure_learned` / `hybrid` 三种模式
3. **Transition State Machine** 替代硬编码 target —— 从 `target_z = obj_z + 0.10` 改为 `desired_substate = "eef_above_pick_object"`
4. **与 VLA 可对比** —— VLA 和 ID Controller 共享 Phase 接口，Gate 统一仲裁

### 7.3 Layer 1: 操作空间 PID 控制器

**核心公式**：

```
error     = target_pos - current_pos
p_term    = kp * error
i_term    = ki * clip(integral_error, -i_max, i_max)  # anti-windup
d_term    = kd * (-current_vel + target_vel)          # velocity damping
ff_term   = target_vel                                 # feedforward (from Layer 2)
v_des     = p_term + i_term + d_term + ff_term
v_des     = clip(v_des, -max_delta, max_delta)
action[:3] = v_des / OSC_POSE_MAX_DELTA  # normalize to [-1, 1]
```

**Phase 专用增益调度**：

| Phase | kp | ki | kd | max_delta | 原因 |
|-------|:---:|:---:|:---:|:---:|------|
| reach_pick | 0.08 | 0.002 | 0.01 | 0.02 | 快速接近，无需精细 |
| grasp | 0.05 | 0.001 | 0.008 | 0.01 | 慢速精细，抑制 overshoot |
| lift | 0.06 | 0.001 | 0.01 | 0.02 | 中等速度直线提升 |
| reach_place | 0.06 | 0.002 | 0.015 | 0.015 | 运输需稳定，稍强阻尼 |
| release | — | — | — | — | 保持 scripted place_state_machine |
| retreat | — | — | — | — | 保持 scripted place_state_machine |

**安全机制**：
- 积分 anti-windup：`clip(integral, -0.03, 0.03)`
- 积分在 target 变化时清零
- 工作空间预测：`eef + v_des*dt` 超出范围则 clip v_des
- NaN 中间值检查：任何 NaN → 输出零动作

### 7.4 Layer 2: 学习型逆动力学模型

**数据来源与限制**：

经过全面 HDF5 数据审计（详情见 `note/problem.md`），发现所有 ~1,500 episodes 存在系统性数据缺陷：
- Phase 0-3 (reach_pick, grasp, lift, reach_place): ✅ 可用，~990 step/ep
- Phase 4-5 (release, retreat): ❌ 系统性缺失，place_state_machine 内部执行但未标记

**ID 模型训练范围**：仅覆盖 phases 0-3。release/retreat 保持 scripted (place_state_machine)。

**数据构建**：

```python
# 从 HDF5 提取: (obs_t, obs_{t+1}) → action_t
# 每条训练样本:
#   input:  [eef_pos_t(3), gripper_t(1), object_poses_t(N*3), phase_onehot(10)]
#   target: action_t(4)  # [dx, dy, dz, gripper]
```

| 数据集 | Episodes | 用途 |
|--------|:---:|------|
| D1-small (200ep) | 200 | ID 模型主训练和验证 |
| D1-500 | 500 | ID 模型 scaling test |
| ControlAudit | 200 | 困难案例 fine-tune |
| **合计** | **~900** | **~900K action steps** |

**模型设计**（轻量，与 VLA 互补）：

```python
class LearnedInverseDynamics(nn.Module):
    """轻量 MLP: ~25K params (vs VLA's 26.8M, ~1000x smaller).
    
    输入 (无视觉):
      eef_pos:        3   # 当前末端位置
      gripper_state:  1   # 夹爪开度
      object_poses:   9   # 3 objects × 3
      phase_onehot:   10  # 当前阶段
      desired_next:   3   # 目标 EEF 位置 (来自 TransitionStateMachine)
      ──────────────────
      total:          26 维
    
    输出: [dx, dy, dz, gripper] (4维)
    """
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(26, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 4),
        )
```

**不确定性估计**（3 模型 Ensemble）：

```python
def predict_with_confidence(self, obs, desired_substate):
    preds = [m(obs, desired_substate) for m in self.ensemble]  # 3 models
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    confidence = 1.0 / (1.0 + np.linalg.norm(std))
    return mean, confidence
```

**Layer 1 + Layer 2 融合**：

```python
def hybrid_action(pid_action, learned_action, learned_confidence):
    if learned_confidence < 0.7:
        return pid_action, "pid_fallback"
    blend = 0.3 * pid_action + 0.7 * learned_action  # PID floor=30%
    return blend, "hybrid"
```

### 7.5 Transition State Machine

替代当前硬编码的 6 阶段 target 坐标，改为语义子状态驱动：

```
State: APPROACH_PICK
  desired_substate: "eef_above_pick_object"
  target_fn: obj_pose_xy + (obj_z + approach_height)
  controller: PID (kp=0.08, kd=0.01)
  completion: dist(eef_xy, obj_xy) < 0.015 AND dist(eef_z, target_z) < 0.02

State: GRASP_EXECUTE
  desired_substate: "object_in_gripper"
  target_fn: obj_pose_xy + (obj_z + descend_offset)
  controller: PID (kp=0.05, kd=0.008) + Learned ID
  completion: gripper_contact AND holding_check
  on_failure: → GRASP_RETRY (max 3)

State: LIFT_OBJECT
  desired_substate: "object_lifted"
  target_fn: obj_pose_xy + (obj_z + lift_height)
  controller: PID (kp=0.06, kd=0.01)
  completion: obj_z_delta > lift_threshold

State: TRANSPORT_TO_TARGET
  desired_substate: "eef_above_target"
  target_fn: target_pose_xy + (target_z + preplace_height)
  controller: PID (kp=0.06, kd=0.015) + Learned ID
  completion: dist(eef_xy, target_xy) < 0.015
  holding_check: periodic (every 10 steps)

State: RELEASE_AND_RETREAT  ← scripted place_state_machine
  desired_substate: "object_placed"
  executor: PlaceStateMachine (safe mode)
  completion: on_predicate_verified
```

**关键改进**：
- 不再硬编码 `target_z = obj_z + 0.10`，改为从 `desired_substate` 规则推导
- Learned ID 只在有意义阶段接入：grasp + transport
- Release/retreat 保持 scripted（简单操作 + 无训练数据）

### 7.6 VLA vs ID Controller 对比评估

**对比方案**：

| 方案 | 描述 |
|------|------|
| A: VLA | VLA Diffusion Policy (全6阶段) + Safety Gate + Scripted Fallback |
| B: ID | 分层 ID Controller (phase 0-3) + Scripted release/retreat |
| C: Scripted | 当前 P-controller teacher (baseline) |

**评估协议**：
- 3 seeds × 100 episodes = 300 episodes per condition
- 2-block: red_on_green + green_on_red 交替
- Gate 使用独立校准集（control_audit_200ep 留出 50ep）
- Bootstrap 1000 resamples, 95% CI

**评估维度**：

| 维度 | 指标 | 说明 |
|------|------|------|
| Task Success | overall, per-phase | **金标准** |
| Action Quality | cos_sim vs teacher, jerk, path efficiency | 轨迹质量 |
| Robustness | green cube success, cross-seed variance | 泛化性 |
| Efficiency | avg steps/ep, inference time/step | 计算开销 |

**决策标准**：
- Overall success: B > A + 5pp → 选 B; A > B + 5pp → 选 A; otherwise → 融合方案
- reach_pick 保留 VLA（已验证 cos=0.88）
- transport 如果 ID 显著改善（cos > 0.65），ID 接管

### 7.7 数据质量问题与缓解

详见 `note/problem.md`：

| 问题 | 影响 | 缓解 |
|------|------|------|
| Phase 4/5 缺失 | ID 模型不能训练 release/retreat | 保持 scripted place_state_machine |
| Phase 名称不匹配 (C2) | 已有数据标签不兼容 | 代码已修复，ID 训练用 phase_id 整数 |
| obs_history 只用1帧 (H1) | 无速度信息 | ID 模型用 eef_pos 差分计算速度 |
| Holding check 盲目继续 (H5) | 训练数据标签噪声 | 用 contact + z_delta 双重验证 |

### 7.8 实施路线图

```
Phase 1: PID Controller 基础 (Week 1-2) ← 当前
  Day 1-2: OperationalSpacePID 实现 + 单元测试
  Day 3-4: 接入 run_embodied_stacking_robosuite.py
  Day 5-6: 参数调优 + 2-cube 100ep baseline eval
  Day 7:   对比 Pure P vs Pure PID, 验证不劣于 P-controller

Phase 2: Learned ID Model (Week 3-4)
  Day 1-2: 从 HDF5 构建 ID 训练数据集
  Day 3-5: 训练 Learned ID 模型 (~25K params)
  Day 6-7: 集成 Layer 1 + Layer 2, 混合推理

Phase 3: Transition State Machine (Week 5)
  Day 1-3: 实现 TransitionStateMachine
  Day 4-5: 替换硬编码 target, 接入 ID Controller
  Day 6-7: 2-cube 完整评估 + 回归测试

Phase 4: VLA vs ID 对比评估 (Week 6)
  Day 1-3: 300ep × 3 conditions 全阶段评估
  Day 4-5: 数据分析 + bootstrap CI + per-phase breakdown
  Day 6-7: 决策: VLA / ID Controller / 融合方案
```

### 7.9 文件变更清单

```
新增文件:
  src/vlm_nsm/embodied/controllers/operational_space_pid.py   # Layer 1 PID
  src/vlm_nsm/embodied/controllers/learned_id_model.py        # Layer 2 Learned ID
  src/vlm_nsm/embodied/controllers/hierarchical_id.py         # 分层融合
  src/vlm_nsm/embodied/transition_state_machine.py            # 状态转移
  src/vlm_nsm/embodied/transition_definitions.py              # 子状态定义
  src/vlm_nsm/training/train_inverse_dynamics.py              # ID 训练脚本
  src/vlm_nsm/data/id_training_dataset.py                     # ID 数据构建
  tests/test_operational_space_pid.py
  tests/test_learned_id_model.py
  tests/test_transition_state_machine.py
  tests/test_hierarchical_id.py

修改文件:
  src/vlm_nsm/embodied/controllers/__init__.py
  src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py  # 主循环接入 ID
  src/vlm_nsm/experiments/collect_vla_demos.py                # 修复 phase 4/5 标签
  src/vlm_nsm/embodied/place_state_machine.py                 # 添加 phase callback

产出:
  generated/configs/control/pid_gain_schedule.json
  generated/results/id/id_model_checkpoints/
  generated/results/id/vla_vs_id_eval/
```



---




## 九、下一阶段计划（2026-06-19 修订）

> 核心反馈：当前任务太简单，规划器作用不明显。新方向：先做复杂任务验证，同时让 BFS 规划器的价值可量化。

### 系统定位

```
训练无关 (training-free):
  Language: Qwen API          ← 零训练
  Planning:  BFS Symbolic     ← 零训练
  Control:   PID Controller   ← 手工调参
  Monitor:   OperatorMonitor  ← 手工规则

已放弃:
  VLA Diffusion Policy        ← gate=0%, 不可行
  Learned ID Model            ← 多对一, 不可行
  谓词编码器                   ← 数据泄漏, 未集成, 当前不需要
```

### 管线验证状态

```
组件              独立验证          端到端串联
─────────────────────────────────────────────
Language → Plan    Qwen parse 5/5 ✅   ✅ (run_e2e_pid_pipeline.py)
Plan → Execute      2-block 91% ✅      ✅ (mock parser, 6/18)
                     3-block 93% ✅     ✅ (仅 3-block replan 路径)
                     Hanoi 84% ✅      ❌ (TaskRunner grasp 步数过长)
Execute → Monitor   Operator 87% ✅    ✅ (E2E 脚本中集成)
Monitor → Replan    3-block 93% ✅     ✅ (仅 3-block)
Full E2E (Qwen)    —                   ✅ parse 5/5, execute ~30% (SkillLibrary gap)
```

## 九-A、四主张收敛（2026-06-21 导师意见）

论文不再扩功能。收敛为四个窄而硬的主张：

1. **Gate Attribution**：VLA 仅对 reach_pick 有贡献（tight gate 38%, calibrated 100%），放宽 gate 伤害成功率（60%→15%）。不是"VLA 从未控制一步"，而是"VLA 只在最简单的 phase 被信任，且越信任越失败"。
2. **Goal-only VLM + Alignment**：VLM 不做规划，只输出目标；Alignment 6 层硬校验；BFS 保证动作序列合法。
3. **BFS Planning**：BFS 在小状态空间比 handcrafted planner 更正确（能处理拆堆/重排，InternalStackingPlanner 不能）。
4. **Option-level Execution**：长时序需要技能边界、OptionResult、failure taxonomy 和 replan（3-block 0%→93%）。

**定位**：structured long-horizon manipulation under known object poses, with explicit attribution and diagnostic execution.

**冻结原则**：不加功能。不复述 VLA/Hanoi/BFS/Alignment/HDF5/ID/谓词编码器/4-block 所有事。只讲这四个主张。

---

### 9.1 下一步（按优先级）

| # | 事项 | 预计 | 状态 |
|:---:|------|:---:|:---:|
| 1 | Gate sweep 实验（5-6 阈值 × 50ep） | 4-5h GPU | 待做 |
| 2 | Phase intervention 实验 | 半天 | 待做 |
| 3 | 3-block 复现（回退 GainSchedule + 50ep） | 半天 GPU | 待做 |
| 4 | LLM skill planning + symbolic verification | 2-3 天 | 待设计 |

### 9.2 已完成（收敛冻结）

**BFS vs InternalStacking 符号层对比**（纯符号，不需要 MuJoCo）：

```
初始: cube1 on cube3 on cube4 on table  (3 层塔)
目标: cube4 on cube3 on cube1            (反向)

BFS (4 states):                       InternalStackingPlanner:
  1. move_to_table(cube1)  ← 拆堆!       1. move(cube3→cube1)  ← 跳过拆堆!
  2. move_on(cube3, cube1)               2. move(cube4→cube3)
  3. move_on(cube4, cube3)
```

InternalStackingPlanner 假设所有 cube 在桌面可直接移动——不生成 MoveToTable。BFS 在 4 个状态内找到正确的拆堆+重堆路径。

物理执行（4-block/3-block reverse）全部失败——塔操作中 gripper 碰撞导致初始坍塌。执行层限制已通过缩小 cube（全部 3cm）和安全回撤缓解，但根本上需要新的 grasp 策略。与规划正确性正交。

### 9.3 搁置

| 事项 | 原因 |
|------|------|
| 4-block reverse 物理执行 | gripper physics 限制，规划正确性已符号层证明 |
| Slide joint 真抽屉 | MuJoCo 无静摩擦，论文 limitations 中说明 |
| Drawer full env | 3-5 天，Pull Task 85% 已证明 pull 原语 |
| 感知层实时接入 | YOLO+stereo 离线已验证 (Δ=-1.7pp)，论文中 discuss |

### 9.4 已关闭

| 事项 | 结论 |
|------|------|
| VLA Diffusion | gate=0%, 放弃 |
| ID 模型 | 多对一, 不可行 |
| 谓词编码器 | 泄漏, 不需要 |
| Drawer | 3-5天, 暂缓 |

---


> 问答记录 [advisor_qa_2026-06-19.md](advisor_qa_2026-06-19.md)
> 改进方案 [improvement_plan.md](improvement_plan.md)
> 系统架构 [design/controller.md](design/controller.md)
> 泛化方案 [design/generalization.md](design/generalization.md)
> 审计意见 [advisor_review_2026-06-18.md](advisor_review_2026-06-18.md)
> 日志: [6/16](logs/2026-06-16-pid.md) [6/17](logs/2026-06-17.md) [6/18](logs/2026-06-18.md)
> 论文: [paper_draft.md](paper_draft.md)
