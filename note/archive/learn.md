# VLM-NSM：神经符号 VLA 机器人系统 —— 从零入门教程

> 适用对象：对具身智能 / 机器人学习了解较少，希望理解本项目全貌的新人  
> 预计阅读时间：60-90 分钟  
> 配套文档：`note/controller.md`（完整方案与日志），`log.md`（项目入口）

---

## 目录

1. [本项目在做什么（一句话 + 一张图）](#1-本项目在做什么)
2. [核心概念：逐个击破术语障碍](#2-核心概念)
3. [系统架构：四层是怎样协作的](#3-系统架构)
4. [代码导览：230 个文件从哪里读起](#4-代码导览)
5. [实验全流程：从数据到评估](#5-实验全流程)
6. [关键发现：我们知道了什么](#6-关键发现)
7. [动手实践：跑起来](#7-动手实践)
8. [常见问题](#8-常见问题)

---

## 1. 本项目在做什么

### 一句话概括

**让机器人听懂人话，自主完成堆叠方块任务。**

你对着机器人说"把红色方块放到绿色方块上面"，机器人用自己的摄像头观察桌面，用机械臂完成抓取、移动、放置的全过程。如果中途方块掉了，它能自动检测到并重新来过。

### 核心挑战

为什么这很难？因为机器人需要同时解决四个问题：

| 问题 | 难度来源 |
|------|---------|
| **理解语言** | "红的放绿的上面" 这句话有多种说法，要正确解析 |
| **规划动作** | 从当前状态到达成目标，需要多少步？什么顺序？ |
| **视觉驱动** | 摄像头看到的像素，怎么变成机械臂该往哪动？ |
| **错误恢复** | 方块掉了怎么办？放歪了怎么办？需要自动检测和重试 |

### 解决思路：神经 + 符号 = 双系统

本项目采用"双系统"方案，灵感来自人类认知（System 1 直觉快 + System 2 逻辑慢）：

```
人类语言 "把红色方块放到绿色方块上面"
        │
        ▼
┌──────────────────────────────────────────┐
│  System 2 (符号系统 — 慢, 精确, 可验证)   │
│                                          │
│  语言解析 → 目标验证 → 符号规划(BFS搜索)   │
│  产出: "先抓起cube1, 放到cube2上面"       │
│        带前提条件 + 预期效果               │
└──────────────────┬───────────────────────┘
                   │ SkillCommand
                   ▼
┌──────────────────────────────────────────┐
│  System 1 (神经VLA — 快, 从像素到动作)    │
│                                          │
│  摄像头RGB → 深度学习模型 → 机械臂动作     │
│  (ResNet看图片 + Diffusion生成动作序列)    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  监测与恢复层                             │
│                                          │
│  每步检查: 物体在夹爪里吗? 放上去了吗?     │
│  失败了: 识别失败类型 → 自动重试或重规划   │
└──────────────────────────────────────────┘
```

**为什么叫"神经符号 VLA"**：

- **神经 (Neuro)** = 深度学习模型（ResNet、Diffusion UNet），从像素学习
- **符号 (Symbolic)** = 基于规则的逻辑系统（谓词、状态图、搜索），精确可验证
- **VLA** = Vision-Language-Action，视觉-语言-动作，即"看、听懂、会动"

### 实验环境

```
仿真器: robosuite (基于 MuJoCo 物理引擎)
机器人: Panda 机械臂 (7自由度)
任务:   2个方块 (红色4cm + 绿色5cm) 堆叠
摄像头: 2个视角 (agentview 主视角 + frontview 前视角)
GPU:    8 × RTX 4090 (24GB 显存)
```

---

## 2. 核心概念

> 本节用最直白的语言解释项目中出现的所有技术术语。第一次遇到的术语用 **粗体** 标注。

### 2.1 基础概念

**机械臂末端执行器 (EEF, End-Effector)**：机械臂的"手指尖"，我们控制的就是这个点的 (x, y, z) 坐标。

**夹爪 (Gripper)**：装在 EEF 上的抓手，可以张开/闭合来抓取物体。数值范围 [-1, 1]，负数表示张开，正数表示闭合。

**动作 (Action)**：每步发给机器人的指令。本项目用 7 维向量 `[dx, dy, dz, rx, ry, rz, gripper]`，但实际只用前 3 维 (移动增量) 和第 7 维 (夹爪开合)，旋转 (rx, ry, rz) 固定为 0。

**观测 (Observation)**：机器人"看到"和"感觉到"的信息，包括：
- RGB 图像：2 个摄像头的画面，各 128×128 像素
- 本体感觉 (Proprioception)：EEF 当前位置 + 夹爪开度

**回合 (Episode)**：机器人从初始状态开始，执行一系列动作，直到成功（方块堆好）或失败（超时/出错）为止。一次完整的尝试叫一个 episode。

### 2.2 符号层概念

**谓词 (Predicate)**：描述世界状态的布尔语句。例如：
- `on(cube1, cube2)` — "cube1 在 cube2 上面"，是真还是假？
- `holding(cube1)` — "机械臂正抓着 cube1"，是真还是假？
- `clear(cube1)` — "cube1 上面没有东西"，是真还是假？
- `handempty()` — "夹爪是空的"，是真还是假？

**场景图 (SceneGraph)**：所有物体 + 所有谓词的集合。相当于世界状态的快照。

**算子 (Operator)**：一个有名字的动作，带有前提条件和预期效果。例如：
```
MoveOn(cube1, cube2)
  前提条件 (Preconditions): clear(cube1) ∧ clear(cube2) ∧ handempty() ∧ graspable(cube1)
  添加效果 (Add Effects):   on(cube1, cube2) ∧ clear(cube1) ∧ handempty()
  删除效果 (Del Effects):   clear(cube2)
```
意思：要把 cube1 放到 cube2 上，必须满足"cube1 上面没东西、cube2 上面没东西、夹爪空着、cube1 可抓"。完成后，cube1 在 cube2 上面了，cube1 上面清空了，但 cube2 不再"clear"了。

**BFS 规划器**：广度优先搜索 (Breadth-First Search)。从当前状态出发，尝试每一个可能的算子，产生新状态，再尝试......最多搜索 10000 个状态，直到找到一条从初始状态到目标状态的路径。因为只涉及 2 个方块和几个谓词，搜索空间很小，非常快 (< 0.01 秒)。

**Alignment（对齐层）**：语言解析器（Qwen 大模型）的输出可能包含错误或不规范的内容。Alignment 是一组规则，用白名单/类型检查/别名修复等手段修正解析结果，确保能正确生成 PDDL。

### 2.3 神经层概念

**BC (Behavior Cloning，行为克隆)**：最简单的模仿学习——收集大量"状态→正确动作"的数据对，训练神经网络直接从状态预测动作。本项目 M0 模型就是 BC 的一个变体。

**Diffusion Policy（扩散策略）**：一个更进阶的方法。核心思想：
1. 训练时：给正确的动作序列加噪声，让模型学会"去噪"（从噪声中恢复正确动作）
2. 推理时：从纯随机噪声开始，经过多步去噪，生成一个动作序列
3. 好处：能产生多模态的动作（同样的目标可以用不同方式达成），比简单的回归更鲁棒

**Action Chunk（动作块）**：不预测单步动作，而是一次预测未来 H=16 步的动作序列。这让动作更连贯，减少抖动。

**DDPM (Denoising Diffusion Probabilistic Model)**：训练时用 100 步加噪/去噪。**DDIM (Denoising Diffusion Implicit Model)**：推理时只用 16 步加速。

**1D UNet**：处理一维序列（动作序列的每个时间步是一个 4 维向量）的卷积神经网络，通过编码器-解码器结构捕获不同尺度的时序特征。

**ResNet18**：一个经典图像分类网络（18 层残差网络），在 ImageNet 上预训练过，用于从 128×128 RGB 图像提取 256 维特征向量。

### 2.4 Safety Gate（安全门控）

**为什么需要 Gate**：VLA 模型是神经网络，可能输出不合理或危险的动作（比如让机械臂撞桌子）。Gate 是一个安全检查层，在 VLA 输出动作后、实际执行前进行审核。

**Gate 检查哪些**：
1. NaN/Inf 检查：动作不能包含非法数值
2. 动作幅度检查：VLA 动作和 scripted teacher 动作的 XYZ 偏差不能超过阈值
3. 工作空间检查：机械臂不能移出桌面范围
4. 相位语义检查：在"抓取闭合"阶段不能张开夹爪

**Gate acceptance**：VLA 预测动作被 Gate 通过的比例。这是本项目衡量 VLA 质量的核心指标——接受率越高，说明 VLA 动作越接近 teacher 的质量。

### 2.5 Teacher 与 Student

**Teacher（教师策略）**：手工编写的脚本控制器（状态机 + P 控制器），不需要学习，直接根据物体位置计算正确的机械臂动作。成功率为 70%（当前版本）。

**Student（学生策略）**：我们训练的神经网络模型（VLA），目标是学会 teacher 的行为。

**为什么需要 Teacher**：
1. **数据生成**：Teacher 执行任务并录制每一步的数据（RGB + 动作 + 本体感觉），作为训练数据
2. **安全网**：当 VLA 动作被 Gate 拒绝时，使用 Teacher 动作作为 fallback
3. **Baseline 对比**：衡量 VLA 是否真的比手工规则更好

**Phase（相位/阶段）**：Teacher 把一次完整的堆叠任务分解为 6 个子阶段：

```
Phase 0: reach_pick   — 机械臂移动到要抓的物体上方
Phase 1: grasp        — 下降并闭合夹爪
Phase 2: lift         — 抬起物体
Phase 3: reach_place  — 移动到目标位置上方
Phase 4: release      — 张开夹爪释放物体
Phase 5: retreat      — 机械臂撤离
```

### 2.6 评估指标

| 指标 | 含义 | 好/坏标准 |
|------|------|----------|
| Overall success | 任务成功完成的 episode 比例 | 越高越好，teacher ~70% |
| Gate acceptance | VLA 动作被 Gate 接受的比例 | 越高说明 VLA 越像 teacher |
| Phase success | 某个 phase 内 VLA 控制不导致失败 | >90% 视为可行 |
| reach_pick cos | VLA 和 teacher 动作方向的一致性 | >0.6 视为方向可用 |
| VLA-caused failures | VLA 控制导致失败的 episode 数 | 越少越好，<5/50 视为安全 |

### 2.7 模型代号

```
M0:  ResNet18 + MLP (单帧预测, 11.8M 参数)
     — 最简单的基线，证明 RGB→动作可行

M1:  ResNet18 + Conditional 1D UNet (Diffusion Policy, 26.8M 参数)
     — 本项目主力模型，一次预测 16 步动作

B3:  Diffusion-State baseline (只用物体位置，不用图像)
B4:  Diffusion-RGB baseline (用图像，没有符号条件) ← 关键对照
B5:  Diffusion-RGB + Symbolic Condition (本项目方法) ← 主模型
```

---

## 3. 系统架构

### 3.1 四层全景

```
Layer 1: Language → Symbolic           Layer 2: Symbolic → Skill
┌──────────────────────┐              ┌──────────────────────┐
│ NL "红方块放绿的上"   │              │ SymbolicPlan         │
│        ↓              │              │   ↓                  │
│ Qwen Parser (API)     │              │ SkillDecomposer      │
│   → Structured Goal   │              │   → SkillCommand     │
│        ↓              │              │   → Phase FSM        │
│ Alignment (rules)     │              │   → Symbolic Cond    │
│        ↓              │              │     (31-dim vector)  │
│ GoalValidator         │              │                      │
│        ↓              │              │ 对象: 告诉VLA        │
│ PDDL → BFS Planner    │              │ - 抓哪个物体          │
│        ↓              │              │ - 放哪里              │
│ SymbolicPlan          │              │ - 当前是什么阶段       │
│ [MoveOn(c1,c2)]       │              │ - 还剩几步            │
└──────────────────────┘              └──────────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │ SkillCommand + SymbolicCondition
                        ▼
Layer 3: VLA Visuomotor                  Layer 4: Safety & Recovery
┌──────────────────────┐              ┌──────────────────────┐
│ Input:                │              │ Safety Gate           │
│  RGB(128×128)×2       │              │  → 批准/拒绝VLA动作   │
│  Proprio(7-dim)       │     ┌───────→│ Scripted Fallback     │
│  SymbolicCond(42-dim) │     │        │ Predicate Monitor     │
│        ↓              │     │        │  → holding(x)=?       │
│ ResNet18 Encoder      │     │        │  → on(x,y)=?          │
│        ↓              │     │        │ OperatorMonitor       │
│ Diffusion 1D UNet     │─────┘        │  → 算子层失败分类      │
│        ↓              │              │  → recovery策略       │
│ Action Chunk (16×4)   │              │ Replanner             │
│  [dx,dy,dz,gripper]   │              │  → 失败后重新规划      │
└──────────────────────┘              └──────────────────────┘
```

### 3.2 数据流 (一次完整的任务执行)

```
Step 1: 用户说 "把红色方块放到绿色方块上面"

Step 2: Qwen 大模型解析 → {goal: "on(cube1, cube2)", objects: {cube1: red, cube2: green}}

Step 3: Alignment 修正 → 确保 object/relation 在白名单内 → GoalValidator 拒绝非法目标

Step 4: PDDL 生成器 → 动态生成规划问题描述文件

Step 5: BFS 规划器 → 在当前 SceneGraph 上搜索 → 找到 [MoveOn(cube1, cube2)]

Step 6: 算子语义提取 → SkillCommand("pick_place", object=cube1, target=cube2,
                                  relation="on", preconditions=[...], expected_add_effects=[...])

Step 7: SymbolicConditionEncoder → 编码为 42 维向量:
        [pick_onehot, target_onehot, relation_onehot, phase_onehot, colors, sizes, plan_context]

Step 8: 对每个 Phase 循环:
        ┌─ reach_pick:
        │   VLA/Diffusion: RGB → 16步动作序列
        │   Gate 审核每步 → 通过用VLA, 拒绝用scripted fallback
        │   检查: EEF 到达物体上方了吗?
        │
        ├─ grasp:
        │   scripted controller 控制夹爪闭合
        │   (这部分还没接VLA)
        grasp 需要的是基于实时触觉反馈的事件驱动控制，而当前 VLA（基于视觉的 Diffusion Policy）的输 UI 模型设计不适合做这件事。要让 VLA 控制 grasp，需要：增加触觉/力觉输入（不只是 RGB）；改用离散动作空间或混合架构（连续 XYZ + 离散 grasp 决策）
        │
        ├─ lift → reach_place → release → retreat
        │   (类似流程)
        │
        └─ Predicate Monitor 检查:
            on(cube1, cube2)=True? holding(cube1)=False? clear(cube1)=True?
            如果失败 → OperatorMonitor 分类失败原因 → 触发 recovery
_OVERRIDE_PHASES = {"reach_pick"}  # 只有 reach_pick 由 VLA 控制

def s1_step(raw_action):
    if phase_name in _OVERRIDE_PHASES:
        # → VLA 预测动作，Gate 审核，通过用 VLA / 拒绝用 scripted
    else:
        # → 直接透传原始 scripted 动作，不做任何修改
        return _orig_step(raw)
各 phase 的控制方式：


Phase 0: reach_pick   →  VLA (M1 Diffusion) + Safety Gate
Phase 1: grasp        →  scripted (grasp_state_machine)
Phase 2: lift         →  scripted (P-controller)
Phase 3: reach_place  →  scripted (S1模式) / VLA (S2模式)
Phase 4: release      →  scripted (place_state_machine)
Phase 5: retreat      →  scripted (P-controller)
S2 模式（--override_phases "reach_pick,reach_place"）下 reach_place 也切到 VLA，但跑出来只有 4% 成功率——transport cos=0.549 太差。所以当前最优配置就是 S1：VLA 只控 reach_pick，其余全是 scripted。
Step 9: Episode 完成 → 记录成功/失败 + 失败类型 + per-step trace
```

### 3.3 两个系统的交互

```
System 2 (符号)                          System 1 (神经/脚本)
─────────────                            ────────────────────
跑一次（规划）                            每个Phase循环执行
↓                                        ↓
产出 SymbolicPlan                       产出 action
↓                                        ↓
SkillCommand ─────────────────→         执行 action
                                        ↓
                            ←── failure_predicates (仅失败时回调)
                                        ↓
Replan ─────────────────────→          重试
```

**关键设计原则**：
1. System 2 只在开始时运行一次，失败时才重新调用
2. System 1 的 scripted 和 VLA 共享同一个 SkillCommand 接口——可以无缝替换
3. OperatorMonitor 在 System 1 执行后检查算子效果，实现"符号验证 + 神经执行"的闭环

---

## 4. 代码导览

> 代码在 `src/vlm_nsm/`，共约 50,000 行。按"你想理解什么"来导航。

### 4.0 目录速查

```
src/vlm_nsm/
├── symbolic/          符号层: 谓词, 规划器, 状态图
├── skills/            技能层: SkillCommand, 符号→技能转换
├── neuro_symbolic/    神经符号桥接: 条件编码, 安全门控, 算子监测
├── vla/               VLA模型: 视觉编码器, Diffusion UNet, 数据集, 训练
├── embodied/          具身控制: 抓取/放置状态机, 故障分类, 执行器
├── executors/         环境包装器: robosuite接口, 脚本执行器
├── execution/         技能执行器: ACT-lite, scripted skill executor
├── perception/        视觉感知: YOLO检测, 相机标定, 双目三角测量, 场景谓词
├── language/          语言接口: Qwen在线解析
├── experiments/       实验脚本(70+): 数据采集, 训练, 评估, 审计, 诊断
├── training/          训练脚本: Stage 20 MLP/RNN/LSTM
├── core/              核心工具: 对齐, 目标解析, PDDL生成
├── observation/       观测源: Oracle(真值), 感知, 噪声注入
├── data/              数据处理: 数据集manifest, phase labeling
├── grounding/         视觉符号接地: 指称表达式, YOLO→符号绑定
├── planners/          外部规划器: Metric-FF, Fast Downward, 河内塔
└── rl/                强化学习: PPO, stack RL env (WIP)
```

### 4.1 理解 "机器人怎么规划动作" → 读 `symbolic/`

**阅读顺序**：
1. [`predicates.py`](src/vlm_nsm/symbolic/predicates.py) (170行) — 谓词定义。什么是 `on(A,B)`、`holding(X)`，以及 `SceneGraph` 如何存储世界状态
2. [`operators.py`](src/vlm_nsm/symbolic/operators.py) (98行) — 算子定义。`MoveOn` 的前提条件和效果
3. [`symbolic_planner.py`](src/vlm_nsm/symbolic/symbolic_planner.py) (86行) — BFS 规划器。从初始状态搜索到目标状态的路径
4. [`goals.py`](src/vlm_nsm/symbolic/goals.py) (48行) — 目标定义
5. [`goal_validator.py`](src/vlm_nsm/symbolic/goal_validator.py) (82行) — 拒绝非法目标

**核心数据结构**：
```python
# 一个谓词
Predicate(name="on", args=("cube1", "cube2"), confidence=1.0)

# 一个场景图
SceneGraph(
    objects={"cube1": ObjectSymbol("cube1", "red", "cube", 0.04),
             "cube2": ObjectSymbol("cube2", "green", "cube", 0.05)},
    predicates={on("cube1","cube2"), clear("cube1"), handempty()}
)
```

### 4.2 理解 "VLA 怎么从像素预测动作" → 读 `vla/`

**阅读顺序**：
1. [`vla/data/vla_dataset.py`](src/vlm_nsm/vla/data/vla_dataset.py) (607行) — 数据集。HDF5 → torch Dataset，怎么切窗口、归一化动作、构造符号条件
2. [`vla/models/vision_encoder.py`](src/vlm_nsm/vla/models/vision_encoder.py) (182行) — 视觉编码器。ResNet18 处理 2 视角图像
3. [`vla/models/diffusion_unet.py`](src/vlm_nsm/vla/models/diffusion_unet.py) (171行) — 1D UNet。3 层编码器-解码器 + FiLM 条件调制
4. [`vla/models/diffusion_policy.py`](src/vlm_nsm/vla/models/diffusion_policy.py) (236行) — 顶层 Diffusion Policy

**核心数据流（训练时）**：
```
HDF5 episode
  ↓ VLADataset.__getitem__()
(agentview: 3×128×128, frontview: 3×128×128, proprio: 7, symbolic_cond: 42, action_chunk: 16×4)
  ↓ DiffusionPolicy.forward()
ResNet → 256×2 + proprio_MLP(7→64) + symbolic_MLP(42→128) → cond: 704-dim
  ↓
Conditional 1D UNet: noise_action(16×4) + cond + timestep → predicted_noise(16×4)
  ↓
Loss = MSE(predicted_noise, actual_noise)
```

### 4.3 理解 "机器人怎么执行抓取和放置" → 读 `embodied/`

**阅读顺序**：
1. [`grasp_state_machine.py`](src/vlm_nsm/embodied/grasp_state_machine.py) (424行) — 抓取状态机：APPROACH → DESCEND → MICRO_ALIGN → CLOSE → LIFT
2. [`place_state_machine.py`](src/vlm_nsm/embodied/place_state_machine.py) (526行) — 放置状态机：PREPLACE → XY_ALIGN → DESCEND → RELEASE → HOLD → RETREAT → VERIFY
3. [`failure_taxonomy.py`](src/vlm_nsm/embodied/failure_taxonomy.py) (217行) — 18 种失败类型标准分类

**核心逻辑（抓取接触检测）**：
```python
# 夹爪闭合后判断是否接触到了物体
contact = (gripper_was_open and          # 之前张开
           gripper_not_fully_closed and   # 没完全闭合（中间有物体）
           gripper_closed_significantly)  # 闭合了至少 20%
# 如果完全闭合了 → nocontact（没抓到东西）
# 如果部分闭合了 → contact（抓到东西了）
```

### 4.4 理解 "VLA 和符号怎么桥接" → 读 `neuro_symbolic/`

**阅读顺序**：
1. [`symbolic_condition.py`](src/vlm_nsm/neuro_symbolic/symbolic_condition.py) (276行) — 将 SkillCommand 编码为 42 维向量
2. [`safety_gate.py`](src/vlm_nsm/neuro_symbolic/safety_gate.py) (320行) — 5 层安全门控
3. [`operator_monitor.py`](src/vlm_nsm/neuro_symbolic/operator_monitor.py) (291行) — **本项目最重要发现之一**：将低层失败标签映射为算子层失败

**OperatorMonitor 的核心逻辑**：
```python
# 输入: "preplace" (低层标签: 放置前阶段失败)
# 输出: "PlaceOnFailed.NOT_HOLDING" (算子层: 放置失败因为没抓着物体)

def classify_failure(self, command, low_level_failure, effect_results):
    if "preplace" in low_level_failure:
        if not holding_satisfied(effect_results):
            return "PlaceOnFailed.NOT_HOLDING"  # 物体运输中掉了！
        else:
            return "PlaceOnFailed.XY_MISALIGN"   # 位置没对准
```

### 4.5 理解 "语言怎么变成符号目标" → 读 `core/` + `language/`

**核心流程**：
```
"把红色方块放到绿色方块上面"
  → Qwen2.5-32B API → {"action": "move_on", "object": "red_cube", "target": "green_cube"}
  → Alignment: red_cube→cube1, green_cube→cube2
  → Goal: [on("cube1", "cube2")]
  → GoalValidator: 检查不包含 self-reference 或 cycle
  → PDDL Generator → BFS Planner → [MoveOn("cube1","cube2")]
```

---

## 5. 实验全流程

### 5.1 数据采集

```bash
# 用 teacher controller 执行 200 个 episode，录制 HDF5 数据
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python \
  -m vlm_nsm.experiments.collect_vla_demos \
  --output_dir generated/datasets/vla_demos/d1_small_200ep \
  --num_episodes 200 --seed 3000 --device cuda:0
```

**产出**：
```
d1_small_200ep/
├── episodes/
│   ├── ep_000000.hdf5   ← 每个 episode 一个文件 (~90MB)
│   ├── ep_000001.hdf5
│   └── ...
├── manifest.jsonl        ← 每行: {episode_path, success, failure_type, ...}
└── stats.json            ← {action_mean: [4], action_std: [4]}
```

**HDF5 内部结构**：
```
/episode_id              str     "ep_000000"
/success                 bool    True/False
/failure_type            str     "preplace" / "nocontact" / ...
/observations/agentview  uint8   (T, 128, 128, 3)
/observations/frontview  uint8   (T, 128, 128, 3)
/actions_4d              float32 (T, 4)    [dx, dy, dz, gripper]
/proprio/eef_pos         float32 (T, 3)    [x, y, z]
/proprio/gripper_open    float32 (T, 1)
/symbolic/phase_id       int16   (T,)      0-5
/symbolic/pick_object_id int16   (T,)      1=cube1, 2=cube2
/symbolic/target_object_id int16 (T,)      1=cube1, 2=cube2, 0=table
/symbolic/relation_id    int16   (T,)      0=on, 1=table, 2=none
```

### 5.2 模型训练

```bash
# M1 (Diffusion Policy)
CUDA_VISIBLE_DEVICES=0 nsm/bin/python -u \
  -m vlm_nsm.vla.training.train_m1_diffusion \
  --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
  --output_dir generated/results/vla/vla_m1_d1_small \
  --epochs 30 --batch_size 16 --device cuda:0 --pretrained
```

### 5.3 模型评估

```bash
# S1 Phase Override（闭环评估）: VLA 控制 reach_pick，其他 scripted
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python \
  -m vlm_nsm.experiments.eval_s1_phase_override \
  --checkpoint generated/results/vla/vla_m1_real_sym/best.pt \
  --num_episodes 50 --seed 9000 --device cuda:0 \
  --output_dir generated/results/vla/my_eval

# 查看结果
nsm/bin/python -c "
import json
r = json.load(open('generated/results/vla/my_eval/report.json'))
for k,v in r.items(): print(f'{k}: {v}')
"
```

### 5.4 B4 vs B5 对照实验（论文核心）

```bash
# B4: 无符号条件 (clean baseline)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python \
  -m vlm_nsm.experiments.eval_s1_phase_override \
  --checkpoint generated/results/vla/vla_m1_b4_zero_sym/best.pt \
  --num_episodes 50 --seed 9000 --device cuda:0 --no_symbolic \
  --output_dir generated/results/vla/b4_eval

# B5: 有符号条件 (ours)  
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl nsm/bin/python \
  -m vlm_nsm.experiments.eval_s1_phase_override \
  --checkpoint generated/results/vla/vla_m1_real_sym/best.pt \
  --num_episodes 50 --seed 9000 --device cuda:0 \
  --output_dir generated/results/vla/b5_eval
```

### 5.5 运行测试

```bash
nsm/bin/python -m pytest tests/ -x --tb=short -q
```

---

## 6. 关键发现

### 发现 1：OperatorMonitor 揭示 preplace 失败根因是运输中滑脱

50ep Operator Audit 发现：**22/22 失败伴随 `holding(x)=False`**。preplace=13 全部映射为 `PlaceOnFailed.NOT_HOLDING`。物体在运输中滑落，不是放置精度问题。修复方向从"改进放置精度"转向"保持运输中不掉落"，Teacher 56%→70%。

### 发现 2：符号条件提升 VLA 动作对齐度 +22pp（论文核心 ablation）

双 seed 交叉验证（各 50ep，tight gate=0.025）：

| | B4 seed 9000 | B4 seed 12345 | B5 seed 9000 | B5 seed 12345 |
|---|---|---|---|---|
| Gate acceptance | 27.3% | 29.0% | 50.9% | 49.8% |
| reach_pick phase | 100% | 100% | 100% | 100% |

**B5 gate − B4 gate = +22.2pp，跨 seed 仅 ±1pp 波动。** 结构化任务语义确实帮助视觉策略做出更合理的动作。

### 发现 3：VLA 在 reach_pick 阶段 100% 安全可控

所有 S1 闭环实验中 reach_pick phase success = 100%。VLA-caused failures ≤ 5/50（tight gate）。但 transport（reach_place）阶段 cos=0.549，还不能安全控制。

### 发现 4：数据 Scaling 在 200ep 后出现平台

100→200ep: 4x 改善。200→500ep: 退化。**Teacher 质量 (~61%) 是硬瓶颈**。

### 发现 5：Gate 是系统的关键安全组件

Tight gate: B4 overall=64%, B5=62%。Permissive gate: B4=6%, B5=15%。**没有 Gate，VLA 无法独立完成任务。**

---

## 8. 技术深度问答（汇报备用）

### Q1: Bridge 1 的 42 维符号条件向量包含什么？每部分起什么作用？

符号条件向量将规划器的"算子语义"翻译为 VLA 能理解的数字信号。具体构成（n=2 个物体时 31 维实际 + 11 维填充 = 42 维）：

| 字段 | 维度 | 编码内容 | 作用 |
|------|:---:|------|------|
| `pick_object_onehot` | 2 | [1,0]=cube1, [0,1]=cube2 | **告诉 VLA 抓哪个物体**——这是 B4 不知道的关键信息 |
| `target_object_onehot` | 3 | +1 维给 table | **告诉 VLA 放哪里**——目标物体或桌面 |
| `relation_onehot` | 3 | on / table / none | **告诉 VLA 操作类型**——堆叠还是放置到桌面 |
| `phase_onehot` | 10 | 当前执行阶段 | **告诉 VLA 当前在做什么**——reach_pick/grasp/lift 等 |
| `action_type_onehot` | 3 | move_on / move_to_table / unknown | **告诉 VLA 动作语义** |
| `pick_color_onehot` | 3 | red / green / blue | **物体的视觉属性**——帮助 VLA 在图像中定位 |
| `target_color_onehot` | 3 | red / green / blue | **目标的视觉属性** |
| `pick_size` | 1 | 0.04/0.06≈0.67（4cm 红） | **物体尺寸**——影响夹爪开合度 |
| `target_size` | 1 | 0.05/0.06≈0.83（5cm 绿） | **目标尺寸** |
| `plan_step_index` | 1 | 0.0（第 1 步） | **告诉 VLA 计划进度** |
| `remaining_steps` | 1 | 0.0（仅剩当前步） | **告诉 VLA 还剩几步** |
| 零填充 | 11 | 全零 | 为 3+ 物体扩展预留 |

**为什么 31 维填到 42 维？** 模型输入维度需要在训练前固定。31 是 2-block 的实际维度，42 是预留了 3-block 扩展空间的模型维度。填充到固定维度是深度学习中的常见做法。

**核心价值**：B4（全零向量）不知道"抓谁、放哪、什么阶段"，只能从图像中猜测。B5（真实编码）有明确的语义引导。这解释了为什么 B5 的 teacher alignment（cos_sim）比 B4 高 17%。

---

### Q2: 两个视角的相机设计是否合适？ResNet18 是什么？512 维特征包含什么？

**相机设计**：

| 参数 | 值 | 理由 |
|------|:---:|------|
| 视角数 | 2（agentview + frontview） | 单视角缺少深度信息，2 视角可通过立体视差推断 3D 位置 |
| 分辨率 | 128×128 | 在精度和推理速度之间折中（512×512 更好但慢 16 倍） |
| Agentview | 机械臂肩部上方 | 主工作视角，类似"机器人眼睛" |
| Frontview | 正前方 | 补充深度信息，帮助判断物体前后关系 |

**是否合适**：对于 2-block 堆叠，当前配置基本够用（reach_pick cos=0.88 证明单视角就能覆盖）。但对于更复杂的 3D 操作（如 transport），128×128 的分辨率可能不足以提供精确的深度信息——模型在长距离运输时方向一致性差（cos=0.55）。改进方向：提高分辨率到 256×256 或增加第三视角（top-down view）。

**ResNet18 是什么**：

ResNet18 是一个 18 层的残差卷积神经网络（2016 年何恺明提出），在 ImageNet（1000 类、120 万张图片）上预训练过。其核心创新是"残差连接"（skip connection）——每层的输出 = 输入 + 卷积结果，解决了深层网络的梯度消失问题。

```
ResNet18 结构（简化）：
输入: 3×128×128 RGB 图像
  → Conv 7×7 + MaxPool → 64×32×32
  → ResBlock×2 (64通道) → 64×32×32      ← 残差: 输入直接加到输出
  → ResBlock×2 (128通道) → 128×16×16
  → ResBlock×2 (256通道) → 256×8×8
  → ResBlock×2 (512通道) → 512×4×4
  → AvgPool → 512维向量
  → fc 层 → 256维特征（本项目替换了原始的 1000 类分类头）
```

**为什么用预训练 ResNet18？** ImageNet 预训练让 ResNet18 学会了通用的视觉特征（边缘、纹理、形状、物体部件）。我们可以用较少的机器人数据（200 episodes）做微调，而不需要从零训练视觉表征。这就是"迁移学习"。

**512 维特征包含什么**：

需要注意：每个视角经过 ResNet18 输出 **256 维**特征（我们替换了最后的 fc 层从 1000→256）。两个视角拼接后 = **512 维**视觉特征。

这些特征是高层次的**语义视觉表征**：
- 物体的位置和姿态（隐含在特征向量中）
- 物体的类别和颜色（从预训练中学到的视觉概念）
- 场景的空间结构（物体之间的相对关系）
- 但**不是**显式的 bounding box 或 (x,y,z) 坐标——那是符号层的职责

最终的条件向量 = 512（视觉）+ 64（proprioception MLP）+ 128（符号条件 MLP）= **704 维**。

---

### Q3: Conditional 1D UNet 扩散策略是什么？128 维条件如何获得？16 步动作序列是什么意思？

**扩散策略（Diffusion Policy）的核心思想**：

传统 BC（行为克隆）直接预测动作：`action = model(observation)`。问题是：
- 只能输出单一动作，无法表达"多种可能"（多模态）
- 在模糊场景下（如物体被遮挡），平均多个可能性会导致动作抖动

扩散策略用生成式方法解决这个问题：

```
训练时：                                      推理时：
干净动作 → 加噪(100步) → 纯噪声                 纯噪声 → 去噪(DDIM 16步) → 干净动作
                ↑                                          ↑
          UNet学习预测噪声                     UNet根据观察条件去噪
```

**为什么叫 1D UNet？** 因为它在**时间维度**（16 步动作序列）上做 1D 卷积，通道维度是动作的 4 个分量 [dx, dy, dz, gripper]。

```
输入: (B, 4, 16)   ← 4 个动作分量 × 16 步时间
  → Encoder: 4→256→512→1024 (每层 Conv1d kernel=5)
  → Bottleneck: 1024→1024
  → Decoder: 1024→512→256 (带 skip connection)
  → Final: 256→4
输出: (B, 4, 16)   ← 预测的噪声（训练）或去噪后的动作（推理）
```

**128 维条件怎么来的？**

你说的"128 维"应该是 `symbolic_mlp` 的输出（42 维符号条件 → Linear → 128 维）。完整的条件向量流程是：

```
条件向量组装（encode_obs 函数）：
  Vision:   ResNet18→256维 × 2视角 = 512维
  Proprio:  7维[ee_xyz+gripper+zeros] → MLP → 64维
  Symbolic: 42维 → MLP → 128维
  ─────────────────────────────────────────
  拼接: 512 + 64 + 128 = 704 维条件向量

UNet 条件注入（FiLM 机制）：
  704维条件 → Linear → 128维 → 拼接 timestep嵌入(128维) → 256维
  → FiLM: 对每层 Conv1d 输出做 scale + shift 调制
```

FiLM（Feature-wise Linear Modulation）不是简单拼接，而是让条件**调制**每一层特征：
```python
x = Conv1d(x)           # 时间维卷积
scale, shift = Linear(cond)  # 条件 → 缩放因子 + 偏移量
x = Norm(x) * (1 + scale) + shift  # FiLM 调制
```

**16 步动作序列是什么意思？**

Diffusion Policy 不预测单步动作，而是一次预测**未来 16 步的动作序列**（action chunk）：

```
单步 BC:  observation → [dx₀, dy₀, dz₀, grip₀]   ← 只预测当前一步
Diffusion: observation → [[dx₀,...,grip₀],          ← 预测未来 16 步
                          [dx₁,...,grip₁],
                          ...
                          [dx₁₅,...,grip₁₅]]
```

**为什么预测 16 步？**：
1. **动作连贯性**：16 步序列比单步更平滑，减少抖动
2. **克服观测延迟**：机器人控制是 10-20Hz，16 步覆盖约 1-2 秒
3. **隐式闭环**：执行时每步重新推理（receding horizon），只用第 1 步，不是开环执行 16 步

---

### Q4: 扩散模型能否改用其他策略？如 Action Chunk 等？

可以，而且 Action Chunk 本身就是 Diffusion Policy 的一部分。以下是几种替代方案及对比：

| 策略 | 代表工作 | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|---------|
| **MLP BC** | 本项目 M0 | 极简，训练快（15min） | 无法表达多模态，抖动大 | Pipeline 验证 |
| **LSTM/RNN BC** | Stage 20（已弃用） | 时序建模 | MSE 不能预测闭环 | 不适合 |
| **Diffusion Policy** | Chi et al. 2023 + 本项目 M1 | 多模态，平滑，最先进 | 推理较慢（16 步去噪） | **当前最佳** |
| **ACT (Action Chunk Transformer)** | Zhao et al. 2023 | CVAE+Transformer，编码器-解码器 | 需要更大的模型和数据 | 多任务、复杂场景 |
| **DDPM with DDIM** | 本项目 | 训练 100 步噪声，推理 16 步加速 | DDIM 牺牲一些质量换速度 | **当前方案** |
| **Consistency Models** | Song et al. 2023 | 1-2 步推理，极快 | 较新，需要更多验证 | 未来方向 |
| **Flow Matching** | Lipman et al. 2023 | 训练更稳定 | 实现复杂 | 未来方向 |

**本项目为什么选 Diffusion Policy？**
1. 在 small data regime（200ep）下表现良好
2. 1D UNet 轻量（26.8M 参数），适合 RTX 4090
3. DDIM 16 步推理满足实时控制需求（~50ms/step）
4. FiLM 条件机制天然支持多模态条件注入（视觉 + proprio + 符号）

**可能的改进方向**：
- **Consistency Distillation**：将 DDIM 从 16 步压缩到 1-2 步
- **ACT-style Transformer**：替换 1D UNet 为 Transformer 编码器-解码器，可能更好处理长距离依赖（帮助改进 transport cos=0.55 的问题）
- **Multi-noise 训练**：同时预测多种噪声水平，增强鲁棒性

---

### Q5: 安全门控（Safety Gate）的设计目的是什么？是否必要？

**设计目的**：在 VLA（不可靠的神经网络）和 scripted controller（可靠但保守的手工规则）之间进行智能仲裁。

**5 层检查**：

| 层 | 检查内容 | 拒绝原因 |
|:---:|------|------|
| ① | NaN/Inf 检查 | 模型输出异常值 |
| ② | 置信度检查 | VLA 不确定性高（当前默认禁用） |
| ③ | XYZ 偏差检查 | 与 teacher 动作偏差 > 阈值 |
| ④ | 工作空间检查 | 预测 EEF 位置超出桌面范围 |
| ⑤ | 阶段语义检查 | 抓取阶段不能张开夹爪，反之亦然 |

**是否必要？是的，绝对必要。** 证据：

```
Gate 配置              B4 成功率    B5 成功率    说明
────────────────────────────────────────────────────
Tight gate (0.025)      60-70%      52-62%     大部分步用 scripted，成功率高
Calibrated gate         6%          15%         几乎所有步用 VLA，接近全败
无 Gate                 推测接近 0%              纯 VLA 无法完成任务
```

**Gate 的故事不是"限制 VLA 的枷锁"，而是"实现安全自主的机制"：**

- **类比自动驾驶**：纯端到端（无安全层）→ 可能输出危险动作。有 safety monitor → 允许 learned policy 在安全边界内自由探索。
- **论文叙事**：Gate 不是一个需要被移除的拐杖，而是神经符号系统的核心组件——它精确体现了"符号约束神经"的设计哲学。
- **Gate calibration 的意义**：通过数据分析确定每个阶段的合理阈值（如 grasp 阶段的偏差容忍度 > reach_pick 阶段），实现对 VLA 能力的**细粒度信任分配**。

---

### Q6: P-controller + 状态机是什么？脚本控制为什么上限提不上去？

**P-controller（比例控制器）**：

最简单的反馈控制器。每一步计算目标位置与当前位置的误差，乘以比例系数得到控制量：

```python
# 每步执行（例如 grasp_state_machine.py 第 139-141 行）：
dx = clip(kp * (target_x - current_x), -max_delta, +max_delta)
dy = clip(kp * (target_y - current_y), -max_delta, +max_delta)
dz = clip(kp * (target_z - current_z), -max_delta, +max_delta)
action = [dx, dy, dz, gripper_command]
```

- `kp=0.05`：比例增益。越小越平滑但越慢，越大越快但可能震荡
- `max_delta=0.02`：单步最大位移（2cm），防止动作过大
- **没有积分项（I）和微分项（D）**——所以叫 P-controller 而非 PID

**状态机（State Machine）**：

将抓取/放置任务分解为有序的阶段序列，每个阶段有自己的目标和退出条件：

```
抓取状态机：                         放置状态机：
  APPROACH → 移动到物体上方 8cm        PREPLACE → 移动到目标上方 15cm
  DESCEND → 下降到物体表面+5mm         MICRO_ALIGN → 精细 XY 对齐
  MICRO_ALIGN → 精细对齐 (±4mm步长)    DESCEND → 慢速下降
  CLOSE → 闭合夹爪（检测接触）         RELEASE → 打开夹爪
  LIFT → 抬起物体（检测 Z 变化）       HOLD → 稳定等待
                                       RETREAT → 撤离
                                       VERIFY → 检查 on(x,y)?
```

**为什么上限提不上去（~80%）？**

| 失败模式 | 占比 | 原因 | 为何 P-controller 无法解决 |
|---------|:---:|------|--------------------------|
| 物体运输中滑落 | 主要 | 夹爪摩擦力不足 | 开环控制，无法感知物体滑动 |
| 放置碰撞 | 较多 | 放置路径碰到已堆叠物体 | 无碰撞检测，无避障规划 |
| 薄物体抓取失败 | 中等 | 接触检测仅靠 open_value | 无力/触觉传感器 |
| Horizon 超时 | 少量 | 某些阶段收敛慢 | 步数预算固定，无法动态调整 |
| 堆叠倒塌 | 3-cube 特有 | 后续操作扰动已堆叠结构 | 单步规划，无全局碰撞感知 |

**根本原因**：P-controller + 状态机是**反应式**的（每步只看当前位置和目标），缺乏：
1. **力/触觉感知**：无法感知物体是否在滑动
2. **碰撞预测**：无法预判路径是否会碰到其他物体
3. **全局优化**：每步贪心最优 ≠ 全局最优轨迹
4. **适应能力**：kp、max_delta 等参数针对特定物体调优，换物体尺寸就失效

**为什么 VLA 可能突破这个上限？** VLA 从数据中学习，可以隐式建模物体动力学、摩擦力、碰撞模式——只要训练数据足够多样。但当前 200ep 的数据量远不足以学到这些。

---

### Q7: BFS 只在重规划中起作用吗？为什么不用 Metric-FF 等更强大的符号规划器？

**BFS 的使用位置**：

BFS `SymbolicPlanner` 设计用于**初始规划**——从当前 SceneGraph 搜索到目标的最短路径。但在实际生产代码中（`run_embodied_stacking_robosuite.py:1027`），使用的是 `InternalStackingPlanner`（一个 20 行的规则式规划器，直接将 target_stack 相邻元素配对），而非 BFS。

重规划（replan）功能使用的是同一个 BFS planner，但输入是**当前符号状态**（从 OperatorMonitor 获取）而非初始状态。

**为什么不用 Metric-FF / Fast Downward？**

| 规划器 | 能力 | 为什么本项目不用 |
|------|------|----------------|
| **InternalStackingPlanner**（当前实际使用） | 仅处理固定顺序堆叠 | ⚠️ 最简单，但连逆序拆堆都处理不了 |
| **BFS SymbolicPlanner**（已实现，已修复接口） | 搜索最优路径，处理任意初始状态 | ✅ 适合 2-3 block 的小状态空间 |
| **Metric-FF** | 启发式搜索，支持复杂 PDDL domain | ❌ 过度设计——2-block 只需搜索几十个状态 |
| **Fast Downward** | 最强大的经典规划器 | ❌ 同上，且需要额外安装和集成 |

**BFS 对当前任务已经足够**：

- 2-block 场景：状态空间约 20-30 个状态，BFS 在 <1ms 内完成
- 3-block 场景：状态空间约 100-200 个状态，BFS 在 <10ms 内完成
- 4-block 场景：状态空间约 500-2000 个状态，BFS 仍可在 <1s 内完成

**真正的问题不是"BFS 太弱"，而是"BFS planner 根本没被接入生产流水线"**——这个问题已在 2026-06-15 的 C3 修复中解决。

**如果需要扩展到更复杂的任务**（如多机器人协作、复杂物体交互），Metric-FF 或 Fast Downward 才有意义。对当前的 2-3 block 堆叠任务，BFS 是最佳选择——简单、可控、无外部依赖。

---

### Q8: 补充 — Scripted Controller 的角色定位

前面 Q5-Q7 共同揭示了一个关键设计哲学问题：**为什么系统中有这么多"非学习"组件？**

```
Scripted Controller 的六个角色：
┌────────────────────────────────────────────────────┐
│ ① Teacher     → 生成 demonstration 训练数据        │
│ ② Safety Prior → 为 Gate 提供动作范围参考          │
│ ③ Fallback    → VLA 动作被拒绝时兜底               │
│ ④ Recovery    → 失败后执行 regrasp/re-place 原语   │
│ ⑤ Baseline    → 量化 VLA 的增量贡献                │
│ ⑥ Skill Component → 高可靠性阶段（grasp/release）  │
│                    保持 scripted，不强行神经化      │
└────────────────────────────────────────────────────┘
```

这六个角色意味着：**Scripted controller 不是 VLA 的竞争对手，而是 VLA 的基础设施。** 在一个安全的机器人系统中，learned policy 和 rule-based safety 是互补的——这正是"神经符号"的核心含义。

---

## 9. 汇报要点速查（2026-06-15 更新）

> 给导师/审稿人汇报时可以直接使用的关键数字和论点。

### 一句话总结

我们构建了一个神经符号 VLA 系统，通过三层桥接（符号条件编码、安全门控、算子监测）将符号规划与视觉运动策略耦合。核心实验证明：**符号条件不是"改善"了 VLA——它是 VLA 能够工作的必要条件。**

### 核心数字

| 指标 | B4（无符号） | B5（有符号） | 意义 |
|------|:---:|:---:|------|
| Gate 接受率 | **0.0%** | **47.0%** | VLA 贡献：0% → 47% |
| Task Success | 63.5% | 61.0% | 相似（scripted 兜底） |
| reach_pick cos | N/A | 0.88 | VLA 方向准确 |
| reach_place cos | 0.55 | 0.55 | 运输阶段瓶颈 |

### 三层桥接叙事

1. **Symbolic Conditioning**：42 维向量编码算子语义 → B4(0%) vs B5(47%) gate 差异
2. **Safety Gate**：5 层检查 → 无 gate 时 B5=15%（100ep）
3. **OperatorMonitor**：preplace→NOT_HOLDING 诊断 → 93% 3-cube 恢复率

### Scripted 六个角色

Teacher · Safety Prior · Fallback · Recovery Oracle · Baseline · Skill Component

→ Scripted 不是 VLA 的竞争对手，而是 VLA 的基础设施

### 代码修复状态

- 6 个 CRITICAL 代码问题已修复
- B4 重新训练（验证过的 `--zero_symbolic`）
- 1549 tests passed（移除 131 废弃测试）
- B4.6 shuffled symbolic 消融训练中


---

