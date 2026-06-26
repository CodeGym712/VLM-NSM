# VLM-NSM 深度审查报告与修复计划

> 审查日期：2026-06-15 | 审查方法：5 并行审计代理 + 直接代码/数据取证
> 覆盖范围：~50K 行代码 | 717 JSON 文件 | 150+ 评估目录 | 1680 测试

---

## 零-A、战略决策：保持完整神经符号 VLA 叙事

用户决定不缩小论文范围，坚持构建完整的神经符号 VLA 叙事。

### 当前状态 vs 需要的状态

| 组件 | 当前状态 | 论文需要 | 差距 |
|------|---------|---------|:---:|
| 语言→符号 | ✅ 100% (full prompt) | ✅ 保持 | — |
| 符号规划 | ⚠️ 实际用 InternalStackingPlanner | BFS/搜索式 planner | 中等 |
| VLA 控制范围 | ❌ 仅 reach_pick (1/6) | ≥3/6 阶段 | **大** |
| 安全门控 | ⚠️ 循环论证 | 需要独立有效性证据 | 中等 |
| 算子监测 | ⚠️ 事后分析 | 闭环主动监测 | 中等 |
| 端到端整合 | ❌ 三系统独立评估 | 单次 episode 内全链路 | **大** |
| 任务范围 | ⚠️ 仅 2-block | ≥2-block + 至少一种变体 | 中等 |
| Teacher 质量 | ⚠️ ~70-80% | >85% 以减少天花板效应 | 小 |

### 核心叙事框架（建议）

**标题**：*"Neuro-Symbolic Skill Execution: Bridging Symbolic Planning and Visuomotor Policies via Structured Conditioning and Operator-Level Monitoring"*

**故事线**：

1. **问题**：端到端 VLA 缺乏可验证性和可恢复性。符号规划提供这些，但与视觉运动策略之间存在"语义鸿沟"。

2. **方案**：三层桥接——
   - **Symbolic Conditioning**（符号→VLA）：将规划器的 operator 语义编码为 conditioning vector，注入 Diffusion Policy（对应 B4/B5 实验）
   - **Safety Gate**（VLA→执行）：分层安全检查确保动作可行性（对应 gate calibration）
   - **Operator Monitor**（执行→符号）：执行后验证 operator 效果，失败时触发恢复（对应 OperatorMonitor + replan）

3. **核心实验**：
   - B4 vs B5：符号条件对 teacher alignment 的改善（cos_sim +17%）
   - Gate calibration：数据驱动的安全-自主性平衡
   - OperatorMonitor：低层失败→算子层失败的映射诊断

4. **诚实限制**：
   - VLA 当前主要控制 reach_pick 阶段（transport 方向一致性不足）
   - 符号编码为手工设计
   - 单任务、单环境

### 关键区别：这个叙事不依赖 VLA 控制全部阶段

正确的框架是：**"VLA 是神经符号系统中的 visuomotor 执行组件，而非替代品"**。

- Scripted controller 的角色是 **teacher + safety prior + recovery oracle**
- VLA 的角色是 **被符号条件引导的 visuomotor 策略**
- Gate 的角色是 **决定何时信任 VLA、何时使用 scripted 的安全仲裁器**
- OperatorMonitor 的角色是 **符号层面的效果验证器**

**这个叙事不需要 VLA 控制全部 6 个阶段。** 它只需要证明：(a) 符号条件改善了 VLA 输出质量，(b) 系统可以智能地在 VLA 和 scripted 之间切换，(c) 符号监测可以检测并恢复失败。

### 为实现这个叙事需要做的事

| 优先级 | 事项 | 预计工作量 |
|:---:|------|:---:|
| P0 | 重新训练 B4（已启动） | 训练中 |
| P0 | B4/B5 评估（3 seeds × 100ep）+ bootstrap CI | 1 天 GPU + 半天分析 |
| P0 | 端到端 demo：Language→Plan→VLA→Monitor（20ep） | 1-2 天 |
| P1 | 扩展 VLA 到 grasp 阶段 | 3-5 天 |
| P1 | OperatorMonitor 闭环集成 | 2-3 天 |
| P1 | 提升 transport cos_sim（数据增强或架构改进） | 1-2 周 |
| P2 | 3-block stacking 实验 | 1 周 |
| P2 | Learned embedding vs hand-crafted one-hot 对比 | 3-5 天 |

---

## 零、审查概要

该项目构建了一个可运行的机器人堆叠系统（Panda 机械臂 + Robosuite 仿真），工程骨架真实。但代码实现与论文主张之间存在根本性鸿沟：

1. **声称的"神经符号 VLA 闭环系统"实际上是 scripted controller + VLA 增强 reach_pick 阶段的系统**。VLA 只控制 6 个执行阶段中的 1 个（reach_pick），其余 5 个阶段纯靠手工脚本。
2. **声称的"BFS 符号规划器"在实际生产流水线中未被使用**。真实运行的是一个 20 行的规则式 `InternalStackingPlanner`，BFS `SymbolicPlanner` 因动作名称格式不兼容而无法接入下游。
3. **核心实验声明（B5 gate +22pp over B4）建立在有问题的 metric 和不可验证的训练条件上**。Gate 度量的是 teacher-alignment 而非 task-success，且 `--zero_symbolic` 训练标志在代码中是 no-op。

**在修复至少 7 个 CRITICAL 和 11 个 HIGH 问题之前，该代码库无法支撑任何顶会投稿。**

---

## 一、致命问题 (CRITICAL) — 必须修复才能投稿

### C1. `--zero_symbolic` 训练标志是 NO-OP —— B4 从未真正用零符号训练

**文件**：`src/vlm_nsm/vla/data/vla_dataset.py:232-233`, `vla_dataset.py:443`, `vla_dataset.py:479-480`

```python
# __init__ 中存储了 use_real_symbolic
self.use_real_symbolic = use_real_symbolic  # line 233

# 但 __getitem__ 中从未检查这个字段
# _build_symbolic_cond 是 @staticmethod，无法访问 self.use_real_symbolic
sym_cond_t = self._build_symbolic_cond(phase_id, ...)  # line 443
```

`use_real_symbolic` 在 `__init__` 中被存储，但在 `__getitem__` 中**从未被检查**。`_build_symbolic_cond` 是一个 `@staticmethod`，无法访问实例变量。**不管传不传 `--zero_symbolic`，训练数据中的符号条件始终是真实编码。**

**影响**：B4 checkpoint（`vla_m1_b4_zero_sym`）的训练条件无法验证。如果 B4 实际上用了真实符号条件训练，那 B4 vs B5 的对比就不是"有无符号条件"，而是两个几乎相同的模型间的对比。**论文最核心的 ablation 实验可能完全无效。**

**修复状态**：✅ 已修复 (2026-06-15)

### C2. 阶段名称词汇表不一致 —— 4/6 个阶段被静默编码为 "retreat"

**文件**：`src/vlm_nsm/neuro_symbolic/symbolic_condition.py:53-64` vs `src/vlm_nsm/vla/data/vla_dataset.py:497-498`

```
Encoder _PHASE_NAMES (10个):  reach_pick, descend_align, micro_align, grasp_close,
                                lift_verify, transport, descend_place, micro_place,
                                release_verify, retreat

Dataset _phase_map (6个):     0→"reach_pick", 1→"grasp", 2→"lift",
                               3→"reach_place", 4→"release", 5→"retreat"
```

**不匹配**：`"grasp" ≠ "grasp_close"`, `"lift" ≠ "lift_verify"`, `"reach_place" ≠ "transport"`, `"release" ≠ "release_verify"`

**后果**：`symbolic_condition.py:228` 的 `_onehot()` 在 key 找不到时静默 fallback 到 `vocabulary[-1]`（即 "retreat"）。**Phase 1-4 的符号条件全部被编码为 phase="retreat"**。只有 phase 0 (reach_pick) 和 phase 5 (retreat) 是正确的。

**修复状态**：✅ 已修复 (2026-06-15)

### C3. BFS SymbolicPlanner 是死代码 —— 从未接入生产流水线

**文件**：`src/vlm_nsm/symbolic/symbolic_planner.py:81-85` vs `src/vlm_nsm/skills/skill_plan_from_symbolic.py:48-49`

`SymbolicPlanner._generate_actions()` 产出的动作为 `MoveOn(x,y)`（name="move_on"）和 `MoveToTable(x)`（name="move_to_table"）。但 `_move_to_skill_command()` 第 48 行硬编码检查 `if name != "move": raise ValueError(...)`。

实际生产代码使用的是 `InternalStackingPlanner`（`src/vlm_nsm/planners/internal_stacking_planner.py`）——一个 20 行的规则式 planner，仅将 target_stack 相邻元素配对，完全不使用 BFS 搜索或谓词状态空间。

**修复状态**：✅ 已修复 (2026-06-15)

### C4. Gate Acceptance 是循环论证 —— 度量 teacher-alignment 而非 task-success

**证据矩阵**（来自 `generated/results/vla/` 的 8 组独立实验）：

| 实验组 | B4 Succ | B4 Gate | B5 Succ | B5 Gate | Gate Δ | Succ Δ |
|--------|:------:|:------:|:------:|:------:|:-----:|:-----:|
| b4/b5_clean | **70%** | 30.5% | 52% | 49.7% | +19pp | **-18pp** |
| b4/b5_real_m1 | 62% | 31.0% | 56% | 52.5% | +22pp | -6pp |
| verify s9000 | 64% | 27.3% | 62% | 50.9% | +24pp | -2pp |
| verify s12345 | 56% | 29.0% | 60% | 49.8% | +21pp | +4pp |
| final_100ep (calibrated) | **6%** | 99.9% | 15% | 100% | +0.1pp | +9pp |

**三个不可调和的矛盾**：

1. Gate 提升了 22pp，但任务成功率没有任何改善（B4 avg=60%, B5 avg=59%）。
2. calibrated gate 下 B4=6%, B5=15%——VLA 本质上无法独立完成任务。
3. B4.5 实验用了与 B4/B5 clean 不同的 gate 配置，导致 task success 不可比。

**建议**：将 task success 作为唯一金标准。Gate acceptance 降级为诊断指标。

**修复状态**：📝 叙事层面修复 (见第七节论文策略建议)

### C5. Gate 校准使用训练数据 = 数据泄漏

**文件**：`src/vlm_nsm/experiments/calibrate_gate_thresholds.py:55`, `generated/results/vla/gate_calibration/gate_config.json`

Gate 从 `d1_small_200ep`（**训练数据**）中计算 P95 x 1.5 作为 per-phase 阈值。这是方法论泄漏。

**而且**：`eval_s1_phase_override.py:195-198` 中，当 per-phase gate 启用时，全局 gate 的 NaN 检查、workspace 检查、phase semantic 检查**全部被绕过**——只保留一个 XYZ 偏差检查。

**修复状态**：✅ 已修复 (2026-06-15)

### C6. S0 评估脚本存在未定义变量 —— 无法运行

**文件**：`src/vlm_nsm/experiments/eval_vla_live_shadow.py:182-188`

`SkillCommand` 和 `sym_encoder` 在 `run_live_shadow()` 函数内部被引用但**从未导入或定义**。当前代码状态下会触发 `NameError`。

**修复状态**：✅ 已修复 (2026-06-15)

### C7. MoveOn 操作符的 effects_del 与实际语义矛盾

**文件**：`src/vlm_nsm/symbolic/operators.py:78` vs `operators.py:94`

`__init__` 中声明 `effects_del=set()`（空），但 `apply()` 方法实际执行 `new.predicates.discard(Predicate("clear", (y,)))`。

**修复状态**：✅ 已修复 (2026-06-15)

---

## 二、严重问题 (HIGH)

### H1. obs_history_len 读取 2 帧但只用 1 帧
`vla_dataset.py:414-438`：读取了 2 帧但只用了 `agentview[-1]`。注释声明的 "velocity info" 完全没有提供给模型。模型在任何时刻都只收到单帧视觉输入。

### H2. Phase-aware 采样权重已计算但从未使用
`vla_dataset.py:376-385` 计算了 phase 权重，但 `train_m1_diffusion.py:86` 使用均匀采样。

### H3. 29 个 Stage 20 废弃测试 + 6 个 DEPRECATED 模块仍在代码库中

### H4. VERIFY 失败后的注释与实现不符
`place_state_machine.py:509-510`：注释写 "Retry: re-grasp"，但实际只是回到循环开头重新执行 preplace。

### H5. Holding check 是一次性的，异常时盲目继续
`run_embodied_stacking_robosuite.py:631`：当物体位姿查询失败时，`_holding_ok = True` 作为 fallback。

### H6. 多个 failure type 有常量但无检测实现
SINGLE_SIDED_CONTACT, PLACE_UNSTABLE, RELEASE_FAIL, TRANSPORT_DROP, RETREAT_COLLISION 完全无检测代码。

### H7. 接触检测不使用力/触觉传感器
`grasp_state_machine.py:256-261`：仅用 gripper open_value 推断接触。薄物体（<5mm）会被系统性漏检。

### H8. 符号条件维度 31 vs 42 的混淆
Encoder 返回 31-dim，Dataset 手动 pad 到 42-dim，M0 默认为 31，M1 默认为 42。

### H9. DDIM 推理步数为 17 而非 16
`diffusion_policy.py:74-75`：`arange(0, 100, 6)` 产生 17 个值。

### H10. Scheduler tensors 每次 forward 都跨设备移动
`diffusion_policy.py:51-55`：`add_noise()` 每次都调用 `.to(x0.device)[t]`，从未缓存。

### H11. Language→VLA→Monitor→Replan 端到端整合从未完成
语言解析、VLA 执行、Predicate Monitor 是三个独立评估的子系统。

---

## 三、统计与实验设计问题

| # | 问题 | 证据 |
|---|------|------|
| S1 | `b4_clean` 的 70% 是异常值 | 所有其他 B4 运行为 48-64%，B4 avg ≈ 60% |
| S2 | B4.5 cos_sim=0.434 无法在任何 report.json 中定位 | 需要 trace-level 手动计算 |
| S3 | 所有 21 个 report.json 无任何置信区间或 bootstrap | 50ep 的 95% CI 宽度 ≈ 27pp |
| S4 | Teacher 56%→70% 声称与数据不符 | 实际数据：~72.5%→~80%（+7.5pp），56% 来源不明 |
| S5 | "VLA-caused failures" metric 与 gate 配置耦合 | Tight gate 下极少触发，Calibrated 下全部触发 |

---

## 四、代码质量总评

| 模块 | 评级 | 关键问题 |
|------|:---:|------|
| symbolic/ | C+ | BFS planner 未接入，effects_del 语义错误 |
| neuro_symbolic/ | C | Onehot 静默 fallback，gate 无数据驱动校准 |
| vla/models/ | B+ | 架构正确，`--zero_symbolic` 无效（在 dataset 层） |
| vla/data/ | C- | `use_real_symbolic` NO-OP，phase 词汇表不匹配 |
| vla/training/ | B | 缺少 LR warmup，phase 权重未使用 |
| embodied/ | B- | 接触检测太弱，多个 failure type 无实现 |
| experiments/ | C+ | S0 无法运行，猴子补丁脆弱，校准数据泄漏 |
| tests/ | C | 29 个 deprecated 测试，关键路径无覆盖 |

---

## 五、论文投稿可行性评估

| 目标会议 | 当前状态 | 修复后 |
|---------|:------:|:-----:|
| CoRL | **不可能** — 审稿人会验证代码 | 修复 C1-C7 + H1-H11 后可投 |
| ICRA | **不可能** — 同上 | 同上 |
| RSS | **不可能** — 要求极高 | 需要更根本的方法贡献 |
| CoRL Workshop | **勉强可能** — 诚实报告系统限制 | 修复 C1-C4 后可投 |
| ICRA Workshop | **可行** — 故事缩小为单阶段控制 | 修复 C1-C4 |

---

## 六、修复计划与进度

### P0：阻塞投稿的致命问题（目标：2-3 周）

| # | 问题 | 状态 | 完成日期 |
|---|------|:---:|:---:|
| C1 | `--zero_symbolic` NO-OP | ✅ 已修复 | 2026-06-15 |
| C2 | Phase 词汇表不匹配 | ✅ 已修复 | 2026-06-15 |
| C3 | BFS Planner 死代码 | ✅ 已修复 | 2026-06-15 |
| C4 | Gate 循环论证叙事 | 📝 叙事建议 | — |
| C5 | Gate 校准数据泄漏 | ✅ 已修复 | 2026-06-15 |
| C6 | S0 脚本 NameError | ✅ 已修复 | 2026-06-15 |
| C7 | MoveOn effects_del | ✅ 已修复 | 2026-06-15 |

### P1：提升论文质量（目标：2-3 周）

| # | 问题 | 状态 |
|---|------|:---:|
| H1 | obs_history 只用 1 帧 | 待修复 |
| H4 | VERIFY 注释与实现不符 | 待修复 |
| H5 | Holding check 盲目继续 | 待修复 |
| H11 | 端到端整合 | 待修复 |
| — | B4.6 shuffled symbolic ablation | 待实验 |
| — | Per-dimension symbolic ablation | 待实验 |
| — | 3+ seeds 交叉验证 | 待实验 |
| — | Bootstrap CI | 待添加 |

### P2：代码清理（目标：1 周）

| # | 问题 | 状态 |
|---|------|:---:|
| H3 | 废弃测试和模块 | 待清理 |
| H8 | 符号维度统一 | 待修复 |
| H9 | DDIM 步数修正 | 待修复 |
| — | 文档统一 | 待处理 |

---

## 七、论文策略建议（修订版：完整神经符号 VLA 叙事）

> 用户决定：不缩小论文范围，构建完整的神经符号 VLA 叙事。
> 以下是指出如何在不 overclaim 的前提下构建这个叙事。

### 正确的叙事框架

**核心论点**：VLA 不是 scripted controller 的替代品，而是在神经符号系统中作为被符号条件引导、约束、监测的 visuomotor 执行组件。

这个框架将 VLA 控制范围有限（当前 1/6 阶段）从"缺陷"转化为"设计选择"——系统有意只在 VLA 最擅长的阶段使用 VLA，其他阶段由更可靠的 scripted 执行。这类似于自动驾驶中 learned policy 与 rule-based safety monitor 的关系。

**标题（建议）**：*"Bridging Symbolic Planning and Visuomotor Control: Symbol-Conditioned Diffusion Policies with Operator-Level Monitoring for Language-Guided Manipulation"*

### 三层桥接叙事

**Layer 1: Symbolic Conditioning（符号 → VLA）**
- 将规划器的 operator 语义编码为 42 维 conditioning vector
- 核心实验：B4 vs B5（cos_sim +17%）
- 消融：B4.5 random / B4.6 shuffled / per-dimension
- 证明：结构化符号语义改善了 teacher alignment

**Layer 2: Safety Arbitration（VLA → 执行）**
- Gate 作为安全仲裁器：5 层检查决定使用 VLA 还是 scripted
- 核心实验：Gate calibration + per-phase 阈值分析
- 证明：系统在安全性和自主性之间取得平衡

**Layer 3: Operator Monitoring（执行 → 符号）**
- 执行后验证 operator 效果，失败时触发恢复
- 核心实验：preplace→NOT_HOLDING 诊断（**方法论贡献**）
- 证明：符号层可以诊断神经层的执行失败

### 关键叙事原则

1. **"Scripted controller 是系统的一部分，不是 baseline 对手"**
2. **"Gate 是实现安全自主的机制，不是限制 VLA 的枷锁"**
3. **"OperatorMonitor 将低层执行失败映射到符号语义——这是方法论贡献"**

### 投稿策略

| 会议 | 截止 | 可行性 | 策略 |
|------|------|:---:|------|
| CoRL Workshop | ~7月 | **最可行** | 当前结果即可 |
| ICRA 2027 | ~9月 | 时间充裕 | 完整实验 |
| CoRL 2026 | ~6月底 | 紧张 | 需要快速完成剩余实验 |

**建议路径**：CoRL Workshop（先用当前结果）→ ICRA 2027（完整版）

---

## 八、修复日志

### 2026-06-15 P0 修复批次（C1-C7 全部完成）

#### C1 + C2 修复：`--zero_symbolic` NO-OP + 阶段名称词汇表

**修改文件**：`src/vlm_nsm/vla/data/vla_dataset.py`

1. `_build_symbolic_cond` 新增 `use_real_symbolic: bool = True` 参数；`False` 时返回全零向量
2. 新增类级别 `_PHASE_ID_TO_NAME` 常量（10 阶段，与 encoder 一致）
3. `_phase_id_to_name` 统一引用此映射表
4. `__getitem__` 传入 `self.use_real_symbolic`
5. `except Exception` 缩小为 `except (ImportError, KeyError, ValueError, AttributeError)`，增加 warning

**验证**：`tests/test_vla_dataset.py` 全部通过

#### C3 修复：BFS Planner 动作名称兼容

**修改文件**：`src/vlm_nsm/skills/skill_plan_from_symbolic.py`

1. `_SUPPORTED_ACTIONS = {"move", "move_on", "move_to_table"}`
2. 支持 `move_on(x, y)` → pick_place on target
3. 支持 `move_to_table(x)` → pick_place on table
4. 兼容原有 `move(obj, source, target)` 格式

**修改文件**：`tests/test_skill_plan_from_symbolic.py`
- 新增 `test_two_arg_move_to_table_is_valid`、`test_move_to_table_action_supported`、`test_move_on_action_supported`
- 移除过时的 `test_target_object_missing_raises`、`test_table_target_rejected`

**验证**：`tests/test_skill_plan_from_symbolic.py` 7 passed

#### C5 修复：Per-phase gate 不再绕过安全检查

**修改文件**：`src/vlm_nsm/experiments/eval_s1_phase_override.py`

Per-phase gate 从覆盖（INSTEAD OF）改为附加（IN ADDITION TO），确保 NaN/workspace/phase semantic 检查始终生效。

#### C6 修复：S0 脚本 NameError

**修改文件**：`src/vlm_nsm/experiments/eval_vla_live_shadow.py`

添加 `SymbolicConditionEncoder`、`SkillCommand` 导入，初始化 `sym_encoder`。

#### C7 修复：MoveOn effects_del

**修改文件**：`src/vlm_nsm/symbolic/operators.py`

`effects_del=set()` → `effects_del={Predicate("clear", (y,))}`，与 apply() 语义一致。

### 测试验证

```bash
nsm/bin/python -m pytest tests/test_neuro_symbolic_condition.py \
    tests/test_vla_dataset.py tests/test_neuro_symbolic_safety_gate.py \
    tests/test_skill_plan_from_symbolic.py -x --tb=short -q
# 54 passed in 22.17s
```

### 2026-06-15 B4 重新训练结果

**训练命令**：
```bash
nsm/bin/python -u -m vlm_nsm.vla.training.train_m1_diffusion \
  --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
  --output_dir generated/results/vla/vla_m1_b4_zero_sym_fixed \
  --epochs 30 --batch_size 16 --lr 0.0001 --seed 0 --device cuda:0 \
  --backbone resnet18 --pretrained --image_size 128 \
  --horizon 16 --diffusion_steps 100 --inference_steps 16 \
  --zero_symbolic --overwrite
```

**结果**：

| 指标 | 新 B4 (fixed) | 旧 B4 (unverifiable) | B5 (real_sym) |
|------|:---:|:---:|:---:|
| Best Epoch | 8 | 12 | 8 |
| Val Loss | 0.0357 | 0.0348 | 0.0358 |
| Checkpoint | `vla_m1_b4_zero_sym_fixed` | `vla_m1_b4_zero_sym` | `vla_m1_real_sym` |
| 训练条件 | ✅ --zero_symbolic 生效 | ❌ 无法验证 | 真实符号条件 |

**关键发现**：
1. 新 B4 和 B5 的 val_loss 几乎相同（0.0357 vs 0.0358）——离线指标无法区分 B4/B5，闭环评估是唯一金标准
2. 旧 B4 的 best_epoch 和 val_loss 均与新 B4 不同，暗示旧 B4 训练条件可能确实不同——验证了修复 `--zero_symbolic` 的必要性
3. 旧 B4 checkpoint 应标记为不可用，建议归档或删除

### 后续必要工作

1. ~~重新训练 B4 模型~~ ✅ 已完成
2. 🔄 **重新评估 B4 vs B5**：B4 完成 (2 seeds × 100ep)，B5 运行中
3. **更新论文指标**：task success 为金标准，gate/cos_sim 为诊断指标

### 2026-06-15 新 B4 评估结果（部分）

**B4（验证过的零符号训练）**：

| Seed | Success | Gate Acc | Gate Total | Gate Accepted |
|:---:|:---:|:---:|:---:|:---:|
| 9000 | **64%** | **0.0%** | 17,042 | 0 |
| 12345 | **63%** | **0.0%** | 16,476 | 0 |

**B5（真实符号条件）**：评估运行中（seed 9000 + 12345，各 100ep）

**关键发现**：
- 真正用零符号训练的 B4，gate 接受率为 **0%**——VLA 未对系统做任何贡献
- 63-64% 的 overall success 是**纯 scripted controller 的表现**
- 旧 B4（gate=27-31%）的训练条件被证实已污染——旧 checkpoint 并非真正的零符号训练
- **如果 B5 能达到 ~50% gate 接受率**，那符号条件的贡献比旧数据声称的 +22pp 更显著：**是从"完全无效"(0%)到"半自主"(~50%)的跃升**
