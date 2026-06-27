# VLM-NSM 系统架构文档

> 更新日期：2026-06-26 | 基于 6/24-6/26 全部日志和实验产出  
> 当前重点：上层神经符号粗粒度选项抽象（Coarse Option Abstraction）

---

## 一、当前系统是什么

一个在 robosuite 仿真中运行的训练无关机器人堆叠系统。

核心思路：**用 coarse semantic options (pick/place) 替代 phase-level skills (reach/grasp/lift) 作为上层神经符号的 action space。** LLM 输出粗粒度任务程序，经过验证器检查后，由合并后的原子技能（AtomicPickSkill/AtomicPlaceSkill）在 PID 控制器上连续执行。

**当前重点**：上层神经符号管线（coarse DSL → verifier → bridge → atomic skill → robosuite）。

**不是**：端到端 VLA 系统。通用操纵平台。sim-to-real 就绪。门控归因不是当前主线。

---

## 二、架构演进：从 Phase-Level 到 Coarse Option

### 2.1 为什么放弃 phase-level skills

SkillLibrary 将执行分解为 6 个独立技能（reach → grasp → lift → reach_place → release_place → retreat），每个技能独立调用 `pid.set_phase()` / `pid.set_target()`。这在 2-block 中产生 0/20 的结果。

**根本原因**：option 粒度过细。6 个独立技能的 `set_phase()`/`set_target()` 打断 PID 积分连续性，技能之间没有传递精度需求——reach 不知道 grasp 需要多高的 XY 精度，grasp 不知道 lift 需要夹持多紧。

**验证实验**（三路径对照，各 20ep）：

| 路径 | 技能数 | 成功率 |
|------|:---:|:---:|
| Direct PID | 1 连续函数 | 20/20 (100%) |
| Decomposed (Reach+Grasp+Lift) | 3 独立技能 | 0/20 (0%) |
| AtomicPickSkill (reach+grasp+lift 合并) | 1 技能，PID 连续 | 19/20 (95%) |

假设验证通过：Decomposed 0% → Atomic 95%。**粒度是根因，不是参数。**

### 2.2 Coarse Option 方案

将上层 action space 定义为：

| Op | 参数 | 下层实现 | 说明 |
|----|------|---------|------|
| `pick(obj)` | object | AtomicPickSkill (reach+grasp+lift, 连续PID) | 抓取并抬升，单选项 |
| `place(obj, tgt)` | object, target | AtomicPlaceSkill (reach_place+release_place) | 放置，单选项 |
| `move(obj, src, tgt)` | object, source, target | AtomicPick + AtomicPlace | 复合操作 |
| `ensure(pred)` | predicate | 纯验证 gate | 不执行动作 |
| `achieve(pred)` | predicate | BFS PlannerAdapter 分解 | 不直接执行 |

**严格禁止**：reach, grasp, lift, reach_place, release_place, retreat——仅作为 AtomicPick/AtomicPlace 内部实现细节，不暴露给上层。

---

## 三、完整执行流程（当前实际运行路径）

```
输入: "把 cube1 叠到 cube2 上"
  │
  ├─ 1. Qwen API (online_qwen_parser.py, task_type="coarse_stacking")
  │     输出 CoarseTaskProgram JSON:
  │     {"steps": [
  │       {"op":"ensure","predicate":["clear","cube2"]},
  │       {"op":"pick","object":"cube1"},
  │       {"op":"place","object":"cube1","target":"cube2"}
  │     ]}
  │
  ├─ 2. CoarseOptionVerifier (verifier_bank.py)
  │     静态检查: 拦截 reach/grasp/lift 等 phase-level ops
  │     状态追踪: handempty → pick → holding → place → handempty
  │     前置条件: clear target, no self-place, no cyclic stack
  │     → 通过/拒绝
  │
  ├─ 3. BFS Reorder (coarse_reorder.py) — 仅在拒绝时触发
  │     提取目标谓词 → BFS PlannerAdapter("bfs").solve() → bottom-up 重排
  │     → 重新验证
  │
  ├─ 4. CoarsePlannerBridge.execution_sequence() (coarse_planner_bridge.py)
  │     ensure → NO-OP（纯验证，已在步骤2完成）
  │     pick  → {"skill":"atomic_pick","object":"cube1"}
  │     place → {"skill":"atomic_place","object":"cube1","target":"cube2"}
  │
  ├─ 5. 执行循环
  │   ├─ AtomicPickSkill.execute_with_pid(env, pid, params)
  │   │     Phase 1  Reach:   PID reach_pick, 收敛 10mm
  │   │     Phase 2  Grasp:   execute_grasp(skip_approach=True, micro_align=True)
  │   │     Phase 3  Lift:    PID lift, 收敛 20mm, holding check (Δz>10mm)
  │   │     → SkillResult(success, diagnostics)
  │   │
  │   └─ AtomicPlaceSkill.execute_with_pid(env, pid, params)
  │         Phase 1  ReachPlace: PID reach_place, approach_height=0.15
  │         Phase 2  Release:    execute_place(kp=0.05, max_delta=0.02)
  │         → SkillResult(success, diagnostics)
  │
  └─ 6. 目标检查
        physical_state_to_predicates() → goal_satisfied(mode="symbolic")
```

---

## 四、模块状态表（按当前重要性排序）

### 4.1 上层神经符号——当前主线

| 模块 | 文件 | 状态 | 关键数据 |
|------|------|:---:|------|
| **Coarse DSL** | `language/coarse_dsl.py` | ✅ 核心 | CoarseStep, CoarseTaskProgram, FORBIDDEN_OPS, is_valid() |
| **Qwen Coarse Parser** | `language/online_qwen_parser.py` | ✅ 核心 | COARSE_DSL_PROMPT, 10/10 parse, 0 forbidden |
| **CoarseOptionVerifier** | `neuro_symbolic/verifier_bank.py` | ✅ 核心 | 14/14 tests, 7/7 invalid 拒绝, 状态追踪 |
| **CoarsePlannerBridge** | `neuro_symbolic/coarse_planner_bridge.py` | ✅ 核心 | execution_sequence(), 5 bugs 已修复 |
| **BFS Reorder** | `neuro_symbolic/coarse_reorder.py` | ✅ 可用 | 1/1 top-down 修复, 不可修复正确拒绝 |
| **TaskComposer** | `language/task_composer.py` | ✅ 核心 | compose() 三模式: coarse/goal_only/failed |
| **OptionContract (coarse)** | `core/option_contract.py` | ✅ 核心 | pick/place/move/ensure/achieve 5 契约 |

### 4.2 执行层——当前主线

| 模块 | 文件 | 状态 | 关键数据 |
|------|------|:---:|------|
| **AtomicPickSkill** | `embodied/skills/atomic_pick_skill.py` | ✅ 核心 | 95% pick-only, PID 连续 |
| **AtomicPlaceSkill** | `embodied/skills/atomic_place_skill.py` | ✅ 核心 | 配合 AtomicPick 90% full |
| **PID Controller** | `embodied/controllers/operational_space_pid.py` | ✅ 核心 | workspace clip + NaN guard + integral reset |
| **Grasp SM** | `embodied/grasp_state_machine.py` | ✅ 核心 | skip_approach, micro_align |
| **Place SM** | `embodied/place_state_machine.py` | ✅ 核心 | kp=0.05, max_delta=0.02 |

### 4.3 规划层

| 模块 | 文件 | 状态 | 关键数据 |
|------|------|:---:|------|
| **BFS PlannerAdapter** | `core/planner_adapter.py` | ✅ 核心 | MoveOn + MoveToTable, 2-5 block |
| **BFS SymbolicPlanner** | `symbolic/symbolic_planner.py` | ✅ 核心 | BFS 搜索, max 10000 states |

### 4.4 支撑模块（已完成，不常修改）

| 模块 | 文件 | 状态 |
|------|------|:---:|
| **PredicateSceneGraph** | `core/predicate_scene_graph.py` | ✅ 稳定 |
| **OptionContract (phase-level)** | `core/option_contract.py` | ✅ 保留（内部使用） |
| **FailureType** | `core/failure_types.py` | ✅ 稳定, 19 值 enum |
| **GoalVerifier + SubgoalVerifier + PlanVerifier + RepairVerifier** | `neuro_symbolic/verifier_bank.py` | ✅ 稳定 |

### 4.5 非当前主线（experimental / deprecated）

| 模块 | 文件 | 状态 | 说明 |
|------|------|:---:|------|
| **FailureSummarizer** | `neuro_symbolic/failure_summarizer.py` | ⚠️ experimental | 8 种故障模式, 未接入执行路径 |
| **RepairProposer** | `neuro_symbolic/repair_proposer.py` | ⚠️ experimental | 12 条规则, 未接入执行路径 |
| **BT Executor** | `embodied/behavior_tree.py` | ⚠️ experimental | 恢复路径正确触发但策略不够有效 |
| **SkillLibrary (6-skill)** | `embodied/skills/` | ❌ deprecated | 0/20, option 粒度根因已证明 |
| **VLA Diffusion Policy** | `vla/models/diffusion_policy.py` | ❌ 已废弃 | 不在当前主线 |
| **Learned ID Model** | `embodied/controllers/learned_id_model.py` | ❌ 已废弃 | 数据泄漏 |
| **谓词编码器** | `training/train_predicate_encoder.py` | ❌ 已废弃 | 数据泄漏 |

---

## 五、核心实验数据

### 5.1 执行可靠性

| 实验 | N | 成功率 | 说明 |
|------|:---:|:---:|------|
| Direct PID 2-block | 20 | **95%** | 手写连续 PID，基准 |
| Atomic pick+place 2-block | 20 | **90%** | 合并选项，接近基准 |
| Decomposed 6-skill 2-block | 20 | **0%** | 粒度过细根因验证 |
| **Oracle coarse 2-block** | 20 | **100%** | 完整管线端到端 ✅ |
| Oracle coarse 3-block | 20 | **0%** | 失败在 AtomicSkill 层（nocontact 40% + verify_failed 60%） |

### 5.2 验证准确性

| 实验 | N | 准确率 | 说明 |
|------|:---:|:---:|------|
| CoarseOptionVerifier 单元测试 | 14 | **100%** | 覆盖所有边缘情况 |
| 非法程序拒绝 | 7 | **100%** | self-place, double-pick, occupied, cyclic, forbidden, unknown |
| Qwen 输出 forbidden ops | 10 | **0** | 完全遵守 coarse DSL 约束 |

### 5.3 Qwen 粗粒度程序生成

| 实验 | N | 成功率 | 说明 |
|------|:---:|:---:|------|
| Parse | 10 | **100%** | 所有指令正确解析 |
| Verify (直接) | 10 | **90%** | 1 例 top-down ordering 被正确拒绝 |
| BFS reorder 修复 | 1 | **100%** | 唯一拒绝例成功修复为 bottom-up |
| 2-block embodied | 25 | **88%** | 5 程序 × 5ep, 接近 oracle |

### 5.4 3-Block 失败根因定位

```
CoarseTaskProgram ✅ → CoarseOptionVerifier ✅ → BFS Ordering ✅
→ execution_sequence ✅ → AtomicPickSkill ❌ nocontact (40%)
                        → AtomicPlaceSkill ❌ verify_failed (60%)
```

**根因在 Atomic Skill / Physics 层**，非 coarse DSL 或 planner 层。AtomicPickSkill 在 3-cube 环境中对 cube2（非 cube1）的 reach 精度不足（XY 误差 ~9mm），MICRO_ALIGN 未能纠正。这是执行层参数适配问题，不属于上层 coarse option 抽象的范围。

---

## 六、3-Block vs 2-Block 差异分析

2-block 100% 但 3-block 0% 的直接原因：

| 因素 | 2-block | 3-block |
|------|--------|---------|
| 首个 pick 目标 | cube1 | cube2 |
| 物体位姿 | cube1 位置经过多轮调参验证 | cube2 在 3-cube env 中位姿不同 |
| AtomicPickSkill 参数 | 对 cube1 有效 | 对 cube2 精度不足 |
| 修复路径 | 不需要 | 需要适配 reach 收敛阈值或 grasp 微调参数 |

**这与上层设计无关**——coarse DSL、verifier、bridge 都正确工作。问题纯粹在执行层。

---

## 七、当前禁止事项

- ❌ 不调 PID 参数
- ❌ 不调 grasp 参数
- ❌ 不试图修复 decomposed 6-skill 路径（已证明不可行）
- ❌ 不把 reach/grasp/lift 作为上层 action
- ❌ 不新增上层模块
- ❌ 不扩 verifier 规则
- ❌ 不接视觉闭环
- ❌ 不把 VLA gate attribution 当作当前主线
- ❌ 不把下层控制参数修复写成主要贡献

---

## 八、新增源码清单（自 2026-06-24 审计以来）

### 上层神经符号

| 文件 | 状态 | 说明 |
|------|:---:|------|
| `language/coarse_dsl.py` | NEW | CoarseStep, CoarseTaskProgram, FORBIDDEN_OPS |
| `neuro_symbolic/coarse_planner_bridge.py` | NEW | coarse→execution bridge |
| `neuro_symbolic/coarse_reorder.py` | NEW | BFS 拒绝后重排 |
| `language/task_composer.py` | REWRITTEN | 支持 coarse DSL 三模式 |
| `language/online_qwen_parser.py` | MODIFIED | +coarse_stacking prompt |
| `neuro_symbolic/verifier_bank.py` | MODIFIED | +CoarseOptionVerifier |
| `core/option_contract.py` | MODIFIED | +5 coarse contracts |

### 执行层

| 文件 | 状态 | 说明 |
|------|:---:|------|
| `embodied/skills/atomic_pick_skill.py` | NEW | reach+grasp+lift 合并 |
| `embodied/skills/atomic_place_skill.py` | NEW | reach_place+release_place 合并 |

### 实验脚本

| 文件 | 状态 | 说明 |
|------|:---:|------|
| `experiments/debug_atomic_pick_skill.py` | NEW | 三路径对照实验 |
| `experiments/run_coarse_dsl_benchmark.py` | NEW | 60-task × 4-mode benchmark |
| `experiments/run_coarse_program_mujoco.py` | NEW | Coarse program robosuite runner |
| `experiments/run_coarse_oracle_eval.py` | NEW | 40-episode oracle eval |
| `experiments/run_qwen_coarse_embodied.py` | NEW | Qwen generation + execution |

### 生成报告

| 文件 | 说明 |
|------|------|
| `generated/reports/coarse_option_embodied_integration_report.md` | 主集成报告 |
| `generated/reports/coarse_planner_bridge_audit.md` | Bridge 语义审计 |
| `generated/reports/coarse_reorder_eval.md` | BFS 重排查验 |
| `generated/reports/qwen_coarse_embodied_eval.md` | Qwen 执行评测 |
| `generated/reports/coarse_option_abstraction_report.md` | Coarse DSL 初始报告 |

---

## 九、论文贡献框架（当前可用）

### 可以写（有数据支持）

| 贡献 | 证据 | 论文价值 |
|------|------|:---:|
| **Option 粒度决定执行可靠性** | Decomposed 6-skill 0% vs Atomic 2-skill 90% vs Direct 95% | ★★★ |
| **Coarse option abstraction 作为上层 action space** | pick/place DSL + verifier + bridge, 2-block 100% | ★★★ |
| **CoarseOptionVerifier 100% 拦截非法程序** | 14/14 tests, 7/7 invalid rejected | ★★☆ |
| **BFS reorder 修复 LLM 排序错误** | Top-down → bottom-up, 1/1 success | ★★☆ |
| **Qwen 生成物理可执行 coarse program** | 10/10 parse, 0/10 forbidden, 88% embodied | ★★☆ |
| **训练无关架构可行性** | PID 95% + BFS + Qwen, 零训练 | ★★☆ |

### 不可以写（数据不足）

- "TaskComposer 提升任务成功率"（执行成功依赖 AtomicSkill，非 TaskComposer）
- "3-block 端到端可用"（0/20，执行层待修复）
- "系统是完整闭环神经符号"（视觉未接入）
- "多任务泛化"（仅测试 stacking）
- "VLA gate attribution 是主线"（不是当前重点）

---

## 十、下一步（参考）

| # | 事项 | 类型 | 说明 |
|:---:|------|------|------|
| 1 | 3-block AtomicSkill 适配 | 修系统 | 让 AtomicPick/Place 在 3-cube 中达到 2-cube 同等水平 |
| 2 | Qwen coarse 全量 benchmark | 补数据 | 60 任务全量 API 测试 |
| 3 | Direct PID + coarse DSL 统一 | 架构 | 消除两条执行路径 |
| 4 | 3-block embodied 端到端 | 里程碑 | 验证 coarse DSL 在 3-block 的完整管线 |

---

> 关联文档：  
> [progress.md](progress.md) — 详细进度  
> [update.md](update.md) — 6/24 上层改造日志  
> [logs/06-25/skilllibrary_failure_report.md](logs/06-25/skilllibrary_failure_report.md) — SkillLibrary 失败完整分析（option 粒度根因）  
> [logs/2026-06-24/upper_ns_phase2_final_report.md](logs/2026-06-24/upper_ns_phase2_final_report.md) — Phase 2: 模块状态 + 证据  
> [logs/2026-06-24/upper_ns_phase3_final_report.md](logs/2026-06-24/upper_ns_phase3_final_report.md) — Phase 3: TaskComposer 40/40  
> [logs/2026-06-24/upper_ns_phase4_final_report.md](logs/2026-06-24/upper_ns_phase4_final_report.md) — Phase 4: SkillLibrary 三 bug + Atomic 突破  
> [../generated/reports/coarse_option_embodied_integration_report.md](../generated/reports/coarse_option_embodied_integration_report.md) — Embodied 集成  
> [../generated/reports/coarse_planner_bridge_audit.md](../generated/reports/coarse_planner_bridge_audit.md) — Bridge 审计
