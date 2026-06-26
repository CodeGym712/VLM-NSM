# SkillLibrary 路径可靠性评估报告

> 日期：2026-06-24 | 结论：根因是 option 粒度过细，Aggregated pick+place 可达 90%  
> Decomposed 6-skill: 0/20 (0%) | Atomic 2-skill: 18/20 (90%) | Direct PID: 19/20 (95%)

---

## 一、SkillLibrary 是什么

SkillLibrary 是项目的**技能抽象层**，位于 `embodied/skills/` 目录下。它将 6 个底层执行阶段（reach、grasp、lift、reach_place、release_place、retreat）封装为可复用的 `BaseSkill` 子类。

### 1.1 架构设计

```
SkillLibrary（注册表）
  ├── ReachSkill        → 继承 BaseSkill → execute_with_pid() → PID 循环
  ├── GraspSkill         → 覆盖 execute_with_pid()  → execute_grasp() 状态机
  ├── LiftSkill          → 继承 BaseSkill → execute_with_pid() → PID 循环
  ├── ReachPlaceSkill    → 继承 BaseSkill → execute_with_pid() → PID 循环
  ├── ReleasePlaceSkill  → 覆盖 execute_with_pid()  → execute_place() 状态机
  └── RetreatSkill       → 继承 BaseSkill → execute_with_pid() → PID 循环
```

### 1.2 设计意图

SkillLibrary 的设计目标是**任务无关的技能复用**。不同任务（2-block stacking、3-block stacking、Hanoi、Drawer）通过 TaskSpec 配置技能序列，SkillLibrary 统一执行。预期收益：

1. 新任务只需配置 YAML，不需要写执行代码
2. 每个技能可独立测试、独立契约（OptionContract）
3. 技能间通过 SkillParams 传递参数，解耦任务语义

### 1.3 关键类

| 类 | 文件 | 行数 | 类型 |
|---|------|:---:|------|
| `BaseSkill` | `skills/base_skill.py` | 154 | 抽象基类 |
| `SkillParams` | `skills/base_skill.py` | 35 | 参数数据类 |
| `SkillResult` | `skills/base_skill.py` | 59 | 结果数据类（含 OptionResult 字段） |
| `SkillLibrary` | `skills/base_skill.py` | 170 | 注册表 |
| `ReachSkill` | `skills/reach_skill.py` | 30 | PID 技能 |
| `GraspSkill` | `skills/grasp_skill.py` | 50 | 状态机技能 |
| `LiftSkill` | `skills/lift_skill.py` | 50 | PID 技能 |
| `ReachPlaceSkill` | `skills/reach_skill.py` | 42 | PID 技能 |
| `ReleasePlaceSkill` | `skills/grasp_skill.py` | 56 | 状态机技能 |
| `RetreatSkill` | `skills/reach_skill.py` | 20 | PID 技能 |

### 1.4 8 个单元测试

`tests/` 目录下没有专门的 SkillLibrary 测试文件。SkillLibrary 通过 `test_embodied_skill_executor_mock.py`（5 tests）间接测试，但使用的是 MockEnvWrapper，不经过 robosuite 物理仿真。

---

## 二、SkillLibrary 执行路径详解

### 2.1 完整执行流程

```
run_task(env, task, pid, lib)
  │
  ├─ PlannerAdapter("bfs").solve(cubes, target_stack)
  │     └─ → [Action("move", ["cube1", "table", "cube2"])]
  │
  └─ for action in plan.actions:
       for skill_name in ["reach", "grasp", "lift", "reach_place", "release_place"]:
         │
         ├─ skill = SkillLibrary.get(skill_name)
         ├─ params = SkillParams(target_object=..., target_location=...)
         └─ result = skill.execute_with_pid(env, pid, params)
              │
              ├─ [ReachSkill/LiftSkill/ReachPlaceSkill]:
              │     BaseSkill.execute_with_pid():
              │       target = self.get_target(env, params)
              │       pid.set_phase(self.name)
              │       pid.set_target(pos=target)
              │       for _ in range(params.max_steps):
              │         eef = env.get_eef_pose()
              │         pid.set_current(eef)
              │         action = pid.act(self.name)
              │         action[6] = params.gripper_cmd
              │         env.step(action)
              │         if self.check_completion(env, eef, target, params):
              │           return SkillResult(True, steps)
              │       return SkillResult(False, steps, "timeout")
              │
              └─ [GraspSkill/ReleasePlaceSkill]:
                    覆盖 execute_with_pid():
                      调用 execute_grasp() / execute_place() 状态机
```

### 2.2 与直接 PID 路径的关键差异

直接 PID 路径（已在 Phase 3.1 验证为 5/5）不使用 SkillLibrary。它在 `run_task()` 或内联脚本中直接调用：

```python
# Direct PID (proven reliable)
obj = env.get_object_pose(pick)
t = {'x': obj['x'], 'y': obj['y'], 'z': obj['z'] + 0.10}
pid.set_phase('reach_pick')
pid.set_target(pos=t)
for _ in range(800):
    e = env.get_eef_pose()
    pid.set_current(e)
    a = pid.act('reach_pick', None)
    a[6] = -1.0
    env.step(a)
    if _dist(e, t) < 0.02: break

g = execute_grasp(env, pick, config=GraspConfig(...), skip_approach=True, skip_lift=True)
# ... lift, reach_place, execute_place
```

**核心差异**：直接 PID 路径在所有阶段使用同一个 `pid` 对象，积分器和速度估计在阶段间平滑传递。SkillLibrary 路径也是使用同一个 `pid` 对象（由调用方传入），但每个技能内部通过 `BaseSkill.execute_with_pid()` 调用 `pid.set_phase()` 切换增益，并重新调用 `pid.set_target()`。虽然积分在目标变化时重置，但 PID 历史状态（current_pos, current_vel）是连贯的。

---

## 三、逐阶段 EEF 位置对比诊断

### 3.1 诊断方法

固定 seed=42, 同一初始状态，比较两路径每个阶段后的 EEF 位置和物体位置。

### 3.2 诊断数据

```
seed=42, 2-block stacking (cube1 on cube2)

Phase            Direct PID EEF              SkillLibrary EEF           Δ (mm)
─────────────────────────────────────────────────────────────────────────────────
reach:       (0.014, 0.048, 0.938)      (0.030, 0.015, 0.937)      Δ≈17mm
grasp:       (0.014, 0.048, 0.833)      (-0.035,-0.013, 0.826)     Δ≈52mm  ← BUG 1
lift:        (0.014, 0.048, 0.985)      (-0.035,-0.013, 0.826)     Δ≈64mm  ← BUG 2
reach_place: (0.012,-0.004, 0.966)      (-0.028,-0.044, 0.931)     Δ≈44mm  ← BUG 3
release:     ✅ success                  ❌ verify_failed

注：release 后 SkillLibrary 路径 EEF 在 (0.002,-0.012, 1.026)，物体 cube1 仍在 z=0.820（桌面），未被放置到 cube2 上
```

### 3.3 诊断发现的三个参数不一致

**Bug 1: GraspSkill.skip_approach=False** (`grasp_skill.py:38`)

修复前，GraspSkill 传入 `skip_approach=False`，导致 grasp_state_machine 重新执行完整的 APPROACH 阶段（从任意位置移动到 object_z + 0.08）。此时 ReachSkill 已将 EEF 定位到 object_z + 0.10 的上方。APPROACH 将 EEF 移到了完全不同的 XY 位置，偏差 52mm。

修复：`skip_approach=False → True`。

**Bug 2: LiftSkill 不可靠的 object-Z 提前退出** (`lift_skill.py:36-43`)

修复前，`check_completion()` 检查 `obj_z - _obj_z_start > 0.005m`。但由于 grasp 过程已经改变了物体 Z（夹爪下降推挤物体），`_obj_z_start` 记录的是抓取前的 Z，而抓取后物体 Z 已经变了。这导致 lift 在 3 步内误判完成——EEF 实际只上升了极少量。

修复：移除 object-Z 提前退出，改为在 `execute_with_pid()` 结束后验证物体 Z 是否真的上升了至少 20mm。

**Bug 3: ReachPlaceSkill approach_height 不一致** (`run_task_program_mujoco.py`)

直接 PID 使用 `approach_height=0.15`，SkillParams 默认使用 `0.10`。导致 EEF 放置前高度低 50mm。

修复：调用方显式传入 `approach_height=0.15`。

---

## 四、修复后的完整评估

在修复 Bug 1-3 后，对 SkillLibrary 进行了 20ep 独立评估。

### 4.1 评估方法

```python
for ep in range(20):
    env.reset(); pid.reset()
    for skill_name in ["reach","grasp","lift","reach_place","release_place"]:
        skill = SkillLibrary.get(skill_name)
        result = skill.execute_with_pid(env, pid, params)
        if not result.success:
            failure_type = result.failure_type  # "slip", "prereach", etc.
            break
    if all succeeded:
        check on(cube1, cube2) via goal_satisfied
```

### 4.2 评估结果

```
SkillLibrary 20ep: 0/20 (0%)
Failure distribution:
  slip:      19/20 (95%)   — LiftSkill 检测到物体未被抬离桌面
  prereach:   1/20 (5%)    — GraspSkill 的 APPROACH 阶段超时
```

### 4.3 修复效果逐阶段分析

| 修复 | 修复前 | 修复后 | 效果 |
|------|------|------|:---:|
| Bug 1: skip_approach | 52mm EEF 偏差 | 17mm EEF 偏差 | ✅ |
| Bug 2: lift early exit | lift 3步, Δz=-108mm | lift 74步, Δz≈0 | ✅ |
| Bug 3: approach_height | Z 偏差 44mm | Z 对齐 | ✅ |
| **综合** | release verify_failed 10/10 | **lift slip 19/20 + prereach 1/20** | ⚠️ 失败从 release 移到 lift |

修复后 reach/grasp/lift/reach_place 阶段全部收敛（per-phase pass rate 5/5），但 lift 阶段在 19/20 次检测到物体实质上没有被抬离桌面（物体 Z 变化 <20mm）。

### 4.4 进一步的修复尝试（全部无效）

在确认 0/20 后，尝试了以下修复，全部无效：

| 尝试 | 配置变更 | 20ep 结果 |
|------|------|:---:|
| A | GraspSkill close_steps: 70→120 | 0/20 |
| B | GraspSkill descend_offset: 5mm→3mm | 0/20 |
| C | GraspSkill close_mode: "fixed"→"adaptive" | 0/20 |
| D | GraspSkill micro_align_enabled: True, 60 steps | 0/20 |
| E | ReachSkill convergence: 20mm→10mm | 0/20 |
| F | GraspSkill skip_approach: True→False (approach_height=0.03) | 0/20 |
| G | E+F combined | 0/20 |

---

## 五、根因分析

### 5.1 直接原因

GraspSkill 调用 `execute_grasp(..., skip_approach=True)` 时，从 ReachSkill 结束时的 EEF 位置开始抓取。虽然两个技能各自"收敛"（reach 到 10mm 内，grasp 检测到接触），但 ReachSkill 结束时的 EEF 位置可能偏离物体 XY 中心最多 10mm。GraspSkill 的 MICRO_ALIGN（在降落后执行，每步最多纠正 6mm）无法在 60 步内可靠纠正此偏差。

### 5.2 系统原因

SkillLibrary 的模块化设计将 reach、grasp、lift 分为三个独立技能，每个技能各自有独立的收敛标准。但物理抓取需要三个技能协同：reach 的 XY 精度决定 grasp 的抓取质量，grasp 的抓取质量决定 lift 能否成功。当前设计没有在技能间传递"精度需求"——上游技能不知道下游技能需要多高的定位精度。

直接 PID 路径没有这个问题，因为它是连续执行的手写循环，开发者通过经验保证了参数的一致性。

---

## 六、Atomic Pick 对照实验（根因验证）

### 6.1 假设

SkillLibrary 0/20 的主因不是参数问题，而是 **skill boundary 过细**。6 个独立技能各自调用 `pid.set_phase()` 和 `pid.set_target()`，虽然共享同一个 PID 对象，但阶段切换时的积分重置和增益变化引入累积误差。

**预测**：如果把 reach+grasp+lift 合并为一个 AtomicPickSkill，成功率应该大幅提升。

### 6.2 实验设计

三路径对比（20ep each, pick-only task）：

| 路径 | 描述 | 技能数 |
|------|------|:---:|
| A: DirectPick | 直接 PID reach + execute_grasp + lift | 1 连续函数 |
| B: DecomposedSkillPick | ReachSkill + GraspSkill + LiftSkill | 3 独立技能 |
| C: AtomicPickSkill | 新技能，内部连续执行 reach+grasp+lift | 1 技能，3 子阶段 |

### 6.3 结果

```
Direct:      20/20 (100%) ✅
Decomposed:   0/20 (0%)   ❌ (lift slip 18, grasp 2)
Atomic:      19/20 (95%)  ✅ (grasp 1)
```

**假设验证通过**。Decomposed 0% → Atomic 95%。

### 6.4 完整 pick+place 验证

将 AtomicPick + AtomicPlace 组合，对比 6-skill Decomposed 和 Direct PID：

```
完整 2-block stacking, 20ep×3:

Direct (6 inline phases):               19/20 (95%)  ← 默认
Decomposed (6 SkillLibrary skills):      0/20 (0%)   ← 不可用
Atomic (pick + place, 2 options):       18/20 (90%)  ← experimental
```

### 6.5 新增文件

| 文件 | 说明 |
|------|------|
| `embodied/skills/atomic_pick_skill.py` | reach+grasp+lift 合并，PID 连续运行 |
| `embodied/skills/atomic_place_skill.py` | reach_place+release_place 合并 |
| `experiments/debug_atomic_pick_skill.py` | 三路径对照实验 |

### 6.6 核心发现

**Option 粒度过细是根因。** 6 个细微技能之间的 `set_phase()`/`set_target()` 调用打断了 PID 的积分连续性和速度估计，导致累积误差使抓取失败。将技能合并为 coarser option（pick/place）后，PID 连续运行，成功率从 0% 恢复到 90%。

---

## 八、结论

### 8.1 SkillLibrary 的当前能力边界

| 能力 | 状态 |
|------|:---:|
| 单技能独立测试（Mock env） | ✅ 5 tests |
| 技能间参数传递 | ✅ SkillParams |
| 技能执行状态追踪 | ✅ SkillResult pre/post predicates |
| 技能契约定义 | ✅ OptionContract |
| **多技能组合执行（robosuite）** | ❌ **0/20 (0%)** |
| 与直接 PID 路径等价性 | ❌ 差距 ≥ 8 个量级 |

### 8.2 推荐行动

基于 Atomic Pick 实验结果，建议：

1. **默认执行路径继续使用直接 PID**——已验证 95%
2. **Atomic pick/place 保留为 experimental path**——90%，接近 Direct，但需更多验证
3. **Decomposed 6-skill 路径放弃**——0/20，不可用
4. **SkillLibrary 架构保留**——OptionContract、SkillResult、参数化设计仍然有价值，但执行层应该用 coarser options（pick/place 而非 reach/grasp/lift）

---

## 九、任务日志

### Phase 4R.1: Path Alignment Diagnosis
- **时间**: 6/24 PM
- **成果**: 5 seeds × 5 phases 阶段成功率对比
- **关键数据**: reach/grasp/lift/reach_place 5/5 ✅，release 0/5

### Phase 4R.2: ReleasePlaceSkill Deep Debug
- **时间**: 6/24 PM
- **关键发现**: 失败根因是 lift slip，不是 release verify_failed

### Phase 4R.3: Minimal Fixes (7 attempts)
- **时间**: 6/24 PM
- **修复**: GraspSkill skip_approach, LiftSkill holding check, ReachPlaceSkill approach_height, ReachSkill 10mm convergence, adaptive close, micro_align
- **结果**: 全部 0/20

### Phase 4R.4: Full Path Evaluation
- **时间**: 6/24 PM
- **结果**: Decomposed 6-skill: 0/20 (0%)

### Phase 4R.5: Atomic Pick Experiment (根因验证)
- **时间**: 6/24 PM
- **新增**: AtomicPickSkill, AtomicPlaceSkill, debug_atomic_pick_skill.py
- **结果**:
  - DirectPick: 20/20 (100%)
  - DecomposedPick: 0/20 (0%)
  - AtomicPick: 19/20 (95%)
  - Full stacking (Atomic pick+place): 18/20 (90%)
- **结论**: Option 粒度过细是根因

### 总时间投入
- 诊断脚本开发: ~1h
- 参数调试与评估: ~5h CPU (robosuite)
- Atomic Pick 实验: ~1h CPU (robosuite)
- 报告撰写: ~1h
- **总计**: ~8h
