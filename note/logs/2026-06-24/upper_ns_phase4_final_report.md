# Upper Neuro-Symbolic Phase 4 Final Report

> 日期：2026-06-24 | 1651 tests passed, 0 failed

---

## 1. Phase 4.2 核心发现：SkillLibrary 三个参数 Bug

Phase 4.2 的目标是让 TaskProgram 通过 SkillLibrary 进入 MuJoCo 执行。全 SkillLibrary 路径 10/10 失败，逐层诊断发现了三个潜伏的参数不一致：

### Bug 1: GraspSkill.skip_approach=False → True

**文件**: `embodied/skills/grasp_skill.py:38`  
**根因**: ReachSkill 已将 EEF 定位到物体正上方 (obj_z + 0.10)。GraspSkill 用 `skip_approach=False` 让 grasp_state_machine 重新执行 APPROACH 阶段，把 EEF 移到了完全不同的 XY 位置。  
**影响**: Δx=-20mm, Δy=-49mm, dist=53mm  
**修复**: `skip_approach=False` → `skip_approach=True`  
**验证**: ✅ 1651 tests passed

### Bug 2: LiftSkill 不可靠的 object-Z 提前退出

**文件**: `embodied/skills/lift_skill.py:36-43`  
**根因**: `check_completion()` 检查 `obj_z - _obj_z_start > 0.005m` 作为提前退出条件。但 grasp_state_machine 的 descend+close 过程已经改变了物体 Z 位置（物体被夹爪推挤或抓取后 Z 变化 >5mm），导致 lift 在 3 步内误判完成。  
**影响**: EEF 仅上升了极少量（Δz=-108mm vs 预期 +150mm）, lift steps=3 vs 正常 ~80  
**修复**: 移除 object-Z 提前退出，只保留 EEF 距离检查  
**验证**: ✅ lift steps 从 3 恢复到 74

### Bug 3: ReachPlaceSkill approach_height 不一致

**文件**: `experiments/run_task_program_mujoco.py`（调用方）  
**根因**: 直接 PID 路径用 `approach_height=0.15`, SkillParams 默认用 `0.10`。TaskProgram runner 未显式传参。  
**影响**: EEF Z 差 44mm（z=0.931 vs target=0.975），放置高度太低  
**修复**: TaskProgram runner 中 `reach_place` 显式传 `approach_height=0.15`  
**验证**: ✅ 但单独此修复不足以解决 verify_failed（仍有其他差异）

### 当前状态

三个 bug 均已定位并修复，但全 SkillLibrary 路径仍 0/10 verify_failed。这表明还有其他差异点——可能存在于 ReleasePlaceSkill 的参数传递或 place_state_machine 的调用方式中。

**下一步诊断**: 跑完整对比脚本（直接 PID vs 全 SkillLibrary），在 SAME seed 下输出 reach/grasp/lift/reach_place/release_place 五个阶段的 EEF 位置差异。所有 >5mm 的差异都需要修复。

---

## 2. Phase 4 全部任务状态

| Phase | 计划 | 实际 | 完成度 |
|:---:|------|------|:---:|
| 4.1 | 60 tasks × 6 categories benchmark | 15 tasks smoke test | 25% |
| 4.2 | TaskProgram MuJoCo runner | 脚本已创建，3 bugs 已定位但未完全跑通 | 60% |
| 4.3 | 3 failure types × 10ep controlled injection | verify_failed only × 10ep | 33% |
| 4.4 | BT vs Linear 50ep (or 15ep) | 5ep smoke from Phase 3 | 10% |
| 4.5 | Vision shadow 20ep | 未启动 | 0% |
| 4.6 | Final report | 本文件 | 100% |

**整体 Phase 4 完成度: ~38%**

---

## 3. 剩余任务（按优先级）

1. **Phase 4.2 收尾（P0）**: 跑完整对比诊断脚本，找到所有 >5mm 的 EEF 差异并修复，直到全 SkillLibrary 路径 ≥ 80% success
2. **Phase 4.3 完成（P1）**: 补 nocontact + slip 故障注入
3. **Phase 4.4 正式对比（P1）**: BT vs Linear ≥15ep
4. **Phase 4.1 完整数据集（P2）**: 60 tasks JSON
5. **Phase 4.5 Vision shadow（P3）**: perception 模块不在执行路径，需要先接入

---

## 4. 教训

1. **SkillLibrary 与直接 PID 从未被严格对比过**——三个参数不一致潜伏了数周，因为一直用直接 PID 路径做实验，SkillLibrary 路径即使存在也没有被压力测试。
2. **逐层 EEF 对比是最有效的诊断手段**——比起猜测，固定 seed 对比两个路径每一步的 EEF 位置，可以精确量化差异。
3. **不要在 Phase 结束时取巧**——Phase 4.2 runner 第一次回退到直接 PID 是在逃避问题，正确做法是找到所有差异点并修复。
