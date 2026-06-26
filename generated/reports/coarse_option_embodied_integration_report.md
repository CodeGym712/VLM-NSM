# Coarse Option Embodied 集成报告

> 日期：2026-06-26 | 阶段：上层神经符号 embodied 集成  
> 管线：CoarseTaskProgram → CoarseOptionVerifier → CoarsePlannerBridge → AtomicPick/PlaceSkill → robosuite  
> 验收：2-block oracle **20/20 (100%)**，超过 85% 阈值 ✅

---

## 一、总览

打通了 coarse option abstraction 到 robosuite embodied execution 的完整管线。

| 层 | 2-block | 3-block | 根因归属 |
|----|:---:|:---:|------|
| **Verifier（CoarseOptionVerifier）** | ✅ 100% | ✅ 100% | 全部 oracle 正确通过；全部 invalid 正确拒绝 (7/7) |
| **Planner（BFS reorder）** | N/A | ✅ 正确 | top-down → bottom-up 修复正确 |
| **Ordering（coarse → execution）** | ✅ 正确 | ✅ 正确 | execution_sequence 正确生成 atomic_pick/atomic_place |
| **Atomic Skill** | ✅ 100% | ❌ 0% | 3-cube env 中 AtomicPickSkill grasp 失败 (nocontact 40%) + AtomicPlaceSkill release 失败 (verify_failed 60%) |
| **Physics（robosuite）** | ✅ 稳定 | ❌ 3-cube 对象位姿不同 | cube2 在 3-cube env 中的位置导致 reach 精度不足 |

---

## 二、管线验证

### 2.1 Oracle 2-Block：20/20 (100%)

```
程序：     pick(cube1), place(cube1, cube2)
验证：     通过
执行：     20/20 成功
平均 pick 步数：433
平均总步数：  ~960
失败类型：   无
```

**这是项目的关键突破**——coarse option abstraction 从 LLM 语义层打通到物理执行层，端到端 100% 可靠。

### 2.2 Oracle 3-Block：0/20 (0%)

```
程序：     pick(cube2), place(cube2,cube3), pick(cube1), place(cube1,cube2)
验证：     通过（bottom-up ordering 正确）
执行失败：
  nocontact:      8/20 (40%) — AtomicPickSkill 对 cube2 执行，grasp 子阶段
  verify_failed: 12/20 (60%) — AtomicPlaceSkill 在 pick 成功后，release 子阶段
```

**根因逐层定位**：

| 失败模式 | 归属层 | 具体原因 |
|---------|:---:|------|
| nocontact | **Atomic Skill / Physics** | cube2 在 3-cube env 中的初始位姿与 2-cube 不同。AtomicPickSkill 的 reach 收敛到约 9mm XY 误差，但 grasp 的 MICRO_ALIGN 在 60 步内未能纠正，导致夹爪未接触物体 |
| verify_failed | **Atomic Skill** | AtomicPlaceSkill 调用 execute_place()，其 verification 在 3-cube 场景中失败。reach_place 92 步快速收敛，但 release 427 步后 verification 仍不通过——物体未稳定放在目标上 |

**明确排除的层**：
- ❌ **Planner**：BFS 正确生成 bottom-up ordering
- ❌ **Ordering**：execution_sequence 映射正确
- ❌ **Verifier**：正确通过验证
- ❌ **Coarse DSL**：pick/place 语义正确

### 2.3 非法程序拒绝：7/7 (100%)

| 非法类型 | 步骤 | Verifier 响应 |
|---------|------|-------------|
| self_place | pick(c1), place(c1,c1) | 拒绝：self-place + target not clear |
| reach_forbidden | reach(c1), grasp(c1) | 拒绝："reach"/"grasp" 是 phase-level，禁止 |
| double_pick | pick(c1), pick(c2) | 拒绝：hand not empty（持有 c1） |
| place_occupied | pick(c1), place(c1,c2), pick(c3), place(c3,c2) | 拒绝：target 'cube2' not clear |
| unknown_object | pick(cube99) | 拒绝：unknown object 'cube99' |
| place_without_pick | place(c1,c2) | 拒绝：not holding 'cube1' |
| cyclic_stack | pick(c1), place(c1,c2), pick(c2), place(c2,c1) | 拒绝：cyclic stack + cube2 not clear |

**全部拒绝在执行前完成，无 robosuite 资源浪费。**

---

## 三、BFS Reorder：Top-Down → Bottom-Up 修复

### 3.1 场景

Qwen 对"把 cube1 叠到 cube2 上，再把 cube2 叠到 cube3 上"生成了 top-down 顺序：

```json
[{"op": "pick", "object": "cube1"},
 {"op": "place", "object": "cube1", "target": "cube2"},
 {"op": "pick", "object": "cube2"},           ← 被拒绝：cube2 上有 cube1，不 clear
 {"op": "place", "object": "cube2", "target": "cube3"}]
```

### 3.2 修复流程

1. **Verifier 拒绝**："step_3_pick: 'cube2' is not clear (has 'cube1' on top)"
2. **提取目标栈**：on(cube1,cube2), on(cube2,cube3) → bottom→top = `["cube3","cube2","cube1"]`
3. **BFS PlannerAdapter("bfs").solve(cubes, ["cube3","cube2","cube1"])**：
   - BFS 计划：`MOVE CUBE2 TABLE CUBE3`, `MOVE CUBE1 TABLE CUBE2`
   - 搜索状态数：2
4. **转换为 coarse steps**：
```json
[{"op": "pick", "object": "cube2"},
 {"op": "place", "object": "cube2", "target": "cube3"},
 {"op": "pick", "object": "cube1"},
 {"op": "place", "object": "cube1", "target": "cube2"}]
```
5. **重新验证**：✅ 通过 CoarseOptionVerifier

### 3.3 不可修复的情况

| 情况 | 拒绝原因 | 修复结果 |
|------|---------|---------|
| self_place | step_1: self-place cube1 on cube1 | 正确拒绝修复 |
| forbidden ops | step_0: 'reach' 是 phase-level，禁止 | 正确拒绝修复 |
| unknown object | step_0: unknown object 'cube99' | 正确拒绝修复 |

### 3.4 Bottom-Up 本身正确的情况

Qwen 在 4/5 的 3-block 程序中已正确生成 bottom-up 顺序，无需修复。

---

## 四、Qwen 粗粒度程序生成与执行

### 4.1 完整 10 程序评测结果

| ID | 指令 | 解析 | 验证 | 禁止操作 | 重排 | 执行 |
|----|------|:---:|:---:|:---:|:---:|:---:|
| Q2_01 | 把 cube1 叠到 cube2 上 | ✅ | ✅ | 0 | — | 5/5 |
| Q2_02 | 捡起 cube1 放到 cube2 上 | ✅ | ✅ | 0 | — | 5/5 |
| Q2_03 | 把 cube2 放到 cube1 上面 | ✅ | ✅ | 0 | — | 4/5 |
| Q2_04 | 把 cube1 移动到 cube2 上 | ✅ | ✅ | 0 | — | 4/5 |
| Q2_05 | 把 cube1 卸下来放到桌上 | ✅ | ✅ | 0 | — | 4/5 |
| Q3_01 | cube1 叠 cube2，然后 cube2 叠 cube3 | ✅ | ❌ | 0 | ✅ 已修复 | N/A |
| Q3_02 | 搭塔：cube3 底座，cube2 中层，cube1 顶层 | ✅ | ✅ | 0 | — | N/A |
| Q3_03 | 把所有方块叠起来，cube1 在最上面 | ✅ | ✅ | 0 | — | N/A |
| Q3_04 | 先把 cube2 放 cube3 上，再把 cube1 放 cube2 上 | ✅ | ✅ | 0 | — | N/A |
| Q3_05 | 排列：cube3 在桌上，cube1 在 cube3 上，cube2 在 cube1 上 | ✅ | ✅ | 0 | — | N/A |

**解析：10/10 (100%) | 验证通过：9/10 (90%) | 禁止操作：0/10 (0%)**  
**2-block 执行成功：22/25 (88%)（5 个程序 × 各 5 轮）**

### 4.2 2-Block 执行明细

```
Q2_01 "把 cube1 叠到 cube2 上"              5/5 ✅
Q2_02 "捡起 cube1 放到 cube2 上"             5/5 ✅
Q2_03 "把 cube2 放到 cube1 上面"             4/5 ✅（1 nocontact）
Q2_04 "把 cube1 移动到 cube2 上"             4/5 ✅（1 nocontact）
Q2_05 "把 cube1 卸下来放到桌上"              4/5 ✅（1 slip）
────────────────────────────────────────────────────────
总计 embodied:                             22/25 (88%)
```

**88% 执行成功率**——由 Qwen 生成的 coarse program 直接驱动物理执行，接近 oracle 的 100%。

### 4.3 Qwen 实际输出示例

**Q2_01 "把 cube1 叠到 cube2 上"**：
```json
{"version":"1.0","steps":[
  {"op":"ensure","predicate":["clear","cube2"]},
  {"op":"pick","object":"cube1"},
  {"op":"place","object":"cube1","target":"cube2"}
]}
```
→ 执行：5/5 ✅

**Q3_02 "搭塔：cube3 底座，cube2 中层，cube1 顶层"**：
```json
{"version":"1.0","steps":[
  {"op":"ensure","predicate":["clear","cube3"]},
  {"op":"pick","object":"cube2"},
  {"op":"place","object":"cube2","target":"cube3"},
  {"op":"ensure","predicate":["clear","cube2"]},
  {"op":"pick","object":"cube1"},
  {"op":"place","object":"cube1","target":"cube2"}
]}
```
→ 验证：✅（正确 bottom-up 顺序！）

### 4.4 关键观察

1. **Qwen 从不输出 forbidden ops**（0/10）——reach/grasp/lift/release_place 完全不出现
2. **Qwen 自动添加 ensure gate**——每个 pick 前加 `ensure(["clear", target])`
3. **Qwen 多数时候正确生成 bottom-up**——4/5 的 3-block 程序顺序正确
4. **88% 2-block 执行成功率**——与 oracle（100%）差距小

---

## 五、3-Block 失败层级归属

```
CoarseTaskProgram ✅ → CoarseOptionVerifier ✅ → BFS Ordering ✅
→ execution_sequence ✅ → AtomicPickSkill ❌ / AtomicPlaceSkill ❌
→ Physics（3-cube env 对象位姿与 2-cube 调参不同）
```

**结论：失败在 Atomic Skill / Physics 层。** Coarse option abstraction（pick/place DSL + verifier + bridge）正确工作到 execution_sequence 输出。问题在于 AtomicPickSkill 和 AtomicPlaceSkill 在 3-cube 环境中的物理执行参数需要调整，但根据当前禁止事项（不调 PID、不调 grasp 参数），此问题留待后续处理。

---

## 六、证据评估

### 充分支持（Strongly Supported）

| 论断 | 证据 |
|------|------|
| **Coarse DSL 拒绝 phase-level 操作** | 100% 拒绝 reach/grasp/lift 输入。Qwen API：0/10 输出含 forbidden ops |
| **Verifier 捕捉非法 pick/place 顺序** | 7/7 非法程序在执行前被拒绝。self-place、double-pick、目标被占、cyclic stack 全部捕获 |
| **Oracle coarse program 可通过 atomic option 执行（2-block）** | 20/20 (100%) 成功。平均 960 步/轮 |
| **BFS reorder 修复 top-down 顺序** | Top-down 3-block → BFS → 正确 bottom-up。重新验证通过 |
| **管线端到端可用（2-block）** | CoarseTaskProgram → Verifier → Bridge → AtomicSkill → robosuite → goal check：100% |

### 弱支持（Weakly Supported）

| 论断 | 证据 |
|------|------|
| **Qwen 可生成有效 coarse program** | 9/10 (90%) 验证通过。2-block 执行 88%。样本量小（n=10） |
| **Qwen 从不输出 forbidden ops** | 0/10。与 prompt 设计一致，但需更大样本量 |

### 不支持（Not Supported Unless Further Data）

| 论断 | 证据缺口 |
|------|---------|
| **TaskComposer 提升执行成功率** | 未单独测试。执行成功依赖 AtomicSkill，非 TaskComposer |
| **Coarse abstraction 提升多任务泛化** | 仅测试了 stacking。Drawer、Hanoi 未测试 |
| **系统是完整闭环神经符号** | 视觉/感知未接入。Scene graph 来自 oracle state，非摄像头 |
| **3-block 端到端可用** | 0/20 oracle。根因：AtomicPickSkill/AtomicPlaceSkill 在 3-cube env 中不可靠 |

---

## 七、输出文件清单

| 文件 | 内容 |
|------|------|
| `generated/reports/coarse_planner_bridge_audit.md` | Task 1：Bridge 语义审计（5 个 bug 已修复） |
| `generated/reports/coarse_option_embodied_integration_report.md` | 本文档：主集成报告 |
| `generated/reports/coarse_reorder_eval.md` | Task 4：BFS reorder 评测 |
| `generated/reports/qwen_coarse_embodied_eval.md` | Task 5：Qwen embodied 评测 |
| `generated/results/upper_ns/coarse_oracle_eval.json` | Task 3：20ep 2-block + 20ep 3-block + 7 非法拒绝 |
| `generated/results/upper_ns/coarse_reorder_eval.json` | Task 4：BFS reorder 结果 |
| `generated/results/upper_ns/qwen_coarse_embodied_eval.json` | Task 5：Qwen 生成+验证+执行 |
| `generated/coarse_programs/oracle_2block.json` | 2-block oracle coarse program |
| `generated/coarse_programs/oracle_3block.json` | 3-block oracle coarse program |
| `generated/coarse_programs/invalid_*.json` | 5 个非法测试程序 |

### 新增/修改源码

| 文件 | 变更 |
|------|------|
| `neuro_symbolic/coarse_planner_bridge.py` | 修复：execution_sequence、decompose_achieve、to_symbolic_actions |
| `neuro_symbolic/coarse_reorder.py` | 新增：BFS reorder 拒绝后修复模块 |
| `experiments/run_coarse_program_mujoco.py` | 新增：单轮 coarse program runner |
| `experiments/run_coarse_oracle_eval.py` | 新增：40 轮批量 oracle 评测 |
| `experiments/run_qwen_coarse_embodied.py` | 新增：Qwen 生成+验证+执行 |
| `language/coarse_dsl.py` | 修复：is_valid() 中 pick 未设置 holding 状态 |
