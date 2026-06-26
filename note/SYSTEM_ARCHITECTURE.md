# VLA-NSM 系统真实状态审计报告

> 审计日期：2026-06-24 | 审计范围：全代码库、实验日志、结果文件、文档
> 证据优先级：代码实现 > 实验日志 > 文档表述

---

## 1. 项目一句话画像

**这个系统真实上是什么**：一个在 robosuite 仿真中运行的训练无关机器人操纵平台。控制端是手工调参的 PID 控制器 + 抓取/放置状态机，规划端是 BFS 符号搜索（仅堆叠 domain），语言端是 Qwen API（仅输出目标谓词）。系统依赖 MuJoCo 提供的 oracle 物体位姿。VLA 扩散模型已被放弃——实验证明其仅对最简单 reach_pick 阶段有贡献，且贡献与任务成功率负相关。

**这个系统不是什么**：不是端到端 VLA 系统。不是通用操纵平台。不是 sim-to-real 就绪的系统。不是神经符号理论创新——是一个工程上可行的训练无关操纵方案，带有诚实的 VLA 失败分析和门控归因方法论。

---

## 2. 实际系统架构

### 2.1 真实执行流程（以 2-block stacking 为例）

```
输入: "把红色方块放到绿色方块上面"
  → Qwen API (online_qwen_parser.py:113) → Goal JSON
  → Alignment 6 层校验 (goal_parser.py) → 合法目标谓词
  → BFS PlannerAdapter("bfs") (planner_adapter.py:70) → Action("move", ["cube1","table","cube2"])
  → run_task() (task_runner.py:68) → 技能循环:
      for skill in ["reach","grasp","lift","reach_place","release_place","retreat"]:
        SkillParams → SkillLibrary.get(skill) → skill.execute_with_pid(env, pid, params)
          → PID.act(phase, gripper) → action[:3] = v_des/0.05 → action[6] = gripper
          → env.step(action) → robosuite OSC 内环 → MuJoCo 仿真
        if not success → break
      → _check_goal(env, cubes, goal_predicates) → success/fail
输出: {"goal_success": True/False, "num_steps": N, "failure_type": "..."}
```

### 2.2 VLA/VLM 在系统中的真实角色

| 组件 | 实际角色 | 参与闭环控制？ |
|------|---------|:---:|
| **VLA Diffusion Policy** | 已放弃。历史实验对象，用于 gate attribution 分析。不参与当前系统执行 | ❌ |
| **Qwen VLM** | 语言→目标谓词翻译器。仅输出 goal JSON，不输出动作，不参与规划，不参与控制 | ❌ |
| **PID Controller** | EEF 位置伺服。每个 skill 的实际执行者。`operational_space_pid.py` | ✅ |
| **Grasp/Place SM** | 抓取和放置状态机。P-controller + 接触检测。`grasp_state_machine.py`, `place_state_machine.py` | ✅ |
| **BFS Planner** | 仅堆叠 domain。算子 {MoveOn, MoveToTable}。不处理 Hanoi/Pull/Push | 规划层 |

### 2.3 模块状态表

| 组件 | 代码位置 | 状态 | 主实验调用 | 备注 |
|------|---------|:---:|:---:|------|
| Qwen Parser | `language/online_qwen_parser.py` | ✅ 正常 | `run_e2e_pid_pipeline.py` | 需要 API key |
| BFS Planner | `symbolic/symbolic_planner.py` | ✅ 正常 | `PlannerAdapter("bfs")` | 仅支持堆叠算子 |
| InternalHanoiPlanner | `planners/internal_hanoi_planner.py` | ✅ 正常 | `run_hanoi_eval.py` | 递归算法，非 BFS |
| InternalStackingPlanner | `planners/internal_stacking_planner.py` | ✅ 正常 | 22 个旧脚本 | 20 行规则式，不会拆堆 |
| PID Controller | `embodied/controllers/operational_space_pid.py` | ✅ 正常 | `task_runner.py`, 所有 eval | 已修复 workspace clip + NaN guard + integral reset |
| Grasp SM | `embodied/grasp_state_machine.py` | ✅ 正常 | 所有任务 | 接触检测仅用 open_value |
| Place SM | `embodied/place_state_machine.py` | ✅ 正常 | 所有任务 | H4 VERIFY fix 已修 |
| TaskRunner | `embodied/task_runner.py` | ✅ 正常 | `run_task()` | 支持 pick_place + init_hook |
| SkillLibrary | `embodied/skills/` | ✅ 正常 | 6 技能 + PullSkill | PullSkill 仅自由 handle，非 slide joint |
| TaskSpec | `core/task_spec.py` | ✅ 正常 | +action_handler +init_hook +planner_kwargs | YAML 加载待完善 |
| OperatorMonitor | `neuro_symbolic/operator_monitor.py` | ✅ 正常 | replan 路径 | 手工查找表，非学习 |
| VLA Diffusion Policy | `vla/models/diffusion_policy.py` | ❌ 已放弃 | 仅 gate attribution 分析 | gate=0% (B4), 47% (B5) |
| Learned ID Model | `embodied/controllers/learned_id_model.py` | ❌ 已放弃 | 无 | 数据泄漏 + 多对一不可行 |
| 谓词编码器 | `training/train_predicate_encoder.py` | ❌ 已放弃 | 无 | 数据泄漏，未集成 |
| Transition State Machine | `embodied/transition_state_machine.py` | ⚠️ 半成品 | 定义但未在生产中使用 | PhaseDefinition.gain_overrides 从未被 runner 调用 |
| 感知层 (YOLO+stereo) | `perception/detector_perception.py` | 🚧 离线验证 | 无 | Stage 27: Δ=-1.7pp, 需实时化工程 |
| Gate Attribution | `experiments/summarize_gate_attribution.py` | ✅ 正常 | 离线分析已有 trace | 不是在线执行组件 |

---

## 3. 实验结果可信度表

| 实验 | 代码/日志位置 | N | 指标 | 结果 | 可信度 | 主要问题 |
|------|-------------|:---:|------|:---:|:---:|------|
| 2-block PID | `generated/results/control/pid_final/` | 100 | success | 91% | **高** | PID vs P 非同一 seed 受控对比 |
| 2-block P-controller | `generated/results/control/pid_vs_p/` | 100 | success | 94% | **高** | 同上 |
| 3-block PID+Replan | `run_3block_replan_v2.py` (6/17) | 15 | success | 93% | **低** | **当前代码路径不可复现 (0/50)**，GainSchedule 改变后失效 |
| Hanoi 3-disk | `generated/results/hanoi_pid_50ep/summary.json` | 50 | success | 84% | **高** | 独立 eval 脚本，非 TaskRunner 路径 |
| Pull Task | 内联测试代码 | 20 | success | 85% | **中** | 自由 handle block，非真抽屉。无正式 eval 脚本和 summary.json |
| VLA B5 tight gate | `generated/results/vla/b5_fixed_s*/` | 4×50 | gate acc | 47%±1.3pp | **高** | gate 参考系是 P-controller 非 PID |
| VLA B4 tight gate | `generated/results/vla/b4_fixed_s*/` | 2×100 | gate acc | 0% | **高** | --zero_symbolic 修复后训练的 B4 |
| VLA calibrated gate | `generated/results/vla/final_b*_100ep/` | 2×100 | success | B4=6%, B5=15% | **高** | gate 校准使用训练数据 (数据泄漏) |
| S0 live shadow | `generated/results/vla/s0_live_shadow/` | 50 | per-phase cos | reach=0.88, grasp=0.66, lift=0.19, place=0.55 | **中** | 脚本存在 NameError，当前代码不可直接运行 |
| Qwen E2E parse | `run_e2e_pid_pipeline.py` | 5 | parse success | 100% (5/5) | **中** | 仅 5 条指令，非系统性测试 |
| Alignment ablation | `eval_alignment_ablation.py` | 56 | reject rate | full: 18/56 pass | **中** | Parser-level（构造 bad JSON），非真实 Qwen 输出。MockGoalParser 不通真对齐 |
| BFS vs InternalStacking | 内联 Python | 1 | plan 正确性 | BFS ✅, ISP ❌ | **高** | 纯符号层对比，不依赖 MuJoCo |
| VLA 消融梯度 | `generated/results/vla/b4_*/`, `b4_6_*/`, `no_phase_*/` | 各 1-2 seeds | gate acc | B4=0%, B4.6=18%, No-phase=42%, B5=47% | **中** | N 小（B4 2×100ep, 其他 1-2×50ep），per-dimension 不全 |
| ID Model "700x" | `id_training_dataset.py:197` | — | val_loss | 0.35→0.0005 | **无效** | 数据泄漏：act_hist 包含 label。修复后 >0.1 |

---

## 4. Gate Attribution 专项评估

### 4.1 当前 gate 实现

Gate 有 5 层检查 (`safety_gate.py:146-247`)，实际启用的关键层是第 3 层——XYZ 偏差：
```python
xyz_dev = norm(vla_action[:3] - scripted_action[:3])
if xyz_dev > max_xyz_deviation: → REJECT
```
`max_xyz_deviation` 默认 0.025（tight gate）。calibrated gate 使用逐阶段阈值（p95×1.5，从训练数据计算）。

**eval_s1_phase_override.py:195-198** 曾存在 bug：per-phase gate 直接覆盖全局 gate 结果，绕过 NaN/workspace/semantic 检查。已在 6/15 修复。

### 4.2 gate attribution 目前能证明的

✅ VLA 动作与 teacher 动作的偏差在非 reach_pick 阶段系统性更大，导致 tight gate 在这些阶段 100% 拒绝  
✅ 放宽 gate 后 VLA 控制份额增加，但任务成功率下降（60%→15%）  
✅ S0 无门控余弦数据独立证实 VLA 在 grasp/lift/transport 方向质量差  
✅ 失败分布集中在 grasp (53%) 和 reach_place (37%)，非 reach_pick  

### 4.3 gate attribution 目前不能证明的

❌ 不能证明"VLA 偏差直接导致任务失败"——失败可能是 gate 拒绝后的 fallback 切换造成的副作用，而非 VLA 动作本身  
❌ 不能证明"任何 VLA 在这个任务上都不行"——仅测试了我们的特定训练配置 (200 demo, 60% teacher, ResNet18, 单帧)  
❌ 不能区分"gate 太严"和"VLA 太差"——缺少中间阈值的 gate sweep 曲线  
❌ 不能直接归因到具体 phase——缺少 phase intervention 实验（逐阶段替换 VLA）  

### 4.4 最短补强路径

1. **Gate sweep**（最高优先级）：5 个中间阈值 × 50ep → VLA% vs success 曲线
2. **Phase intervention**：保持其他 5 阶段 PID，仅 VLA 控制一个阶段 → 直接证明"哪个阶段 VLA 不行"
3. **Counterfactual rollout**：从相同初始状态出发，分别执行 VLA 和 PID → 对比 phase completion

---

## 5. 论文主张风险清单

| # | 风险表述 | 严重度 | 为什么危险 | 应如何改写 | 需要补什么 |
|---|---------|:---:|------|------|------|
| 1 | "结构化执行优于学习型控制" | 🔴 高 | 过于绝对。仅在特定条件下成立（精确操纵、已知位姿、小状态空间、200 demo 训练的特定 VLA）。其他任务/设定下可能不成立 | "Under the conditions of precise block stacking with known object poses and limited demonstration data, a structured PID+BFS system achieves higher reliability than the specific VLA configuration tested." | 限定条件范围 |
| 2 | "VLA 不适合精确操纵" | 🔴 高 | 宣称太强。仅测试了一个 VLA 配置。我们的 VLA 训练条件差（200 demo, 60% teacher, 单帧 RGB） | "In our experimental setup, the VLA's effective control contribution was limited to the reach phase, and its actions in contact/transport phases were systematically rejected by safety constraints." | 限定"under our training conditions" |
| 3 | "gate attribution 证明 VLA 贡献有限" | 🟡 中 | gate 对比的是 P-controller（非 PID），且 gate 阈值未经 sweep 验证 | 已有四重证据链支撑，需补充 gate sweep | Gate sweep 实验 |
| 4 | "符号规划是成功关键" | 🟡 中 | BFS 仅在堆叠任务中起作用。Hanoi 用专用递归算法。Pull Task 不需要 BFS | "BFS symbolic planning provides provably correct action sequences for stacking tasks, outperforming rule-based alternatives on non-trivial initial states." | 限定 domain 范围 |
| 5 | "training-free 系统具有泛化性" | 🟡 中 | "泛化性"仅在同类型任务（桌面物体操纵）内。没有新环境、新机器人、新物理属性的验证 | "The system demonstrates multi-task capability within tabletop object manipulation, handling stacking, tower rearrangement, and pull operations without task-specific neural network training." | 限定"within the same task family" |
| 6 | "E2E 管线已验证" | 🟢 低 | Qwen parse 5/5 + PID 执行 1/5 (mock) / 3/10 (Qwen)。管线连通但执行成功率偏低。描述准确但需标注成功率 | 现有表述可接受，标注成功率即可 | — |
| 7 | "VLA gate=0%" | 🟢 低 | 已在多个文档中修正为"VLA 仅在 reach_pick 有贡献（tight 38%, calibrated 100%），其他 phase 0%"。旧文档可能残留 | 确保全局一致 | 全局搜索并统一表述 |

---

## 6. 最值得保留的贡献

严格判断当前真正可写进论文的贡献：

| 贡献 | 可信度 | 论文价值 |
|------|:---:|:---:|
| **C1: Gate Attribution 方法论** | 高 | ★★★ | 不依赖特定 VLA 训练质量。四重证据链。作为评估工具可被复用到其他系统 |
| **C2: 训练无关架构的可行性** | 高 | ★★☆ | PID+BFS+Qwen 零训练。四任务 84-93%。任务本身简单但在社区中罕见如此诚实地 report |
| **C3: BFS vs 规则式规划器的符号层对比** | 高 | ★★☆ | 干净、可复现、不需要 MuJoCo。证明搜索比手工规则更正确 |
| **C4: Option-level replan** | 中 | ★★☆ | 3-block 0%→93% 展示了重规划价值，但当前代码不可复现——需修复或诚实降级 |
| **OperatorMonitor 诊断价值** | 高 | ★☆☆ | preplace→NOT_HOLDING 是好的 case study，但本质是手工查找表 |

**不建议写的**：VLA 扩散模型作为系统组件、ID 模型前馈、谓词编码器学习、slide joint 抽屉、端到端 VLA E2E

---

## 7. 下一步优先级

按"先补证据、再改叙事、最后扩展系统"排序：

| # | 事项 | 类型 | 预计 | 理由 |
|:---:|------|------|:---:|------|
| 1 | **Gate sweep 实验** | 补证据 | 4-5h GPU | 论文主图。5 阈值 × 50ep → VLA% vs success 曲线。消除"gate 太严"质疑 |
| 2 | **Phase intervention 实验** | 补证据 | 半天 | 逐阶段 VLA 替换 → 直接证明"哪个阶段不行"。比 gate 间接推断强一个档次 |
| 3 | **统一 VLA 叙事** | 改叙事 | 1h | 全局搜索替换"gate=0%从未控制"→"仅 reach_pick 贡献，与成功率负相关" |
| 4 | **回退 GainSchedule 复现 3-block 93%** | 补证据 | 半天 GPU | 消除不可复现的 red flag |
| 5 | **构造 Alignment 真实 Qwen 测试集** | 补证据 | 2h | 50-100 条真实 Qwen 指令 + 分类统计。替代当前 parser-level 消融 |
| 6 | **录制任务视频** | 展示 | 2h GPU | 2-block / 3-block / Hanoi / Pull。每任务 30s demo |
| 7 | **LLM skill planning + verification** | 扩展系统 | 2-3 天 | 解决规划层泛化瓶颈。系统最大的长期价值 |
| 8 | **Pull Task 正式 eval** | 补证据 | 1h GPU | 50ep + summary.json 落盘 |

---

> 问答记录 [advisor_qa_2026-06-19.md](advisor_qa_2026-06-19.md)
> 改进方案 [improvement_plan.md](improvement_plan.md)
> 论文初稿 [paper_draft.md](paper_draft.md)
> 项目仪表盘 [progress.md](progress.md)
> 已知问题 [problem.md](problem.md)
> 日志: [6/16](logs/2026-06-16-pid.md) [6/17](logs/2026-06-17.md) [6/18](logs/2026-06-18.md)
