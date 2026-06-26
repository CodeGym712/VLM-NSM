# VLM-NSM 代码结构

> 更新：2026-06-10 | 230 文件, 49K行 | 1680 测试

---

## 核心模块

```text
src/vlm_nsm/
├── symbolic/           # 符号层: 谓词、状态图、BFS规划器、算子
│   ├── predicates.py
│   ├── state_extractor.py
│   ├── goals.py
│   ├── goal_validator.py
│   ├── operators.py      # MoveOn, MoveToTable (带preconditions/effects)
│   └── symbolic_planner.py
│
├── neuro_symbolic/      # 神经符号层 (Phase 2.0新增)
│   ├── symbolic_condition.py   # SkillCommand → 符号向量
│   ├── safety_gate.py          # 5层安全检查
│   └── operator_monitor.py     # per-operator效果验证
│
├── vla/                 # VLA模型与训练 (Phase 0新增)
│   ├── models/
│   │   ├── vision_encoder.py       # ResNet18/34 多视角编码
│   │   ├── diffusion_unet.py       # Conditional 1D UNet
│   │   └── diffusion_policy.py     # DiffusionPolicy + DDPM/DDIM
│   ├── data/
│   │   └── vla_dataset.py          # HDF5数据集 + phase采样
│   └── training/
│       ├── train_m0.py             # M0 (ResNet+MLP BC) 训练
│       └── train_m1_diffusion.py   # M1 (Diffusion) 训练
│
├── embodied/            # 具身控制
│   ├── controllers/     # P-controller, ReplayController
│   ├── grasp_state_machine.py     # 闭环抓取(含MICRO_ALIGN)
│   ├── place_state_machine.py     # 闭环放置(含high-place)
│   ├── skill_step_decomposer.py
│   ├── demonstration_recorder.py
│   ├── control_trace.py           # 逐step执行日志
│   └── failure_taxonomy.py        # 18种标准化失败
│
├── skills/              # 技能接口
│   ├── skill_types.py             # SkillCommand (含preconditions/effects)
│   └── skill_plan_from_symbolic.py
│
├── execution/           # 技能执行
│   ├── scripted_skill_executor.py
│   └── act_skill_executor.py
│
├── executors/           # 执行器 (部分DEPRECATED)
│   ├── robosuite_env_wrapper.py
│   ├── symbolic_executor.py
│   └── vla_executor.py            # DEPRECATED
│
├── experiments/         # 实验脚本 (40+)
│   ├── collect_vla_demos.py       # 主数据采集
│   ├── eval_s1_phase_override.py  # S1闭环评估
│   ├── eval_vla_live_shadow.py    # S0 Live Shadow
│   ├── audit_vla_dataset.py       # D0审计
│   └── run_embodied_stacking_robosuite.py  # 主episode runner
│
├── training/            # 训练 (含DEPRECATED Stage 20)
│   ├── train_stage20_goal_conditioned_rnn.py  # DEPRECATED
│   └── train_stage20_goal_conditioned_mlp.py  # DEPRECATED
│
├── perception/          # 视觉感知 (YOLO, stereo, camera calib)
├── language/            # Qwen在线解析
├── planners/            # Hanoi/Stacking/Metric-FF
├── rl/                  # PPO (历史)
├── grounding/           # 视觉-符号grounding
└── observation/         # SceneState, Oracle/Perception sources
```

---

## 新增 vs 历史

```text
Phase 0-2.0 新增 (6/7-6/10):
  neuro_symbolic/  (3 files)
  vla/             (8 files)
  embodied/control_trace.py, failure_taxonomy.py
  experiments/collect_vla_demos.py, eval_s1_phase_override.py, eval_vla_live_shadow.py, audit_vla_dataset.py

Phase 2.0 修改:
  skills/skill_types.py (+preconditions/effects)
  skills/skill_plan_from_symbolic.py (+_extract_predicate_dicts)
  embodied/place_state_machine.py (+high_place, +periodic holding check)
  embodied/grasp_state_machine.py (+MICRO_ALIGN, +grasp profiles)
  experiments/run_embodied_stacking_robosuite.py (+holding check+regrasp, +high_place, +best_config)

已标记 DEPRECATED:
  training/train_stage20_goal_conditioned_rnn.py
  training/train_stage20_goal_conditioned_mlp.py
  data/stage20_role_conditioned_dataset.py
  embodied/stage20_learned_executor.py
  executors/vla_executor.py
  executors/diffusion_executor.py
```

---

## 测试覆盖

```text
VLA targeted: 78 tests (operator_monitor, symbolic_condition, safety_gate, vla_dataset, vision_encoder, diffusion_policy)
Full regression: 1680 passed, 4 skipped
```

## Phase 2.0-3.0 新增文件 (2026-06-09 ~ 06-12)

### 符号抽象管线
```text
src/vlm_nsm/symbolic/
  graph_learner.py           # 从NSM移植: 状态迁移图学习+最小互模拟
  operator_learner.py        # DFA→算子定义提取
  dfa_to_pddl.py             # DFA→PDDL直接转换器

src/vlm_nsm/experiments/
  run_symbolic_learning.py   # 数据适配器: manifest→(state,action,next_state)三元组
```

### Per-Operator Diffusion
```text
src/vlm_nsm/vla/
  data/vla_dataset.py         # +operator_filter参数, +use_real_symbolic
  training/train_m1_diffusion.py  # +--operator_filter, +--zero_symbolic
```

### Hanoi 环境
```text
src/vlm_nsm/executors/
  stack_three_cube_env.py     # 3-cube + 3-peg Hanoi环境
```

### 输出
```text
generated/results/symbolic/
  stacking_op50_graph.dfa     # 2-block, 50ep
  stacking_d1best_graph.dfa   # 2-block, 500ep
  stacking_3cube_graph.dfa    # 3-cube, 合成BFS
  stacking_3cube_real_graph.dfa  # 3-cube, 真实24ep
  pddl/domain.pddl            # 学到的参数化PDDL domain

generated/videos/
  hanoi_pegs.mp4              # Hanoi塔搭建 + peg可视化
  hanoi_demo.mp4              # Hanoi完整demo
  3cube_single_c1_on_c2.mp4   # 3-cube单步成功

generated/results/vla/
  vla_m1_pick/                # M1-Pick (phase 0-2)
  vla_m1_place/               # M1-PlaceOn (phase 3-5), val=0.025
  vla_m1_b4_zero_sym/         # B4 baseline (zero symbolic training)
  b4_clean/                   # B4闭环结果
  b5_clean/                   # B5闭环结果
