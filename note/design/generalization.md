# 系统泛化性改造设计方案

> 日期：2026-06-16 | 关联：[progress.md](progress.md), [problem.md](problem.md), [controller.md](controller.md)

---

## 一、问题诊断

### 1.1 当前紧耦合点

```
run_embodied_stacking_robosuite.py
  ├─ 硬编码 6 阶段 (reach_pick→grasp→lift→reach_place→release→retreat)
  ├─ 硬编码 target = object_pose + offset
  ├─ 硬编码 InternalStackingPlanner（20 行规则式）
  ├─ 硬编码 31-dim 手工 one-hot 符号条件
  ├─ 硬编码 grasp_state_machine / place_state_machine
  └─ 硬编码 robosuite Stack 环境
```

换任务 = 重写 runner。这是根本问题。

### 1.2 三个核心问题

| 问题 | 现状 | 目标 |
|------|------|------|
| **泛化性** | runner 与 stacking 任务紧耦合 | Task 定义与执行分离，新任务只需配置 |
| **谓词学习** | 31-dim 手工 one-hot，不可扩展 | 从数据学习谓词嵌入，自动发现语义 |
| **规划器** | InternalStackingPlanner（规则式），BFS 未接入 | 统一规划接口，BFS for 小状态 + Metric-FF for 大状态 |

---

## 二、Task + Skill 抽象架构

### 2.1 核心思路

将 runner 拆分为三层：

```
┌─────────────────────────────────────────────┐
│ Task Definition (YAML/JSON)                  │
│   name, env, objects, planner, skills, goal  │
└──────────────────┬──────────────────────────┘
                   │ TaskSpec
┌──────────────────▼──────────────────────────┐
│ Skill Executor (Task-Agnostic)               │
│   PID controller + Skill Library            │
│   skill.execute(env, params) → result       │
└──────────────────┬──────────────────────────┘
                   │ skill commands
┌──────────────────▼──────────────────────────┐
│ Environment Adapter                          │
│   get_object_pose, get_ee_pose, step, reset  │
└─────────────────────────────────────────────┘
```

### 2.2 TaskSpec 定义

```python
@dataclass
class TaskSpec:
    """Declarative task definition — no code change needed for new tasks."""
    
    # Identity
    name: str                          # "stack_2block", "hanoi_3disk", "drawer_open"
    env_name: str                      # robosuite env name
    env_kwargs: dict                   # {"cube_names": [...], "robots": "Panda", ...}
    
    # Planning
    planner: str                       # "bfs", "internal_stacking", "internal_hanoi", "metric_ff"
    n_objects: int
    
    # Skills needed (ordered)
    skills: list[str]                  # ["reach", "grasp", "lift", "reach", "place", "retreat"]
    
    # Goal
    goal_predicates: list[list]        # [["on", "cube1", "cube2"]]
    goal_check: str                    # "predicate" | "pose" | "custom"
    
    # Controller
    controller: str = "pid"            # "pid" | "p_controller"
    pid_gains: dict | None = None      # per-skill PID gain overrides
    
    # Objects
    objects: dict[str, dict]           # {"cube1": {"size": 0.04, "color": "red"}, ...}
    work_objects: list[str]            # objects manipulated by the task
    fixture_objects: list[str]         # static objects (table, pegs, etc.)
    
    # Monitor
    monitor_enabled: bool = True
    max_replans: int = 3
    
    # Language
    language_instructions: list[str]   # E2E demo instructions
```

### 2.3 Skill Library

每个 Skill 是一个独立模块，绑定 PID 控制器 + 执行逻辑：

```python
class BaseSkill(ABC):
    """Abstract skill — all skills implement this interface."""
    
    name: str                          # "reach", "grasp", "lift", "place", "pull", "push"
    
    @abstractmethod
    def get_target(self, env, params: SkillParams) -> dict:
        """Compute target position from params + env state."""
        ...
    
    @abstractmethod
    def check_completion(self, env, params: SkillParams) -> bool:
        """Check if skill has completed successfully."""
        ...
    
    @abstractmethod
    def execute(self, env, pid: OperationalSpacePID, params: SkillParams) -> SkillResult:
        """Execute skill using PID controller."""
        ...


class ReachSkill(BaseSkill):
    """Move EEF to a target position."""
    
    name = "reach"
    
    def get_target(self, env, params):
        obj = env.get_object_pose(params.target_object)
        return {
            "x": obj["x"], "y": obj["y"],
            "z": obj["z"] + params.approach_height,
        }
    
    def check_completion(self, env, params):
        eef = env.get_eef_pose()
        target = self.get_target(env, params)
        return dist(eef, target) < params.reach_threshold
    
    def execute(self, env, pid, params):
        target = self.get_target(env, params)
        pid.set_phase("reach")
        pid.set_target(pos=target)
        for _ in range(params.max_steps):
            eef = env.get_eef_pose()
            pid.set_current(eef)
            action = pid.act("reach", None)
            env.step(action)
            if self.check_completion(env, params):
                return SkillResult(success=True)
        return SkillResult(success=False, failure="timeout")


class PullSkill(BaseSkill):
    """Pull an object (for drawer)."""
    
    name = "pull"
    
    def get_target(self, env, params):
        eef = env.get_eef_pose()
        return {
            "x": eef["x"] + params.pull_distance,
            "y": eef["y"],
            "z": eef["z"],
        }
    ...
```

### 2.4 Task Runner（任务无关）

```python
def run_task(task: TaskSpec, env, pid, max_steps=1000) -> TaskResult:
    """Generic task runner — works for any TaskSpec."""
    
    planner = create_planner(task.planner)
    plan = planner.solve(task)
    
    skill_library = SkillLibrary(pid)
    
    for step_idx, (skill_name, params) in enumerate(plan.skills):
        skill = skill_library.get(skill_name)
        result = skill.execute(env, pid, params)
        
        if not result.success and task.monitor_enabled:
            # Check predicates, attempt replan
            if task.max_replans > 0:
                current_state = extract_predicates(env)
                plan = planner.replan(current_state, task.goal_predicates)
                task.max_replans -= 1
                continue
        
        if not result.success:
            return TaskResult(success=False, failure=result.failure)
    
    # Verify goal
    goal_ok = check_goal(env, task.goal_predicates)
    return TaskResult(success=goal_ok)
```

### 2.5 示例：Hanoi Task 配置文件

```yaml
# configs/tasks/hanoi_3disk.yaml
name: hanoi_3disk
env_name: Stack
env_kwargs:
  cube_names: [cube1, cube2, cube3]
  robots: Panda
planner: internal_hanoi
n_objects: 3
skills: [reach, grasp, lift, reach, place, release, retreat]  # 可复用 stack skills
goal_predicates:
  - [on, cube1, cube2]
  - [on, cube2, cube3]
  - [on, cube3, peg3]
monitor_enabled: true
max_replans: 3
language_instructions:
  - "Move all cubes from peg1 to peg3, one at a time."
  - "Transfer the tower from left to right."
```

---

## 三、谓词学习方案

### 3.1 现状

```python
# 手工 31-dim one-hot, 2-block specific:
# pick_object_onehot(2) + target_object_onehot(3) + relation_onehot(3) 
# + phase_onehot(10) + pick_color(3) + target_color(3) + pick_size(1) + ...
```

问题：新任务需要重新设计维度（Hanoi 需要 peg 位置，drawer 需要 handle pose），不可扩展。

### 3.2 方案：Learned Predicate Embeddings

从 HDF5 数据联合学习：**对象表示 + 关系分类 + 相位编码**

```
当前 (hand-crafted):              方案 (learned):
─────────────────────             ─────────────
31-dim one-hot                    d-dim dense embedding
每维固定语义                      语义从数据中涌现
新任务需重新设计                   新任务只需重新训练
与 VLA 耦合                       独立模块，可插拔
```

### 3.3 架构

```python
class LearnedPredicateEncoder(nn.Module):
    """从 observation 学习谓词嵌入，替代手工 one-hot."""
    
    def __init__(self, n_objects=3, embed_dim=64):
        # Object embeddings (learned, not one-hot)
        self.object_embed = nn.Embedding(n_objects + 1, 32)  # +1 for table
        
        # Spatial relation classifier
        self.relation_head = nn.Sequential(
            nn.Linear(32 * 2 + 3 * 2, 64),  # 2 object embeds + 2 poses
            nn.ReLU(), nn.Linear(64, 3),      # on/table/none
        )
        
        # Phase encoder (from time-series context)
        self.phase_encoder = nn.GRU(32, 32, batch_first=True)
        
        # Predicate MLP: object embeds + relation + phase → predicate vector
        self.predicate_mlp = nn.Sequential(
            nn.Linear(32 * 2 + 3 + 32, 128),
            nn.ReLU(), nn.Linear(128, embed_dim),
        )
    
    def forward(self, pick_obj, target_obj, pick_pose, target_pose, phase_history):
        # Encode objects
        pick_emb = self.object_embed(pick_obj)
        target_emb = self.object_embed(target_obj)
        
        # Classify relation
        rel_input = torch.cat([pick_emb, target_emb, pick_pose, target_pose])
        rel_logits = self.relation_head(rel_input)
        
        # Encode phase
        _, phase_hidden = self.phase_encoder(phase_history)
        
        # Combine
        pred_input = torch.cat([pick_emb, target_emb, rel_logits, phase_hidden[-1]])
        predicate_vec = self.predicate_mlp(pred_input)
        
        return predicate_vec, rel_logits


# 训练数据: 从 HDF5 提取
# input:  (pick_id, target_id, pick_xyz, target_xyz, phase_seq)
# target: (predicate_vector from VLA training labels, relation_class)
```

### 3.4 训练流程

```bash
# Step 1: 从现有 HDF5 构建谓词训练集
python -m vlm_nsm.training.build_predicate_dataset \
  --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
  --output generated/datasets/predicate_training/

# Step 2: 训练 LearnedPredicateEncoder (~10K params)
python -m vlm_nsm.training.train_predicate_encoder \
  --data_dir generated/datasets/predicate_training/ \
  --output generated/results/predicate/encoder_v1/ \
  --embed_dim 64 --epochs 30

# Step 3: 替换 symbolic_condition.py 中的手工 one-hot
# symbolic_condition.py: SymbolicConditionEncoder → LearnedPredicateEncoder
```

### 3.5 优势

- **泛化**：新任务只需要新的 HDF5 数据，不需要重新设计维度
- **紧凑**：64-dim vs 31-dim（但信息密度高得多）
- **与 VLA 兼容**：嵌入向量直接拼接进 VLA conditioning
- **可解释**：relation head 输出可解释的 on/table/none 分类

---

## 四、规划器策略

### 4.1 对比

| 规划器 | 状态空间 | 最优性 | 速度 | 复杂任务 | 当前状态 |
|--------|:---:|:---:|:---:|:---:|:---:|
| InternalStackingPlanner | 仅 stacking | N/A | <1ms | ❌ | 生产线在用 |
| BFS SymbolicPlanner | <1000 states | ✅ 最短路径 | <10ms | ❌ | 已修复(C3)，未接入 |
| InternalHanoiPlanner | 仅 Hanoi | ✅ 最优 | <1ms | ❌ | 已有，未接入 |
| Metric-FF | <100K states | ✅ 启发式 | 10-100ms | ✅ | 适配器已有 |
| Fast Downward | <1M states | ✅ 最优 | 100ms-1s | ✅ | 适配器已有 |

### 4.2 建议：三层规划策略

```
TaskSpec.planner →
  ┌─────────────────────────────────────────────┐
  │           PlannerAdapter (统一接口)           │
  │  solve(task_spec) → Plan(skills, params)     │
  └──────────────────┬──────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
  BFS (默认)    Metric-FF        Fast Downward
  ≤100 states   100-10K states    >10K states
  2-block       3-4 block        多机器人
  Hanoi 3-4     Hanoi 5-6        复杂场景
  无外部依赖     PDDL 生成        PDDL 生成
```

**核心原则**：
- **BFS 是首选**：当前任务状态空间 <200，BFS 最优且零依赖
- **Metric-FF 作扩展**：3-block+ 或复杂 Hanoi，自动切换到启发式搜索
- **Fast Downward 备用**：未来多机器人场景
- **统一接口**：`PlannerAdapter` 根据 `task_spec.planner` 自动选择

### 4.3 当前 BFS 问题的根因

BFS `SymbolicPlanner` 之前是死代码（C3），因为：
1. Operator 名称不匹配：planner 产出 `MoveOn(x,y)` 但下游检查 `name == "move"`
2. 没有正确接入 SkillCommand 转换链路

**修复方案**：
```python
# planner_adapter.py — 统一规划接口
class PlannerAdapter:
    PLANNERS = {
        "bfs": SymbolicPlanner,
        "internal_stacking": InternalStackingPlanner,
        "internal_hanoi": InternalHanoiPlanner,
        "metric_ff": MetricFFAdapter,
        "fast_downward": FastDownwardAdapter,
    }
    
    def solve(self, task_spec: TaskSpec) -> Plan:
        planner_cls = self.PLANNERS[task_spec.planner]
        planner = planner_cls()
        raw_plan = planner.solve(**task_spec.planner_kwargs)
        return self._to_skill_commands(raw_plan, task_spec)
    
    def _to_skill_commands(self, raw_plan, task_spec):
        """统一转换：planner actions → SkillCommand + Phase + SymbolicCond"""
        ...
```

---

## 五、实施路线图

### Phase 1: Task 抽象层（2-3天）

```
Day 1: TaskSpec dataclass + YAML config loader
Day 2: Skill BaseClass + ReachSkill/GraspSkill/LiftSkill 重构
Day 3: TaskRunner (generic) + 2-block 回归验证 (必须保持 91%)
```

**交付**：`configs/tasks/stack_2block.yaml` → 与原 runner 结果一致

### Phase 2: Hanoi 任务（1-2天）

```
Day 1: HanoiTaskSpec 配置 + InternalHanoiPlanner 接入
Day 2: 3-disk Hanoi eval (50ep) + Monitor+Replan 集成
```

**交付**：Hanoi 3-disk running on PID, `configs/tasks/hanoi_3disk.yaml`

### Phase 3: 谓词学习（3-5天）

```
Day 1-2: LearnedPredicateEncoder 实现
Day 2-3: 从 HDF5 构建谓词训练集 + 训练
Day 4: 替换 symbolic_condition.py
Day 5: VLA B4/B5 重新对比 (optional)
```

**交付**：`learned_predicate_encoder.py`, 64-dim learned embeddings

### Phase 4: 规划器升级（1-2天）

```
Day 1: PlannerAdapter 统一接口 + Metric-FF 集成
Day 2: BFS vs Metric-FF 对比 (2-block)
```

### Phase 5: Drawer 任务（3-5天）

```
Day 1-2: MuJoCo drawer XML + RobosuiteEnvWrapper adapter
Day 2-3: PullSkill + PushSkill + grasp_handle
Day 4-5: DrawerTaskSpec + 评估
```

### 优先级

| 优先级 | Phase | 原因 |
|:---:|------|------|
| **P0** | Phase 1: Task 抽象 | 所有泛化的基础，必须不影响现有 91% |
| **P0** | Phase 2: Hanoi | 验证泛化能力，基础设施最完备 |
| **P1** | Phase 4: 规划器升级 | BFS 接入 + Metric-FF 扩展 |
| **P1** | Phase 3: 谓词学习 | 长期收益大，但不阻塞 Hanoi |
| **P2** | Phase 5: Drawer | 需要新建环境和技能 |
