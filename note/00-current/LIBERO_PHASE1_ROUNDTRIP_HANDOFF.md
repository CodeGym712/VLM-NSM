---
project: VLM-NSM
date: 2026-07-03
status: active
phase: P1-L0R
owner: local-and-server
---
# LIBERO P1-L0R 状态恢复交接清单

## 当前主线

从无人工 Operator 标签的官方 LIBERO 示范中恢复稳定谓词轨迹、自动发现事件边界并归纳参数化一阶 Operator；显式规划器负责长程组合，神经策略负责连续执行。当前不得转向纯 VLA 或下层 PID 调参。

## 已完成证据

- 20 个 HDF5：LIBERO Spatial 10 + Goal 10；
- 1000 demonstrations、125978 时间步、约 12.03 GB；
- 1000/1000 含 `states/actions/model_file/init_state`；
- 1000/1000 state/action 等长，action dimension=7；
- 500 条 `LIBERO_SPATIAL_v1/92D`，500 条 `LIBERO_GOAL_v1/79D`；
- 已生成 20 文件×shortest/median/longest，共 60 条 round-trip cases；
- 每条选择 0/25/50/75/100% 五点，共 300 个候选恢复点。

## Day 1 文件质量结论

以下四个文件结构和连接关系通过：

```text
environment_versions.json
source_inventory.json
episode_manifest.jsonl
roundtrip_cases.jsonl
```

但必须先修正：

1. `source_inventory.sha256` 当前为 `SKIP_TOO_LARGE`，需生成真实 SHA-256；
2. `bddl_content_hash/repository_version/env_info` 为空；
3. `model_file` 实际只有 300 字符，应改名 `model_file_prefix`；
4. `state_sample` 只有前 15 维，应改名 `state_prefix15`；
5. `xml_hash/init_state_hash` 改为完整 64 位 hex SHA-256；
6. 增加 `dataset_root`，runner 不依赖当前工作目录；
7. round-trip runner 必须通过 `source_file + demo_key + t` 从 HDF5 读取完整 XML、state 和 action，禁止使用 JSON 前缀样本进行恢复。

## 服务器任务顺序

### S1：重建不可变 manifest

输出：

```text
generated/libero_audit/environment_versions.json
generated/libero_audit/source_inventory.json
generated/libero_audit/episode_manifest.jsonl
generated/libero_audit/roundtrip_cases.jsonl
```

环境版本必须包含：Python、LIBERO commit、robosuite、MuJoCo、NumPy、h5py、controller config/hash、renderer backend。

### S2：30-point smoke

先选择：

- 1 个 Spatial HDF5；
- 1 个 Goal HDF5；
- 每文件 shortest/median/longest 三条；
- 每条五个时间点。

总计 `2 files × 3 demos × 5 snapshots = 30 points`。

每个点执行：

1. 从 HDF5 读取完整 per-demo `model_file`；
2. 创建/重建对应环境；
3. 从 HDF5 读取完整 `states[t]`；
4. 写入 simulator state 并 `forward()`；
5. 立即读回，计算 full/qpos/qvel max-abs 与 RMSE；
6. 若 `t<T-1`，执行 `actions[t]`，与记录的 `states[t+1]` 比较；
7. 从 25/50/75% 点回放至结尾，比较 reward、done、task success；
8. 保存异常堆栈，不静默跳过。

输出：

```text
generated/libero_audit/roundtrip_smoke30.jsonl
generated/libero_audit/roundtrip_smoke30_summary.json
```

### S3：300-point 全量门

仅当 smoke30 环境创建、完整 state 写回和 one-step 路径全部工作后，扩大到 20 文件、60 demos、300 点。

输出：

```text
generated/libero_audit/roundtrip300.jsonl
generated/libero_audit/roundtrip300_summary.json
```

### S4：命名实体映射

从 `sim.model` 导出 body/joint/site name→index，并将 BDDL objects/fixtures/regions 映射到 simulator entity。对象位姿必须从 body/object API 读取，抽屉和炉灶从对应 joint qpos 读取。

禁止使用：

```text
(state_dim - robot_dim) / 7
```

输出：

```text
generated/libero_audit/entity_registry.json
generated/libero_audit/bddl_entity_mapping.jsonl
generated/libero_audit/entity_mapping_audit.json
```

## G1 硬门槛

- 20/20 文件可创建环境；
- 300 点 immediate state max-abs≤1e-6 的比例≥99%；
- one-step max-abs≤1e-5 的比例≥95%；
- suffix reward/done/success 一致率≥95%；
- task-relevant BDDL entity→sim entity 覆盖 100%；
- 20 tasks×3 frames 的人工关键实体/关节正确率≥95%；
- source、BDDL、XML、init state 和环境版本可追溯。

## Go / Stop

- G1 全过：进入 P2 Oracle predicate traces；
- round-trip 通过但实体映射失败：只允许一天修复，仍失败则停止 P2；
- immediate/one-step 失败：检查 XML、版本和 controller，一天内不能定位则停止 P2；
- HDF5↔Parquet 映射当前不阻塞 Operator Pilot，但 P6 视觉技能训练前必须完成；
- G1 前禁止谓词训练、Operator 聚类、VLA/diffusion 训练和新增大规模自采数据。

## GitHub→服务器流程

本地只提交代码、配置、测试和 Markdown；HDF5、JSONL 审计结果、模型与视频留在服务器。服务器每次运行前记录 Git commit，结果目录写入相同 commit 与环境版本。
