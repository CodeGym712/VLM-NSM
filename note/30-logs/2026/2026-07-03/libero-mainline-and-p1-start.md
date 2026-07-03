---
date: 2026-07-03
project: VLM-NSM
type: decision_and_stage_start
status: conditional_go_roundtrip_required
---
# LIBERO 新主线与 P1 启动记录

## 决策

论文主线调整为“官方示范驱动的可规划符号算子归纳与神经技能执行”。LIBERO `spatial + goal` 为主数据，Hanoi-300 为《The Price Is Not Right》复现数据，现有自采 HDF5 只作低层基线。

## 服务器证据

- `/rhome/taosy/LIBERO/` 可导入，LIBERO commit `8f1084e`；
- 环境为 Python 3.10.12、robosuite 1.4.0、MuJoCo 3.1.4；
- Parquet：Spatial 432 + Goal 428，共 860 episodes；
- 原始 HDF5：20 files、1000 demos、125978 time steps、约 12.03 GB；
- 1000/1000 demos 含 `states/actions/model_file/init_state`，state/action 等长；
- 500 条为 92D Spatial schema，500 条为 79D Goal schema；
- 已建立 60 条 shortest/median/longest round-trip cases，共 300 个候选恢复点。

## 审计结论

`CONDITIONAL_GO_ROUNDTRIP_REQUIRED`：HDF5 结构完整性通过，但现有测试只读取离线 snapshot，没有证明 environment rebuild、完整 state write-back、one-step 和 suffix replay。通过前不能启动谓词抽取和 Operator 训练。

## P1 当前任务

1. 生成真实 HDF5/BDDL/XML/init SHA-256 与环境版本 manifest；
2. 修正审计字段：`model_file_prefix`、`state_prefix15`、完整 hash 和 `dataset_root`；
3. 先运行 2 files×3 demos×5 points 的 30-point smoke；
4. smoke 通过后运行 20 files×3 demos×5 points 的 300-point round-trip；
5. 从 `sim.model` 建立 BDDL entity→body/joint/site 映射；
6. G1 通过后才进入 Oracle predicate traces。

## G1

- 20/20 文件能创建环境；
- immediate state max-abs≤1e-6 的恢复点≥99%；
- one-step max-abs≤1e-5 的恢复点≥95%；
- suffix reward/done/success 一致率≥95%；
- task-relevant BDDL entity→sim entity 覆盖 100%；
- 版本、完整哈希和审计报告可追溯。

## 禁止事项

G1前不训练新VLA、不实现复杂Operator聚类、不继续采集大规模同质Pick/Place数据。

## Obsidian同步

已更新 `VLM-NSM 系统迭代` 的总索引，并新增主线说明、详细架构、阶段计划、LIBERO审计和 ADR-002。原固定Domain目标推断路线保留为首要回退路线。
