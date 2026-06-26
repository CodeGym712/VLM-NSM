# Stage 6C: 实验矩阵补跑、度量修正与报告表格固化

## 概述

Stage 6C 在 Stage 6A/6B 的基础上：
1. 修正 language variation 指标语义（schema_validity、alignment_applicable）
2. 补跑被覆盖和未完成实验组合（full、weak、medium 各 prompt variant）
3. 增加 medium prompt，更好展示 Alignment 修复价值
4. 完成 Qwen2.5-14B vs Qwen2.5-32B 对比
5. 生成 report-ready 表格和失败案例分析

## 重要说明

- Qwen 模型用于 inference/evaluation，不是 training。
- 所有实验使用 temperature=0.0，确保确定性输出。
- 使用 device_map="auto" 在 4×4090 GPU 上运行 Qwen2.5-32B-Instruct。
- n_blocks=3 固定，与 easy set 一致。
- Mock parser 用于单元测试和快速验证；Qwen parser 用于实际评估。

## 实验结果记录

### Easy set 结果（Stage 6A）

- Hanoi 15 samples + Stacking 15 samples，mock parser（仅验证 easy set 完整性）
- Qwen2.5-32B-Instruct easy set: 15/15 成功 (full prompt + full alignment)

### Hard set full prompt 结果（Stage 6B）

- hanoi_hard_zh_en.jsonl: 22 samples, 6 categories
  - alias_heavy (4), underspecified (5), object_synonym (4),
    mixed_language (3), noisy_format (3), misleading (3)
- stacking_hard_zh_en.jsonl: 22 samples, 6 categories
  - alias_heavy (4), underspecified (5), object_synonym (4),
    mixed_language (3), noisy_format (3), partial_goal (3)
- Qwen2.5-32B full prompt + full alignment:
  - Hanoi 22/22 goal success (100%)
  - Stacking 22/22 goal success (100%)

### Weak prompt 结果

- Hanoi hard weak prompt: JSON parse 21/22, Alignment 0/22, Goal 0/22
  → 主要测试 schema prompt 必要性，而非 alignment 修复能力
- 结果显示 weak prompt 产生可解析但 schema-invalid 的输出，验证了 prompt 工程的重要性

### Medium prompt 说明

新增 medium prompt 介于 full 和 weak 之间：
- 要求输出 JSON，给出最小 JSON schema
- 提供 objects 和 valid_predicates
- 不提供完整 canonical 示例链
- 不详细解释 alias 规则
- 不强调 completeness repair
- 目的：让 Alignment Layer 有机会发挥作用

### Full prompt repair_rate=0 解释

full prompt 包含：
1. 完整的 goal_predicates 示例（Hanoi/O Stacking 链式结构）
2. 严格的 alias 规则（above→on 等）
3. completeness repair 指令
4. PDDL 格式和 type 约束

模型在 full prompt 下直接产生规范输出，Alignment Layer 无需修复。
这是 prompt engineering 的成功，而非 alignment 模块无效。

## 文件名覆盖 Bug

原输出文件名：
`language_variation_{task}_{difficulty}_{model}.json`

不包含 prompt_variant 和 alignment_mode，导致不同组合结果互相覆盖。

修复后文件名：
`language_variation_{task}_{difficulty}_{model}_{prompt_variant}_{alignment_mode}.json`

影响范围：首批 8 个组合被覆盖（full/weak x no_alignment/no_repair）。
修复后使用 `run_qwen_all_modes.py` 和 `run_language_matrix.py` 补跑。

## 指标语义修正

### no_alignment 模式

- `alignment_applicable = false`（新增字段）
- `alignment_success = null`（不是 True）
- summary 中 alignment_success_rate 显示为 "N/A"

### schema_validity 字段（新增）

- 检查 goal_predicates 是否存在且为 `list[list]`
- JSON parse 成功但没有 goal_predicates → `schema_validity=false`
- JSON parse 失败 → `schema_validity=false`
- summary 新增 schema_validity_rate

## 代码变更

### 新增文件
- `prompts/hanoi_goal_parser_medium.txt` — Hanoi medium prompt
- `prompts/stacking_goal_parser_medium.txt` — Stacking medium prompt
- `src/vlm_nsm/experiments/run_language_matrix.py` — 矩阵运行器
- `tests/test_schema_validity_metrics.py` — schema_validity 测试
- `tests/test_no_alignment_summary.py` — no_alignment 语义测试
- `tests/test_medium_prompt_files.py` — medium prompt 测试
- `tests/test_language_matrix_runner.py` — 矩阵运行器测试
- `tests/test_report_tables_stage6c.py` — report table 测试

### 修改文件
- `src/vlm_nsm/experiments/run_language_variation.py`
  - 新增 `_check_schema_validity()` 函数
  - 结果字段新增 `alignment_applicable`、`schema_validity`
  - no_alignment 模式: `alignment_success=None`, `alignment_applicable=False`
  - 支持 "medium" prompt variant
  - summary 打印新增 schema_validity_rate 和 N/A 对齐率
- `src/vlm_nsm/experiments/summarize_language_variation.py`
  - 聚合新增 schema_validity_rate
  - no_alignment 显示 N/A
- `src/vlm_nsm/experiments/export_report_tables.py`
  - 各表格添加过滤逻辑
  - 支持 schema_validity_rate 列
  - no_alignment 显示 N/A
- `src/vlm_nsm/experiments/analyze_qwen_failures.py`
  - 新增 `schema_invalid` 分类
  - 增强 repair_stats 字段
- 多个测试文件更新以适配新字段

## 实验矩阵

### Qwen2.5-32B hard matrix

For each of hanoi/stacking:
- prompt_variant=full: full, no_alignment, no_repair, no_alias_repair, no_completeness_repair
- prompt_variant=medium: full, no_alignment, no_repair, no_alias_repair, no_completeness_repair
- prompt_variant=weak: full

### Qwen2.5-14B hard matrix

For each of hanoi/stacking:
- prompt_variant=full: full
- prompt_variant=medium: full, no_alignment
- prompt_variant=weak: full

## 资源需求

- Qwen2.5-32B-Instruct: ~64GB fp16, 需要 4×4090 (device_map="auto")
- Qwen2.5-14B-Instruct: ~28GB fp16, 可在单卡运行
- Mock parser: 无需 GPU

## Qwen2.5-14B 路径检查

模型路径: /rhome/public/model/Qwen2.5-14B-Instruct
如果不存在或显存不足，记录日志并跳过，保留 32B 结果为主结果。
