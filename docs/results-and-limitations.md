# 结果口径与已知限制

本页区分当前公开仓库的历史报告、可执行入口与尚未核实的结论。它不改写原始实验文件，也不引入本地 CALVIN 项目的结果。

## 1. 仿真成绩的具体范围

[Stage 10 表](../generated/report_tables/stage_10/deployment_policy_comparison.md) 报告：
- scripted-only：100 episodes，88.0%；
- DAgger reach-only hybrid：100 episodes，92.0%；
- reach-only 三次 50-episode 运行：96%、90%、94%，平均 93.3%。

这里的 hybrid 仅在 `reach_pick` 阶段使用 BC，其余阶段使用脚本。不能将它描述为全流程学习控制、端到端 VLA 或真机部署。

[最终策略表](../generated/report_tables/final/final_embodied_policy_comparison.md) 与 Stage 10 表的 seed 行、步数及评测规模表述不同，不能将不同表混为一次统一实验。本次文档更新只引用明确指向的 Stage 10 记录，没有独立复算原始轨迹。

[仿真协议](simulation_protocol.md) 明确说当前 `--seed` 是元数据，未固定 robosuite 初始化。因此：
- “三次运行”不等于三个训练种子；
- 不能由 seed 标签推断确定性或配对初态；
- 88% 的 scripted 成绩不是已证明的仿真性能上限，也不能推出环境固有失败率为 12%；
- 92% 与 88% 的表面差异不能单独证明统计显著的提升。

## 2. 语言结果暂不用于首页总准确率

[发布摘要](../generated/report_tables/final/final_v0_1_rc1_summary.md) 记录 984 samples / 556 failures；但 [最终模型比较表](../generated/report_tables/final/final_language_model_comparison.md) 的 Samples 均为 0，准确率列为空。

这表示现有聚合材料存在需要回查的口径或导出问题。未检查原始样本与导出逻辑前：
- 不据此发布按模型划分的准确率；
- 不从“批处理任务完成”推出“样本任务成功”；
- 不把总评估行数解释成互不重复的语言指令数；
- 不跨 prompt、alignment mode、模型和任务直接合并成绩。

后续应按 run 身份回查逐样本记录、有效分母、重复与错误分类，再重新生成汇总。此文档变更不修复或猜测原始计数。

## 3. 5/5 演示使用 mock 解析器

[端到端演示报告](../generated/report_tables/stage_10/e2e_demo_summary.md) 明确列出 `MockGoalParser`，五条指令对应相同堆叠目标。

它支持的是该 mock→对齐→规划→混合执行流程的样例集成结果，不支持真实 Qwen 端到端成功率、开放语言泛化或五种不同操控技能。mock、cached、online 及 fallback 结果必须分别报告。

## 4. RGB 接地不等于全流程视觉控制

[Stage 18 感知报告](stage_18_rgb_detector_pipeline.md) 指出三路比较使用了不同语言模型，存在混杂，不能据此证明检测器优于 oracle。

同一报告说明 RGB 推算位置主要用于场景接地与校验，低层控制仍使用仿真状态或已有控制器。没有完整输入审计前，不将该流程描述为“全程无 oracle”或“纯视觉闭环控制”。

## 5. 演示文件与复现产物

已提供录制代码与 [媒体清单](../assets/demo/v0_1_1/demo_media_manifest.md)，但本次检查的仓库树未包含 MP4/PNG 演示文件。支持录制不等于已发布演示。

历史 checkpoint、数据与部分 generated 文件可能不在 Git 中。运行命令存在不等于全新 clone 已具备全部输入。详见 [复现指南](reproduction.md)。

## 6. 可公开陈述与待完成项

| 可以据当前材料陈述 | 尚不应声称 |
|---|---|
| 实现了语言目标结构化、符号对齐与规划接口 | 真实语言模型已稳定实现通用目标理解 |
| 提供 robosuite 混合控制与历史结果报告 | 全流程学习控制或真机验证 |
| 存在技能命令接口和可选执行路径 | 新技能学习策略已取代默认控制器 |
| 存在 RGB 接地与几何分析工作 | 同模型、无混杂的完整视觉消融已完成 |
| 提供本地轻量测试入口 | 当前完整仿真／模型测试已在本次更新中通过 |

后续展示优先补充：可获取的产物、真实视频、可追溯逐次评测、修正后的语言汇总，以及与实际初始化一致的重复实验说明。
