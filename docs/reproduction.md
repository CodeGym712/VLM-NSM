# 分层复现指南

本指南针对 GitHub 当前公开实现：语言目标解析、符号对齐与规划、robosuite 堆叠。它不是本地 CALVIN / VLA-Adapter 研究工作区的安装说明。

## 选择复现层级

| 层级 | 目标 | 主要依赖 | 可以验证什么 |
|---|---|---|---|
| A | 轻量本地测试 | Windows、Python 3.10、项目基础依赖 | 所选纯 Python / symbolic 测试 |
| B | mock 语言与符号流程 | 项目安装、语言变体输入 | 接口连通与符号流程，不代表语言模型能力 |
| C | 真实语言模型评测 | 模型运行依赖与本地权重，或 API 授权 | 对应模型、提示和输入集下的解析表现 |
| D | 物理仿真 | PyTorch、robosuite/MuJoCo、控制器 checkpoint 与配置 | 对应配置下的闭环堆叠结果 |

先完成适合本机的层级，不要把轻量测试通过解释为 GPU / 仿真验收通过。

## A：轻量本地测试

按 [Windows 环境说明](local-development.md) 创建 Python 3.10 环境并安装项目：

```powershell
python -m pip install -e .
python scripts\run_local_tests.py -q
```

使用明确列出测试模块的 runner。不要将仓库全量测试默认视为轻量测试：可选依赖和未提交实验产物可能在收集阶段就被导入。

记录实际解释器版本、依赖、退出码与测试摘要。历史报告的 passed 数量不是新环境的预期保证。

## B：mock 符号流程

```bash
python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser mock --difficulty easy
```

可替换为 `--task stacking` 或 `--difficulty hard`。检查生成的原始解析、目标谓词、规划与最终目标检查字段，而不只看程序是否退出。

mock 不调用真实语言模型；其结果只能说明相应模拟解析路径可运行。规划器和目标检查的具体语义应结合源码确认，不能把固定任务解法的成功全部归因于语言理解。

## C：本地 Qwen 或在线模型

本地 Qwen 示例：

```bash
python -m vlm_nsm.experiments.run_language_variation \
  --task hanoi --n_blocks 3 --parser qwen_text \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --model_name qwen25_14b --difficulty hard --max_samples 1
```

先验证一个样本，再扩大输入集。最小安装不包含完整模型栈；按照所选模型与项目实际运行环境配置。此指南不保证某个模型在指定显卡上可直接加载。

每次运行记录：
- 代码提交、模型精确版本、提示模板、数据文件；
- parser、prompt variant、alignment mode、是否 fallback；
- 原始输出、解析错误、规划结果、目标成功定义；
- 有效样本数、失败数、缺失样本与耗时。

在线方式见 [集成说明](stage_14_online_qwen_integration.md) 和 [演示指南](demo_guide.md)。API 凭证通过环境提供，不写入仓库。mock、缓存输出、真实在线调用及 fallback 分开汇总；不同模型之间的结果不能直接作为同模型消融。

## Simulation and artifact checklist

进入层级 D 之前检查：

| 输入 | 文档指定路径 / 依据 |
|---|---|
| 控制器 checkpoint | `generated/results/stage_7B-4/dagger_reach_bc/best.pt` |
| 执行配置 | `generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json` |
| 运行环境 | PyTorch、robosuite 1.5.2、兼容 MuJoCo 依赖；见 runtime setup |
| 策略阶段 | 默认仅 `reach_pick` 由 BC 控制，其余阶段 scripted |
| 评测入口 | `eval_bc_hybrid` 与共享 episode runner |
| 无头渲染 | 与实际环境匹配的 `MUJOCO_GL` 设置 |

这些路径是所需产物的位置，不代表它们已随 Git 提供。缺少权重、数据或配置时，先按 [产物清单](artifact_manifest.md) 核实获取方式；没有获取方式就记录为复现阻塞，不生成占位权重。

准备好环境与产物后，参考既有入口：

```bash
bash scripts/run_canonical_demo.sh
bash scripts/run_canonical_eval_20ep.sh
```

前者包含 mock 解析；20 次评测适合检查流程，不保证落在历史成功率范围内。视频录制见 [演示指南](demo_guide.md)，仿真依赖见 [runtime setup](robosuite_runtime_setup.md)。

## 运行记录与提交边界

至少保留：代码版本、环境、checkpoint 身份、配置、parser、控制阶段、episode 数、实际随机初始化方法、原始结果与退出码。不能仅凭 `--seed` 参数声称环境已确定性固定。

对外提交配置、脚本、小型结果和可公开演示。大权重与数据按各自条款单独托管或提供获取步骤。当前指南没有重新运行模型或仿真，不能替代运行验收。
