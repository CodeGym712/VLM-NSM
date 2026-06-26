# DashScope Qwen API Setup

This project uses `OnlineQwenGoalParser` as an OpenAI-compatible chat client.
For Alibaba Cloud Model Studio / DashScope compatible mode, keep the API key
outside the repository and export it in your shell.

## 1. Install Client Dependency

```bash
pip install "openai>=1.30.0"
```

## 2. Configure Environment

Copy the template and fill the key in your private shell session:

```bash
cp configs/dashscope_qwen.env.example /tmp/vlm_nsm_dashscope.env
# edit /tmp/vlm_nsm_dashscope.env and fill QWEN_API_KEY
source /tmp/vlm_nsm_dashscope.env
```

Default endpoint:

```bash
export QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Set `QWEN_MODEL` to a model enabled in your Alibaba Cloud account, for example:

```bash
export QWEN_MODEL="qwen-plus"
```

If you use a Qwen3.6 model, record that as a separate cloud-backend condition;
do not merge it with prior offline Qwen2.5-32B results.

## 3. Smoke Test

```bash
bash scripts/smoke_dashscope_qwen.sh
```

Expected: a short response preview and no authentication/model errors.

## 4. Parse-Only Evaluation

```bash
bash scripts/run_dashscope_parse5.sh
```

Output:

```text
generated/results/stage_15/dashscope_qwen_parse_5/summary.json
```

## 5. E2E Demo

Requires Robosuite/GPU runtime:

```bash
MUJOCO_GL=egl bash scripts/run_dashscope_e2e_demo.sh
```

Output:

```text
generated/results/stage_15/e2e_dashscope_qwen_parser_driven/
```

## Notes

- `QWEN_API_KEY` is intentionally not stored in the repository.
- `OnlineQwenGoalParser` can talk to DashScope or any local OpenAI-compatible
  endpoint.
- The default project parser remains `mock`; cloud Qwen must be requested
  explicitly with `--parser online_qwen`.
