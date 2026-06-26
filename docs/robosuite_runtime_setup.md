# Robosuite Runtime Setup

## System Environment

- **Platform**: Linux 5.15.0-176-generic (x86_64)
- **Python**: system python3 (3.10) — robosuite is NOT installed in the uv venv
- **Robosuite**: 1.5.2 (installed system-wide at `/usr/lib/python3.10/site-packages/robosuite`)
- **MuJoCo**: 3.8.1 (bundled with robosuite)
- **GPU**: NVIDIA (8× GPU, headless rendering via `MUJOCO_GL=egl`)

## Why System Python?

The project uv venv (`uv run python`) cannot import robosuite because:
- robosuite has native MuJoCo bindings that were installed system-wide
- uv venv does not inherit system-site-packages by default

All robosuite-dependent experiments MUST use `python3` (system) directly.

## Running Commands

### Headless (offscreen, no display):

```bash
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 20
```

### With rendering:

Requires an X display. On a headless server, use `Xvfb`:
```bash
MUJOCO_GL=egl Xvfb :99 -screen 0 1024x768x24 &
DISPLAY=:99 python3 -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 5 --render
```

### Specific evaluation scripts:

```bash
# Grasp parameter search (quick)
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.search_grasp_params --quick

# Full stacking evaluation
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.run_embodied_stacking_robosuite --num_episodes 20

# Failure analysis
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.analyze_robosuite_failures --num_episodes 20

# Step budget analysis
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.analyze_robosuite_step_budget --num_episodes 10

# Execution param search
MUJOCO_GL=egl python3 -m vlm_nsm.experiments.search_robosuite_execution_params --quick
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `MUJOCO_GL` | `egl` | Headless GPU rendering (required on servers) |
| `DISPLAY` | `:99` | X display for `--render` mode (with Xvfb) |

## Import Guard Pattern

All robosuite-dependent experiments use this pattern:
```python
def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
        return True
    except ImportError:
        return False
```

At module level or in main, check `_try_import_robosuite()` and exit gracefully with a clear error message.

## Known Limitations

1. **uv venv without robosuite**: `uv run python` will fail for robosuite imports. Always use `python3` for robosuite experiments.
2. **MUJOCO_GL=egl required**: Without this, MuJoCO will try (and fail) to open an X display on headless servers.
3. **Panda robot only**: The BASIC composite controller config is calibrated for Panda. Other robots may need different configs.
4. **Stack env only**: The wrapper is designed for the `Stack` environment with 2 cubes (cubeA, cubeB). Other environments need adapter changes.
