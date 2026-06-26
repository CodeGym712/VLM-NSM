# Migration Audit Report (Redacted)

> Original audit: 2026-06-26; redacted for repository publication: 2026-06-27

## Outcome

The project uses a split development workflow:

- Windows/local: source editing plus pure-Python unit and symbolic tests.
- GitHub: source review and synchronization.
- GPU server: simulation, vision, VLM, training, and generated-artifact validation.

## Security finding

The original report quoted a live DashScope credential while documenting two historical hard-coded credential findings. The owner has revoked that credential. All literal credential values and examples have been removed from this published report; only environment-variable based configuration is permitted.

Online Qwen/DashScope calls read `QWEN_API_KEY`. Never place its value in source files, reports, committed shell history, or example configuration.

## Portability findings

- Python 3.10 is the supported local baseline.
- The minimal dependency set intentionally excludes PyTorch, HDF5, MuJoCo/robosuite, vision packages, local Qwen weights, and generated experiment results.
- A local test profile collects only verified pure-Python modules; marker filtering alone cannot prevent import-time failures from optional dependencies.
- Repository and configuration paths are resolved from the checkout or supplied explicitly through environment variables.

## Verification record

The Windows-localization change adds regression checks for credential-shaped values, private server roots, manifest validity, marker assignment, and the curated local suite. Full simulation and model validation remains a server responsibility.
