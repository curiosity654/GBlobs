# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `pcdet/`. The main areas are `pcdet/datasets/` for dataset loaders and preprocessing, `pcdet/models/` for detectors, heads, and backbones, `pcdet/ops/` for custom CUDA extensions, and `pcdet/utils/` for shared helpers. Runtime entrypoints are in `tools/`, including `tools/train.py`, `tools/test.py`, dataset configs in `tools/cfgs/`, and launcher scripts in `tools/scripts/`. Supporting docs and figures live in `docs/` and container setup is under `docker/`.

## Build, Test, and Development Commands
Install dependencies and build the package in editable mode:
```bash
pip install -r requirements.txt
python setup.py develop
```
Run single-GPU training or evaluation from the repo root:
```bash
python tools/train.py --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml
python tools/test.py --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml --ckpt path/to/checkpoint.pth
```
For distributed runs, use the provided wrappers such as `tools/scripts/torch_train.sh`, `tools/scripts/dist_train.sh`, and `tools/scripts/slurm_train_v2.sh`. Generate dataset metadata with the dataset-specific module commands documented in `docs/GETTING_STARTED.md`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise module names like `gblobs_vfe.py` or `waymo_dataset.py`. Keep imports grouped by standard library, third-party, then local modules. Match the surrounding style before refactoring; this codebase does not define a repo-wide formatter or linter config.

## Testing Guidelines
There is no dedicated unit-test suite in this repository. Validate changes with the smallest realistic workflow: dataset info generation for data changes, `python tools/train.py ...` for training paths, and `python tools/test.py ... --ckpt ...` for evaluation paths. When touching CUDA ops in `pcdet/ops/`, rebuild with `python setup.py develop` and run a targeted training or eval smoke test.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Update README.md` and `Single config for train/eval`. Keep commit titles brief, focused, and under roughly 72 characters. For pull requests, include the purpose, affected configs or datasets, exact commands run, and before/after metrics when behavior changes. Link related issues or papers, and add screenshots only when updating docs or visual outputs.

## Configuration & Data Tips
Treat dataset paths, checkpoints, and large generated artifacts as local environment state, not source-controlled files. Keep new experiment configs in the closest matching subfolder of `tools/cfgs/` and prefer extending existing YAML patterns instead of creating one-off layouts.
