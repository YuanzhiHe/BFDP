# BFDP: Diagnosing Benchmark-Faithful Evaluation in CALVIN

Official code release for the paper **"Diagnosing Benchmark-Faithful Evaluation in CALVIN: Export Truncation, Teacher Semantics, and Symbolic Reset Fidelity."**

BFDP is a diagnostic protocol for separating true policy weakness from benchmark-path corruption in long-horizon manipulation evaluation. This release contains the experiment code used for:

- export truncation diagnosis
- exact-reset teacher replay checks
- symbolic-reset failure localization
- local bridge training and evaluation utilities
- CALVIN / RLBench probing and adapter scripts

## Overview

Benchmark-native failure is not always policy failure. BFDP audits the path from exported demonstrations to native oracle-scored rollout and isolates three common failure sources:

1. export horizon corruption
2. teacher-semantic mismatch
3. symbolic reset invalidity

The repository is organized as:

- `core_code/config/`: default experiment configuration
- `core_code/scripts/`: runnable training, export, probe, replay, and analysis scripts
- `core_code/svh_dp/`: local package for training, evaluation, and benchmark adapters
- `core_code/vendor/`: bundled supporting code used by the local pipeline

## Setup

The included `requirements.txt` covers the Python packages directly imported by the released code:

- `torch`
- `numpy`
- `PyYAML`
- `hydra-core`
- `omegaconf`

Recommended environment: Python `3.10+`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Optional Dependencies

Some scripts rely on external benchmark stacks that are intentionally not pinned here:

- CALVIN-related scripts require a local CALVIN workspace and dataset
- RLBench-related scripts require `RLBench` and `PyRep`
- RLBench support in this codepath is Linux-oriented

## Quick Start

```bash
python core_code/scripts/run_training_testing.py --variant full
python core_code/scripts/check_benchmark_backends.py
python core_code/scripts/export_rlbench_adapter_sample.py
python core_code/scripts/probe_calvin_native_eval.py --help
```

Default configuration:

```text
core_code/config/default.yaml
```

## Release Scope

This repository intentionally excludes large or local-only artifacts such as:

- datasets
- logs
- checkpoints
- private benchmark workspaces
- experiment analysis notes

`.gitignore` is configured to prevent future accidental uploads of those assets.
