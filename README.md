# BFDP: Benchmark-Path Fidelity in Parallel Robot-Policy Evaluation

Official code release for the paper **"Benchmark-Path Fidelity in Parallel Robot-Policy Evaluation: A CALVIN Case Study."**

BFDP is a staged, cheap-first set of single-worker controls, run before a parallel evaluation campaign, that separates defects in a project's own evaluation path from artefacts of the diagnostic controls themselves. This release contains the experiment code used for:

- export truncation diagnosis
- exact-reset teacher replay checks
- start-state mismatch localisation on the symbolic path
- local bridge training and evaluation utilities
- CALVIN / RLBench probing and adapter scripts

## Overview

Benchmark-native failure is not always policy failure, and it is not always a benchmark problem either. BFDP audits the path from exported demonstrations to native oracle-scored rollout, and in this case study it separates three findings of three different kinds:

1. **A confirmed export-path truncation defect.** A fixed 8-step export ends before the effect-producing part of the recorded episode. In a paired intervention on the same 14 episodes, with export horizon as the only changed factor, exact-reset teacher replay moves from 0/14 to 14/14.

2. **Start-state sensitivity of open-loop teacher replay under the documented neutral reset.** The symbolic path returns a single canonical `robot_obs` for all 64 probed symbolic conditions. This is *not* a benchmark defect: CALVIN resets the robot to a neutral position at the start of every evaluation sequence by design, "to avoid biasing the policies through the robot's initial pose". The exact-versus-symbolic split therefore measures how sensitive an open-loop teacher program is to the start state it is replayed from.

3. **A limitation of open-loop teacher replay under the official chained protocol.** Exact continuous replay reaches 1/3 on an official three-subtask probe. Executing each subtask from the terminal state of the previous one is the official rule, so a recorded open-loop program is expected to degrade along a chain even under a perfectly faithful harness. A teacher score below 1.0 is not evidence of a defect.

**Only the first of these is a defect, and it lies in this project's own evaluation path, not in CALVIN.** An earlier version of this work described findings 2 and 3 as benchmark defects; that attribution was incorrect and has been withdrawn.

The repository is organized as:

- `src/config/`: default experiment configuration
- `src/scripts/`: runnable training, export, probe, replay, and analysis scripts
- `src/svh_dp/`: local package for training, evaluation, and benchmark adapters
- `src/vendor/`: bundled supporting code used by the local pipeline

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
python src/scripts/run_training_testing.py --variant full
python src/scripts/check_benchmark_backends.py
python src/scripts/export_rlbench_adapter_sample.py
python src/scripts/probe_calvin_native_eval.py --help
```

Default configuration:

```text
src/config/default.yaml
```

## Reproducibility

A paper-specific release tag pinning the CALVIN, PyBullet, Python, CUDA and PyTorch versions, and publishing the training seeds, dataset version, episode identifiers and per-table run configurations, is in preparation.

## Release Scope

This repository intentionally excludes large or local-only artifacts such as:

- datasets
- logs
- checkpoints
- private benchmark workspaces
- experiment analysis notes

`.gitignore` is configured to prevent future accidental uploads of those assets.
