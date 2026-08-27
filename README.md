# BFDP — Benchmark-Path Fidelity in Parallel Robot-Policy Evaluation: A CALVIN Case Study

Diagnostic code and probe scripts for the paper *Benchmark-Path Fidelity in Parallel
Robot-Policy Evaluation: A CALVIN Case Study* (under review, The Journal of Supercomputing).

## Overview

BFDP is a staged, cheap-first set of single-worker controls, run before a parallel
evaluation campaign, that separates defects in a project's own evaluation path from
artefacts of the diagnostic controls themselves. Applied to CALVIN scene D with a
modular vision-language-to-diffusion pipeline, it separates three findings of three
different kinds:

1. **A confirmed export-path truncation defect.** A fixed 8-step export ends before the
   effect-producing part of the recorded episode. In a paired intervention on the same
   14 episodes, with export horizon as the only changed factor, exact-reset teacher
   replay moves from 0/14 to 14/14.

2. **Start-state sensitivity of open-loop teacher replay under the documented neutral
   reset.** The symbolic path returns a single canonical `robot_obs` for all 64 probed
   symbolic conditions. This is *not* a benchmark defect: CALVIN resets the robot to a
   neutral position at the start of every evaluation sequence by design, "to avoid
   biasing the policies through the robot's initial pose". The exact-versus-symbolic
   split therefore measures how sensitive an open-loop teacher program is to the start
   state it is replayed from.

3. **A limitation of open-loop teacher replay under the official chained protocol.**
   Exact continuous replay reaches 1/3 on an official three-subtask probe. Executing
   each subtask from the terminal state of the previous one is the official rule, so a
   recorded open-loop program is expected to degrade along a chain even under a
   perfectly faithful harness. A teacher score below 1.0 is not evidence of a defect.

**Only the first of these is a defect, and it lies in the authors' evaluation path, not
in CALVIN.** An earlier version of this work described findings 2 and 3 as benchmark
defects; that attribution was incorrect and has been withdrawn.

## Reproducibility

A paper-specific release tag pinning the CALVIN, PyBullet, Python, CUDA and PyTorch
versions, and publishing the training seeds, dataset version, episode identifiers and
per-table run configurations, is in preparation.
