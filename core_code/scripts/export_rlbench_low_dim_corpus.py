from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.rlbench_adapter import (
    collect_low_dim_rollouts,
    load_rollout_export,
    merge_rollout_exports,
)
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_task_tag(tasks: list[str], max_readable_tasks: int = 4) -> str:
    lowered = [task.lower() for task in tasks]
    if len(lowered) <= max_readable_tasks:
        return "-".join(lowered)
    readable_prefix = "-".join(lowered[:max_readable_tasks])
    digest = hashlib.sha1(",".join(lowered).encode("utf-8")).hexdigest()[:8]
    return f"{readable_prefix}-plus{len(lowered) - max_readable_tasks}-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--num-shards", type=int, help="Override number of export shards.")
    parser.add_argument("--episodes-per-shard", type=int, help="Override episodes collected per shard.")
    parser.add_argument("--horizon", type=int, help="Override rollout horizon.")
    parser.add_argument("--policy", choices=["zero", "random"], help="Override collection policy.")
    parser.add_argument("--base-seed", type=int, help="Override shard base seed.")
    parser.add_argument("--output-dir", help="Override output directory for corpus exports.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task list override for multi-task corpus export.",
    )
    parser.add_argument(
        "--tasks-from-probe",
        nargs="+",
        help="Optional probe JSON path(s); successful_tasks will be merged when --tasks is omitted.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    rlbench_cfg = config["benchmarks"]["rlbench"]
    corpus_cfg = rlbench_cfg["corpus"]
    output_dir = ensure_dir(args.output_dir or corpus_cfg["output_dir"])
    num_shards = corpus_cfg["num_shards"] if args.num_shards is None else args.num_shards
    episodes_per_shard = (
        corpus_cfg["episodes_per_shard"]
        if args.episodes_per_shard is None
        else args.episodes_per_shard
    )
    horizon = corpus_cfg["rollout_horizon"] if args.horizon is None else args.horizon
    policy = corpus_cfg["policy"] if args.policy is None else args.policy
    base_seed = corpus_cfg["base_seed"] if args.base_seed is None else args.base_seed
    if args.tasks is not None:
        tasks = args.tasks
    elif args.tasks_from_probe is not None:
        tasks = []
        for probe_path in args.tasks_from_probe:
            with open(probe_path, "r", encoding="utf-8") as handle:
                tasks.extend(json.load(handle)["successful_tasks"])
    else:
        tasks = corpus_cfg.get("tasks", [rlbench_cfg["task"]])
    tasks = dedupe_preserve_order(tasks)

    task_tag = build_task_tag(tasks)
    run_tag = (
        f"{task_tag}_{rlbench_cfg['arm_action_mode'].lower()}_"
        f"{policy}_{len(tasks)}t_{num_shards}x{episodes_per_shard}_h{horizon}"
    )
    shard_paths = []
    shard_summaries = []
    shard_payloads = []

    global_shard_index = 0
    for task in tasks:
        for shard_index in range(num_shards):
            shard_path = output_dir / f"{run_tag}_{task.lower()}_shard_{shard_index:02d}.json"
            summary = collect_low_dim_rollouts(
                rlbench_cfg=rlbench_cfg,
                adapter_cfg=rlbench_cfg["adapter"],
                output_path=shard_path,
                task=task,
                episodes=episodes_per_shard,
                horizon=horizon,
                policy=policy,
                seed=base_seed + global_shard_index,
            )
            shard_paths.append(str(shard_path))
            shard_summaries.append(summary.to_dict())
            shard_payloads.append(load_rollout_export(shard_path))
            global_shard_index += 1

    corpus_path = output_dir / f"{run_tag}_corpus.json"
    summary_path = output_dir / f"{run_tag}_summary.json"
    manifest_path = output_dir / f"{run_tag}_manifest.json"
    merged_payload, merged_summary = merge_rollout_exports(
        payloads=shard_payloads,
        output_path=corpus_path,
    )
    write_json(corpus_path, merged_payload)
    write_json(summary_path, merged_summary.to_dict())
    write_json(
        manifest_path,
        {
            "task_name": tasks[0] if len(tasks) == 1 else tasks,
            "task_names": tasks,
            "arm_action_mode": rlbench_cfg["arm_action_mode"],
            "policy": policy,
            "num_tasks": len(tasks),
            "num_shards": num_shards,
            "episodes_per_shard": episodes_per_shard,
            "rollout_horizon": horizon,
            "base_seed": base_seed,
            "corpus_path": str(corpus_path),
            "summary_path": str(summary_path),
            "shard_paths": shard_paths,
            "shard_summaries": shard_summaries,
        },
    )

    print(f"saved_corpus={corpus_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_manifest={manifest_path}")
    print(merged_summary.to_dict())


if __name__ == "__main__":
    main()
