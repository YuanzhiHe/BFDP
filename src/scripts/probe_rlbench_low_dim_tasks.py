from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.rlbench_adapter import collect_low_dim_rollouts
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task list override for probing.",
    )
    parser.add_argument(
        "--tasks-from-catalog",
        help="Optional catalog JSON path; task_names will be used when --tasks is omitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of tasks to probe after task selection.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional starting index applied after task selection.",
    )
    parser.add_argument("--episodes", type=int, help="Override episodes per probe task.")
    parser.add_argument("--horizon", type=int, help="Override horizon per probe task.")
    parser.add_argument("--policy", choices=["zero", "random"], help="Override probe policy.")
    parser.add_argument("--base-seed", type=int, help="Override base seed for probes.")
    parser.add_argument("--output", help="Optional output path for the probe report.")
    args = parser.parse_args()

    config = load_config(args.config).data
    rlbench_cfg = config["benchmarks"]["rlbench"]
    probe_cfg = rlbench_cfg["probe"]
    if args.tasks is not None:
        tasks = args.tasks
    elif args.tasks_from_catalog is not None:
        with open(args.tasks_from_catalog, "r", encoding="utf-8") as handle:
            tasks = json.load(handle)["task_names"]
    else:
        tasks = probe_cfg["candidate_tasks"]
    if args.start_index:
        tasks = tasks[args.start_index :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    episodes = probe_cfg["episodes"] if args.episodes is None else args.episodes
    horizon = probe_cfg["horizon"] if args.horizon is None else args.horizon
    policy = probe_cfg["policy"] if args.policy is None else args.policy
    base_seed = probe_cfg["base_seed"] if args.base_seed is None else args.base_seed

    log_dir = ensure_dir(config["paths"]["log_dir"])
    output_path = Path(args.output) if args.output else log_dir / "rlbench_low_dim_task_probe.json"
    tmp_dir = ensure_dir(log_dir / "rlbench_low_dim_task_probe_tmp")
    results = []
    successful_tasks = []

    for index, task in enumerate(tasks):
        probe_path = tmp_dir / f"{task.lower()}_probe.json"
        try:
            summary = collect_low_dim_rollouts(
                rlbench_cfg=rlbench_cfg,
                adapter_cfg=rlbench_cfg["adapter"],
                output_path=probe_path,
                task=task,
                episodes=episodes,
                horizon=horizon,
                policy=policy,
                seed=base_seed + index,
            )
            result = {
                "task_name": task,
                "success": True,
                "episodes": summary.episodes,
                "total_steps": summary.total_steps,
                "obs_dim": summary.obs_dim,
                "action_dim": summary.action_dim,
                "policy": summary.policy,
                "horizon": summary.horizon,
                "probe_output_path": str(probe_path),
            }
            successful_tasks.append(task)
        except Exception as exc:  # pragma: no cover - environment-dependent
            result = {
                "task_name": task,
                "success": False,
                "error": str(exc),
            }
        results.append(result)

    payload = {
        "candidate_tasks": tasks,
        "task_count": len(tasks),
        "start_index": args.start_index,
        "successful_tasks": successful_tasks,
        "failed_tasks": [item["task_name"] for item in results if not item["success"]],
        "policy": policy,
        "episodes": episodes,
        "horizon": horizon,
        "base_seed": base_seed,
        "results": results,
    }
    write_json(output_path, payload)
    print(f"saved_probe={output_path}")
    print(payload)


if __name__ == "__main__":
    main()
