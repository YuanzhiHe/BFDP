from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_adapter import load_calvin_export, summarize_calvin_export
from svh_dp.utils.common import ensure_dir, write_json


def _selected_obs(episode: dict) -> list[float]:
    first_step = episode["steps"][0]
    return first_step["robot_obs"][:9] + first_step.get("scene_obs", [])[:24]


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _wrapped_distance(a: list[float], b: list[float]) -> float:
    count = min(len(a), len(b))
    if count == 0:
        return 0.0
    diffs = []
    for idx in range(count):
        if idx in {3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}:
            diffs.append(_wrapped_delta(a[idx], b[idx]))
        else:
            diffs.append(a[idx] - b[idx])
    return sum(value * value for value in diffs) / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-rollout-path", required=True)
    parser.add_argument("--source-rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--top-k-per-anchor",
        type=int,
        default=8,
        help="How many same-task nearest full episodes to keep for each anchor episode, including the anchor itself.",
    )
    args = parser.parse_args()

    anchor_payload = load_calvin_export(args.anchor_rollout_path)
    source_payload = load_calvin_export(args.source_rollout_path)

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    source_episodes = source_payload["episodes"]
    source_by_task: dict[str, list[dict]] = {}
    for episode in source_episodes:
        task_name = episode.get("task_name")
        if not task_name:
            continue
        source_by_task.setdefault(task_name, []).append(episode)

    selected_by_source_index: dict[int, dict] = {}
    neighborhood_specs: list[dict[str, object]] = []
    for anchor_episode in anchor_payload["episodes"]:
        anchor_task = anchor_episode.get("task_name")
        anchor_source_index = anchor_episode.get("source_sequence_index")
        anchor_obs = _selected_obs(anchor_episode)
        candidates = []
        for candidate in source_by_task.get(anchor_task, []):
            candidate_source_index = candidate.get("source_sequence_index")
            if not isinstance(candidate_source_index, int):
                continue
            candidates.append(
                {
                    "source_sequence_index": candidate_source_index,
                    "instruction": candidate.get("instruction"),
                    "task_name": anchor_task,
                    "distance_mse": _wrapped_distance(anchor_obs, _selected_obs(candidate)),
                    "episode": candidate,
                }
            )
        candidates.sort(key=lambda item: (item["distance_mse"], item["source_sequence_index"]))
        kept = candidates[: max(1, args.top_k_per_anchor)]
        neighborhood_specs.append(
            {
                "anchor_source_sequence_index": anchor_source_index,
                "anchor_instruction": anchor_episode.get("instruction"),
                "task_name": anchor_task,
                "neighbors": [
                    {
                        "source_sequence_index": item["source_sequence_index"],
                        "instruction": item["instruction"],
                        "distance_mse": item["distance_mse"],
                    }
                    for item in kept
                ],
            }
        )
        for item in kept:
            selected_by_source_index[item["source_sequence_index"]] = item["episode"]

    selected_episodes = [
        selected_by_source_index[source_sequence_index]
        for source_sequence_index in sorted(selected_by_source_index)
    ]

    remapped = []
    for export_index, episode in enumerate(selected_episodes):
        episode_copy = dict(episode)
        episode_copy["sequence_index"] = export_index
        remapped.append(episode_copy)

    export_payload = {
        "dataset_path": source_payload["dataset_path"],
        "split": source_payload["split"],
        "lang_folder": source_payload["lang_folder"],
        "action_key": source_payload["action_key"],
        "horizon": source_payload["horizon"],
        "export_mode": source_payload.get("export_mode", "full"),
        "task_filter": sorted({episode.get("task_name") for episode in remapped if episode.get("task_name")}),
        "selection_mode": "official_sequence_neighborhood",
        "selection_anchor_rollout_path": str(Path(args.anchor_rollout_path).resolve()),
        "selection_source_rollout_path": str(Path(args.source_rollout_path).resolve()),
        "top_k_per_anchor": args.top_k_per_anchor,
        "selection_neighborhoods": neighborhood_specs,
        "episodes": remapped,
    }
    write_json(output_path, export_payload)

    summary_payload = summarize_calvin_export(export_payload, output_path=output_path).to_dict()
    summary_payload["selection_mode"] = export_payload["selection_mode"]
    summary_payload["top_k_per_anchor"] = args.top_k_per_anchor
    summary_payload["selection_anchor_rollout_path"] = export_payload["selection_anchor_rollout_path"]
    summary_payload["selection_source_rollout_path"] = export_payload["selection_source_rollout_path"]
    summary_payload["selection_neighborhoods"] = neighborhood_specs
    write_json(summary_path, summary_payload)

    print(f"saved_rollouts={output_path}")
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
