from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    wrapped_indices = {
        3, 4, 5,
        12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29,
        30, 31, 32,
    }
    for idx in range(count):
        if idx in wrapped_indices:
            diffs.append(_wrapped_delta(a[idx], b[idx]))
        else:
            diffs.append(a[idx] - b[idx])
    return sum(value * value for value in diffs) / count


def _prefix_sign_pattern(episode: dict, prefix_len: int) -> list[int]:
    prefix = []
    steps = episode.get("steps", [])
    if not steps:
        return prefix
    final_sign = 1 if steps[-1]["action"][-1] >= 0.0 else -1
    for step_idx in range(prefix_len):
        if step_idx < len(steps):
            value = steps[step_idx]["action"][-1]
            prefix.append(1 if value >= 0.0 else -1)
        else:
            prefix.append(final_sign)
    return prefix


def _hamming_distance(a: list[int], b: list[int]) -> int:
    count = min(len(a), len(b))
    distance = sum(1 for idx in range(count) if a[idx] != b[idx])
    distance += abs(len(a) - len(b))
    return distance


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summarize(payload: dict, output_path: Path) -> dict:
    episodes = payload["episodes"]
    task_counts: dict[str, int] = {}
    for episode in episodes:
        task_name = episode.get("task_name", "unknown")
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
    return {
        "selection_mode": payload["selection_mode"],
        "selection_anchor_rollout_path": payload["selection_anchor_rollout_path"],
        "selection_source_rollout_path": payload["selection_source_rollout_path"],
        "prefix_len": payload["prefix_len"],
        "top_k_per_anchor": payload["top_k_per_anchor"],
        "sequences": len(episodes),
        "total_steps": sum(len(episode.get("steps", [])) for episode in episodes),
        "task_names": sorted(task_counts),
        "task_counts": task_counts,
        "output_path": str(output_path),
        "selection_specs": payload["selection_specs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-rollout-path", required=True)
    parser.add_argument("--source-rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--top-k-per-anchor", type=int, default=8)
    args = parser.parse_args()

    anchor_payload = _load(args.anchor_rollout_path)
    source_payload = _load(args.source_rollout_path)
    source_by_task: dict[str, list[dict]] = {}
    for episode in source_payload.get("episodes", []):
        task_name = episode.get("task_name")
        if task_name:
            source_by_task.setdefault(task_name, []).append(episode)

    selected_by_key: dict[tuple[str, str], dict] = {}
    selection_specs: list[dict] = []
    for anchor_episode in anchor_payload.get("episodes", []):
        task_name = anchor_episode.get("task_name")
        if not task_name:
            continue
        anchor_obs = _selected_obs(anchor_episode)
        anchor_prefix = _prefix_sign_pattern(anchor_episode, args.prefix_len)
        candidates = []
        for candidate in source_by_task.get(task_name, []):
            candidate_prefix = _prefix_sign_pattern(candidate, args.prefix_len)
            candidates.append(
                {
                    "source_sequence_index": candidate.get("source_sequence_index"),
                    "instruction": candidate.get("instruction"),
                    "task_name": task_name,
                    "prefix_hamming": _hamming_distance(anchor_prefix, candidate_prefix),
                    "distance_mse": _wrapped_distance(anchor_obs, _selected_obs(candidate)),
                    "prefix_pattern": candidate_prefix,
                    "episode": candidate,
                }
            )
        candidates.sort(
            key=lambda item: (
                item["prefix_hamming"],
                item["distance_mse"],
                item["source_sequence_index"],
            )
        )
        kept = candidates[: max(1, args.top_k_per_anchor)]
        selection_specs.append(
            {
                "anchor_source_sequence_index": anchor_episode.get("source_sequence_index"),
                "anchor_instruction": anchor_episode.get("instruction"),
                "task_name": task_name,
                "anchor_prefix_pattern": anchor_prefix,
                "neighbors": [
                    {
                        "source_sequence_index": item["source_sequence_index"],
                        "instruction": item["instruction"],
                        "prefix_hamming": item["prefix_hamming"],
                        "distance_mse": item["distance_mse"],
                        "prefix_pattern": item["prefix_pattern"],
                    }
                    for item in kept
                ],
            }
        )
        for item in kept:
            key = (task_name, str(item["source_sequence_index"]))
            selected_by_key[key] = item["episode"]

    selected_episodes = sorted(
        selected_by_key.values(),
        key=lambda episode: (
            episode.get("task_name", ""),
            episode.get("source_sequence_index", -1),
        ),
    )
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
        "selection_mode": "official_sequence_behavior_matched",
        "selection_anchor_rollout_path": str(Path(args.anchor_rollout_path).resolve()),
        "selection_source_rollout_path": str(Path(args.source_rollout_path).resolve()),
        "prefix_len": args.prefix_len,
        "top_k_per_anchor": args.top_k_per_anchor,
        "selection_specs": selection_specs,
        "episodes": remapped,
    }

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    summary_payload = _summarize(export_payload, output_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
