from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _squared_distance(a: list[float], b: list[float]) -> float:
    count = min(len(a), len(b))
    if count == 0:
        return 0.0
    return sum((a[idx] - b[idx]) ** 2 for idx in range(count)) / count


def _selected_obs_from_episode(episode: dict) -> list[float]:
    first_step = episode["steps"][0]
    return first_step["robot_obs"][:9] + first_step.get("scene_obs", [])[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-log", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--top-k-per-subtask",
        type=int,
        default=6,
        help="Number of nearest teacher episodes to keep for each native subtask.",
    )
    args = parser.parse_args()

    native_payload = _load_json(args.native_log)
    rollout_payload = _load_json(args.rollout_path)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    teacher_episodes = rollout_payload["episodes"]
    selected_episode_indices: set[int] = set()
    selections: list[dict[str, object]] = []

    for sequence in native_payload.get("sequence_results", []):
        for native_subtask in sequence.get("subtask_results", []):
            native_initial_obs = native_subtask.get("initial_selected_obs", [])
            subtask = native_subtask["subtask"]
            instruction = native_subtask["language_annotation"]
            candidates: list[dict[str, object]] = []
            for episode in teacher_episodes:
                if episode.get("task_name") != subtask:
                    continue
                teacher_initial_obs = _selected_obs_from_episode(episode)
                candidates.append(
                    {
                        "teacher_sequence_index": episode["sequence_index"],
                        "teacher_source_sequence_index": episode.get("source_sequence_index"),
                        "instruction": episode.get("instruction"),
                        "distance_mse": _squared_distance(native_initial_obs, teacher_initial_obs),
                    }
                )
            candidates.sort(key=lambda item: item["distance_mse"])
            kept = candidates[: args.top_k_per_subtask]
            for item in kept:
                selected_episode_indices.add(int(item["teacher_sequence_index"]))
            selections.append(
                {
                    "native_sequence_index": sequence["sequence_index"],
                    "subtask": subtask,
                    "language_annotation": instruction,
                    "top_matches": kept,
                }
            )

    selected_episodes = [
        episode for episode in teacher_episodes if episode["sequence_index"] in selected_episode_indices
    ]
    selected_episodes.sort(key=lambda item: item["sequence_index"])
    remapped_episodes = []
    for remapped_index, episode in enumerate(selected_episodes):
        episode_copy = dict(episode)
        episode_copy["matched_sequence_index"] = episode["sequence_index"]
        episode_copy["sequence_index"] = remapped_index
        remapped_episodes.append(episode_copy)

    payload = {
        "dataset_path": rollout_payload["dataset_path"],
        "split": rollout_payload["split"],
        "lang_folder": rollout_payload["lang_folder"],
        "action_key": rollout_payload["action_key"],
        "horizon": rollout_payload["horizon"],
        "task_filter": rollout_payload.get("task_filter"),
        "source_rollout_path": str(Path(args.rollout_path).resolve()),
        "matching_source_native_log": str(Path(args.native_log).resolve()),
        "top_k_per_subtask": args.top_k_per_subtask,
        "episodes": remapped_episodes,
    }
    summary = {
        "source_rollout_path": str(Path(args.rollout_path).resolve()),
        "matching_source_native_log": str(Path(args.native_log).resolve()),
        "top_k_per_subtask": args.top_k_per_subtask,
        "selected_sequences": len(remapped_episodes),
        "selected_total_steps": sum(len(ep["steps"]) for ep in remapped_episodes),
        "selected_task_names": sorted({ep.get("task_name") for ep in remapped_episodes}),
        "selections": selections,
    }

    write_json(output_path, payload)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
