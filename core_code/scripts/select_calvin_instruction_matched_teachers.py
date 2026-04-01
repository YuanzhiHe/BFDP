from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _token_set(text: str) -> set[str]:
    normalized = _normalize_text(text)
    return {token for token in normalized.split(" ") if token}


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


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
        help="Number of instruction-prioritized teacher episodes to keep for each native subtask.",
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
    task_counts: Counter[str] = Counter()
    exact_match_counts: Counter[str] = Counter()
    normalized_exact_counts: Counter[str] = Counter()

    for sequence in native_payload.get("sequence_results", []):
        for native_subtask in sequence.get("subtask_results", []):
            native_initial_obs = native_subtask.get("initial_selected_obs", [])
            subtask = native_subtask["subtask"]
            instruction = native_subtask["language_annotation"]
            normalized_native_instruction = _normalize_text(instruction)
            task_counts[subtask] += 1

            candidates: list[dict[str, object]] = []
            for episode in teacher_episodes:
                if episode.get("task_name") != subtask:
                    continue
                teacher_instruction = episode.get("instruction", "")
                teacher_initial_obs = _selected_obs_from_episode(episode)
                exact_match = teacher_instruction == instruction
                normalized_exact_match = _normalize_text(teacher_instruction) == normalized_native_instruction
                token_overlap = _token_overlap_score(instruction, teacher_instruction)
                candidates.append(
                    {
                        "teacher_sequence_index": episode["sequence_index"],
                        "teacher_source_sequence_index": episode.get("source_sequence_index"),
                        "instruction": teacher_instruction,
                        "distance_mse": _squared_distance(native_initial_obs, teacher_initial_obs),
                        "exact_instruction_match": exact_match,
                        "normalized_exact_instruction_match": normalized_exact_match,
                        "token_overlap": token_overlap,
                    }
                )

            if any(item["exact_instruction_match"] for item in candidates):
                exact_match_counts[subtask] += 1
            if any(item["normalized_exact_instruction_match"] for item in candidates):
                normalized_exact_counts[subtask] += 1

            candidates.sort(
                key=lambda item: (
                    not item["exact_instruction_match"],
                    not item["normalized_exact_instruction_match"],
                    -item["token_overlap"],
                    item["distance_mse"],
                    item["teacher_sequence_index"],
                )
            )
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
        "selection_mode": "instruction_prioritized_state_match",
        "episodes": remapped_episodes,
    }
    summary = {
        "source_rollout_path": str(Path(args.rollout_path).resolve()),
        "matching_source_native_log": str(Path(args.native_log).resolve()),
        "top_k_per_subtask": args.top_k_per_subtask,
        "selection_mode": "instruction_prioritized_state_match",
        "selected_sequences": len(remapped_episodes),
        "selected_total_steps": sum(len(ep["steps"]) for ep in remapped_episodes),
        "selected_task_names": sorted({ep.get("task_name") for ep in remapped_episodes}),
        "task_counts": dict(task_counts),
        "exact_match_counts": dict(exact_match_counts),
        "normalized_exact_match_counts": dict(normalized_exact_counts),
        "exact_match_rate_by_task": {
            task: exact_match_counts[task] / count if count else 0.0
            for task, count in sorted(task_counts.items())
        },
        "normalized_exact_match_rate_by_task": {
            task: normalized_exact_counts[task] / count if count else 0.0
            for task, count in sorted(task_counts.items())
        },
        "selections": selections,
    }

    write_json(output_path, payload)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
