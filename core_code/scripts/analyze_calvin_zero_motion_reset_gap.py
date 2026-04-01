from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROBOT_EULER_SELECTED_INDICES = {3, 4, 5}
SCENE_ORIENTATION_SELECTED_INDICES = {
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
}


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _wrapped_vector_stats(a: list[float], b: list[float]) -> dict[str, float | int]:
    count = min(len(a), len(b))
    if count == 0:
        return {"count": 0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    diffs = []
    for idx in range(count):
        if idx in ROBOT_EULER_SELECTED_INDICES or idx in SCENE_ORIENTATION_SELECTED_INDICES:
            diffs.append(_wrapped_delta(float(a[idx]), float(b[idx])))
        else:
            diffs.append(float(a[idx]) - float(b[idx]))
    abs_diffs = [abs(value) for value in diffs]
    return {
        "count": count,
        "mae": sum(abs_diffs) / count,
        "mse": sum(value * value for value in diffs) / count,
        "max_abs": max(abs_diffs),
    }


def _selected_obs_from_teacher_step(step: dict) -> list[float]:
    return list(step["robot_obs"][:9]) + list(step.get("scene_obs", [])[:24])


def _match_teacher_episode(episodes: list[dict], subtask: str, instruction: str) -> dict | None:
    for episode in episodes:
        if episode.get("task_name") == subtask and episode.get("instruction") == instruction:
            return episode
    for episode in episodes:
        if episode.get("task_name") == subtask:
            return episode
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-motion-log", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    zero_payload = _load_json(args.zero_motion_log)
    rollout_payload = _load_json(args.rollout_path)
    teacher_episodes = rollout_payload.get("episodes", [])

    results: list[dict[str, object]] = []
    for sequence in zero_payload.get("sequence_results", []):
        if not sequence.get("subtask_results"):
            continue
        subtask = sequence["subtask_results"][0]
        teacher_episode = _match_teacher_episode(
            teacher_episodes,
            subtask["subtask"],
            subtask["language_annotation"],
        )
        if teacher_episode is None or not teacher_episode.get("steps"):
            continue
        teacher_initial = _selected_obs_from_teacher_step(teacher_episode["steps"][0])
        native_initial = list(subtask.get("initial_selected_obs", []))
        native_next = list(subtask.get("final_selected_obs", []))
        results.append(
            {
                "subtask": subtask["subtask"],
                "language_annotation": subtask["language_annotation"],
                "teacher_sequence_index": teacher_episode.get("sequence_index"),
                "native_sequence_index": sequence["sequence_index"],
                "native_initial_vs_teacher_initial": _wrapped_vector_stats(native_initial, teacher_initial),
                "native_next_vs_teacher_initial": _wrapped_vector_stats(native_next, teacher_initial),
                "native_next_vs_native_initial": _wrapped_vector_stats(native_next, native_initial),
            }
        )

    payload = {
        "zero_motion_log": str(Path(args.zero_motion_log).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "comparison_count": len(results),
        "comparisons": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
