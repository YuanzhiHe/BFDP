from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


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
WRAPPED_SELECTED_OBS_INDICES = ROBOT_EULER_SELECTED_INDICES | SCENE_ORIENTATION_SELECTED_INDICES


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _vector_stats(a: list[float], b: list[float]) -> dict[str, float | int]:
    count = min(len(a), len(b))
    if count == 0:
        return {"count": 0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    diffs = [a[idx] - b[idx] for idx in range(count)]
    abs_diffs = [abs(value) for value in diffs]
    return {
        "count": count,
        "mae": sum(abs_diffs) / count,
        "mse": sum(value * value for value in diffs) / count,
        "max_abs": max(abs_diffs),
    }


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _wrapped_vector_stats(
    a: list[float],
    b: list[float],
    wrapped_indices: set[int],
) -> dict[str, float | int]:
    count = min(len(a), len(b))
    if count == 0:
        return {"count": 0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    diffs = []
    for idx in range(count):
        if idx in wrapped_indices:
            diffs.append(_wrapped_delta(a[idx], b[idx]))
        else:
            diffs.append(a[idx] - b[idx])
    abs_diffs = [abs(value) for value in diffs]
    return {
        "count": count,
        "mae": sum(abs_diffs) / count,
        "mse": sum(value * value for value in diffs) / count,
        "max_abs": max(abs_diffs),
    }


def _mean_action(actions: list[list[float]]) -> list[float]:
    if not actions:
        return []
    action_dim = len(actions[0])
    return [
        sum(action[idx] for action in actions) / len(actions)
        for idx in range(action_dim)
    ]


def _action_component_stats(a: list[float], b: list[float]) -> dict[str, dict[str, float | int]]:
    translation = _vector_stats(a[:3], b[:3])
    rotation = _vector_stats(a[3:6], b[3:6])
    gripper = _vector_stats(a[6:7], b[6:7])
    return {
        "translation": translation,
        "rotation": rotation,
        "gripper": gripper,
    }


def _selected_obs_from_teacher_step(step: dict) -> list[float]:
    return step["robot_obs"][:9] + step.get("scene_obs", [])[:24]


def _first_threshold_step(step_summaries: list[dict[str, object]], key: str, threshold: float) -> int | None:
    for summary in step_summaries:
        stats = summary.get(key, {})
        value = stats.get("mae")
        if isinstance(value, (int, float)) and value > threshold:
            return int(summary["step"])
    return None


def _first_nested_threshold_step(
    step_summaries: list[dict[str, object]],
    key: str,
    nested_key: str,
    threshold: float,
) -> int | None:
    for summary in step_summaries:
        stats = summary.get(key, {})
        if not isinstance(stats, dict):
            continue
        nested = stats.get(nested_key, {})
        if not isinstance(nested, dict):
            continue
        value = nested.get("mae")
        if isinstance(value, (int, float)) and value > threshold:
            return int(summary["step"])
    return None


def _match_teacher_episode(episodes: list[dict], subtask: str, instruction: str) -> dict | None:
    for episode in episodes:
        if episode.get("task_name") == subtask and episode.get("instruction") == instruction:
            return episode
    for episode in episodes:
        if episode.get("task_name") == subtask:
            return episode
    return None


def _summarize_subtask(native_subtask: dict, teacher_episode: dict | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "subtask": native_subtask["subtask"],
        "language_annotation": native_subtask["language_annotation"],
        "native_rollout_steps": native_subtask["rollout_steps_attempted"],
        "native_success": native_subtask["rollout_success"],
        "teacher_episode_found": teacher_episode is not None,
    }
    if teacher_episode is None:
        return summary

    teacher_steps = teacher_episode["steps"]
    native_trace = native_subtask["rollout_trace"]
    overlap = min(len(teacher_steps), len(native_trace))

    teacher_actions = [step["action"] for step in teacher_steps[:overlap]]
    native_actions = [step["action"] for step in native_trace[:overlap]]
    step_summaries: list[dict[str, object]] = []
    for step_idx in range(overlap):
        teacher_step = teacher_steps[step_idx]
        native_step = native_trace[step_idx]
        teacher_selected_obs = _selected_obs_from_teacher_step(teacher_step)
        native_next_obs = native_step.get("next_obs", {}).get("selected_obs", [])
        step_summaries.append(
            {
                "step": step_idx,
                "obs_delta": _vector_stats(native_next_obs, teacher_selected_obs),
                "wrapped_obs_delta": _wrapped_vector_stats(
                    native_next_obs,
                    teacher_selected_obs,
                    WRAPPED_SELECTED_OBS_INDICES,
                ),
                "wrapped_robot_obs_delta": _wrapped_vector_stats(
                    native_next_obs[:9],
                    teacher_selected_obs[:9],
                    ROBOT_EULER_SELECTED_INDICES,
                ),
                "wrapped_scene_obs_delta": _wrapped_vector_stats(
                    native_next_obs[9:],
                    teacher_selected_obs[9:],
                    {idx - 9 for idx in SCENE_ORIENTATION_SELECTED_INDICES},
                ),
                "action_delta": _vector_stats(
                    native_step.get("action", []),
                    teacher_step["action"],
                ),
                "action_component_delta": _action_component_stats(
                    native_step.get("action", []),
                    teacher_step["action"],
                ),
            }
        )

    summary.update(
        {
            "teacher_sequence_index": teacher_episode.get("sequence_index"),
            "teacher_source_sequence_index": teacher_episode.get("source_sequence_index"),
            "teacher_episode_steps": len(teacher_steps),
            "aligned_steps": overlap,
            "teacher_initial_selected_obs": (
                teacher_steps[0]["robot_obs"][:9] + teacher_steps[0].get("scene_obs", [])[:24]
                if teacher_steps
                else []
            ),
            "teacher_final_selected_obs": (
                teacher_steps[overlap - 1]["robot_obs"][:9]
                + teacher_steps[overlap - 1].get("scene_obs", [])[:24]
                if overlap
                else []
            ),
            "native_initial_selected_obs": native_subtask.get("initial_selected_obs", []),
            "native_final_selected_obs": native_subtask.get("final_selected_obs", []),
            "first_action_delta": _vector_stats(
                native_subtask.get("first_action", []),
                teacher_steps[0]["action"] if teacher_steps else [],
            ),
            "final_action_delta": _vector_stats(
                native_subtask.get("final_action", []),
                teacher_steps[overlap - 1]["action"] if overlap else [],
            ),
            "mean_action_delta": _vector_stats(
                _mean_action(native_actions),
                _mean_action(teacher_actions),
            ),
            "first_action_component_delta": _action_component_stats(
                native_subtask.get("first_action", []),
                teacher_steps[0]["action"] if teacher_steps else [],
            ),
            "final_action_component_delta": _action_component_stats(
                native_subtask.get("final_action", []),
                teacher_steps[overlap - 1]["action"] if overlap else [],
            ),
            "mean_action_component_delta": _action_component_stats(
                _mean_action(native_actions),
                _mean_action(teacher_actions),
            ),
            "initial_obs_delta": _vector_stats(
                native_subtask.get("initial_selected_obs", []),
                teacher_steps[0]["robot_obs"][:9] + teacher_steps[0].get("scene_obs", [])[:24]
                if teacher_steps
                else [],
            ),
            "initial_wrapped_obs_delta": _wrapped_vector_stats(
                native_subtask.get("initial_selected_obs", []),
                teacher_steps[0]["robot_obs"][:9] + teacher_steps[0].get("scene_obs", [])[:24]
                if teacher_steps
                else [],
                WRAPPED_SELECTED_OBS_INDICES,
            ),
            "final_obs_delta": _vector_stats(
                native_subtask.get("final_selected_obs", []),
                teacher_steps[overlap - 1]["robot_obs"][:9]
                + teacher_steps[overlap - 1].get("scene_obs", [])[:24]
                if overlap
                else [],
            ),
            "final_wrapped_obs_delta": _wrapped_vector_stats(
                native_subtask.get("final_selected_obs", []),
                teacher_steps[overlap - 1]["robot_obs"][:9]
                + teacher_steps[overlap - 1].get("scene_obs", [])[:24]
                if overlap
                else [],
                WRAPPED_SELECTED_OBS_INDICES,
            ),
            "teacher_gripper_pattern": [step["action"][-1] for step in teacher_steps[:overlap]],
            "native_gripper_pattern": [step["action"][-1] for step in native_trace[:overlap]],
            "step_deltas": step_summaries,
            "first_obs_mae_over_0p05_step": _first_threshold_step(step_summaries, "obs_delta", 0.05),
            "first_obs_mae_over_0p10_step": _first_threshold_step(step_summaries, "obs_delta", 0.10),
            "first_wrapped_obs_mae_over_0p05_step": _first_threshold_step(step_summaries, "wrapped_obs_delta", 0.05),
            "first_wrapped_obs_mae_over_0p10_step": _first_threshold_step(step_summaries, "wrapped_obs_delta", 0.10),
            "first_wrapped_robot_obs_mae_over_0p05_step": _first_threshold_step(
                step_summaries,
                "wrapped_robot_obs_delta",
                0.05,
            ),
            "first_wrapped_robot_obs_mae_over_0p10_step": _first_threshold_step(
                step_summaries,
                "wrapped_robot_obs_delta",
                0.10,
            ),
            "first_wrapped_scene_obs_mae_over_0p05_step": _first_threshold_step(
                step_summaries,
                "wrapped_scene_obs_delta",
                0.05,
            ),
            "first_wrapped_scene_obs_mae_over_0p10_step": _first_threshold_step(
                step_summaries,
                "wrapped_scene_obs_delta",
                0.10,
            ),
            "first_action_mae_over_0p05_step": _first_threshold_step(step_summaries, "action_delta", 0.05),
            "first_action_mae_over_0p10_step": _first_threshold_step(step_summaries, "action_delta", 0.10),
            "first_translation_mae_over_0p05_step": _first_nested_threshold_step(
                step_summaries,
                "action_component_delta",
                "translation",
                0.05,
            ),
            "first_translation_mae_over_0p10_step": _first_nested_threshold_step(
                step_summaries,
                "action_component_delta",
                "translation",
                0.10,
            ),
            "first_rotation_mae_over_0p05_step": _first_nested_threshold_step(
                step_summaries,
                "action_component_delta",
                "rotation",
                0.05,
            ),
            "first_rotation_mae_over_0p10_step": _first_nested_threshold_step(
                step_summaries,
                "action_component_delta",
                "rotation",
                0.10,
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-log", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    native_payload = _load_json(args.native_log)
    rollout_payload = _load_json(args.rollout_path)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    teacher_episodes = rollout_payload["episodes"]
    comparisons: list[dict[str, object]] = []

    for sequence in native_payload.get("sequence_results", []):
        for native_subtask in sequence.get("subtask_results", []):
            teacher_episode = _match_teacher_episode(
                teacher_episodes,
                native_subtask["subtask"],
                native_subtask["language_annotation"],
            )
            comparison = _summarize_subtask(native_subtask, teacher_episode)
            comparison["native_sequence_index"] = sequence["sequence_index"]
            comparisons.append(comparison)

    payload = {
        "native_log": str(Path(args.native_log).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
