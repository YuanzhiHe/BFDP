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


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _label_for_selected_index(selected_index: int, semantics_by_scene_index: dict[int, dict]) -> dict[str, object]:
    if selected_index < 9:
        return {
            "space": "robot_obs",
            "selected_index": selected_index,
            "robot_index": selected_index,
        }
    scene_index = selected_index - 9
    semantic = semantics_by_scene_index.get(scene_index, {})
    return {
        "space": "scene_obs",
        "selected_index": selected_index,
        "scene_index": scene_index,
        "semantic": semantic,
    }


def _per_index_residuals(
    native_selected_obs: list[float],
    teacher_selected_obs: list[float],
    semantics_by_scene_index: dict[int, dict],
) -> list[dict[str, object]]:
    residuals: list[dict[str, object]] = []
    count = min(len(native_selected_obs), len(teacher_selected_obs))
    for idx in range(count):
        raw = float(native_selected_obs[idx]) - float(teacher_selected_obs[idx])
        if idx in ROBOT_EULER_SELECTED_INDICES or idx in SCENE_ORIENTATION_SELECTED_INDICES:
            wrapped = _wrapped_delta(float(native_selected_obs[idx]), float(teacher_selected_obs[idx]))
        else:
            wrapped = raw
        record = _label_for_selected_index(idx, semantics_by_scene_index)
        record.update(
            {
                "native_value": float(native_selected_obs[idx]),
                "teacher_value": float(teacher_selected_obs[idx]),
                "raw_delta": raw,
                "wrapped_delta": wrapped,
                "raw_abs_delta": abs(raw),
                "wrapped_abs_delta": abs(wrapped),
                "wrap_reduction": abs(raw) - abs(wrapped),
            }
        )
        residuals.append(record)
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--scene-semantics-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    trace_payload = _load_json(args.trace_log)
    semantics_payload = _load_json(args.scene_semantics_log)
    semantics_by_scene_index = {
        int(item["scene_index"]): item for item in semantics_payload.get("scene_obs_labels", [])
    }

    results: list[dict[str, object]] = []
    aggregate_scene: dict[int, list[float]] = {}
    aggregate_robot: dict[int, list[float]] = {}

    for comparison in trace_payload.get("comparisons", []):
        step_deltas = comparison.get("step_deltas", [])
        if len(step_deltas) <= args.step:
            continue
        teacher_steps = comparison.get("aligned_steps", 0)
        if teacher_steps <= args.step:
            continue

        native_log = _load_json(trace_payload["native_log"])
        matching_subtask = None
        for sequence in native_log.get("sequence_results", []):
            if sequence.get("sequence_index") != comparison.get("native_sequence_index"):
                continue
            for subtask in sequence.get("subtask_results", []):
                if subtask.get("subtask") == comparison.get("subtask"):
                    matching_subtask = subtask
                    break
            if matching_subtask is not None:
                break
        if matching_subtask is None:
            continue

        native_selected_obs = (
            matching_subtask["rollout_trace"][args.step]["next_obs"]["selected_obs"]
        )
        teacher_selected_obs = (
            comparison["teacher_initial_selected_obs"]
            if args.step == 0
            else comparison["step_deltas"][args.step - 1].get("teacher_selected_obs_placeholder", [])
        )

        if not teacher_selected_obs:
            # Teacher trace isn't embedded in the wrapped comparison payload; reconstruct from aligned-step convention:
            # native step i next_obs is compared against teacher step i selected obs.
            teacher_selected_obs = (
                comparison["teacher_initial_selected_obs"]
                if args.step == 0
                else []
            )
        if not teacher_selected_obs:
            continue

        residuals = _per_index_residuals(native_selected_obs, teacher_selected_obs, semantics_by_scene_index)
        for residual in residuals:
            if residual["space"] == "scene_obs":
                scene_index = int(residual["scene_index"])
                aggregate_scene.setdefault(scene_index, []).append(float(residual["wrapped_abs_delta"]))
            else:
                robot_index = int(residual["robot_index"])
                aggregate_robot.setdefault(robot_index, []).append(float(residual["wrapped_abs_delta"]))

        results.append(
            {
                "subtask": comparison["subtask"],
                "language_annotation": comparison["language_annotation"],
                "native_sequence_index": comparison["native_sequence_index"],
                "step": args.step,
                "raw_obs_mae": comparison["step_deltas"][args.step]["obs_delta"]["mae"],
                "wrapped_obs_mae": comparison["step_deltas"][args.step]["wrapped_obs_delta"]["mae"],
                "wrapped_robot_obs_mae": comparison["step_deltas"][args.step]["wrapped_robot_obs_delta"]["mae"],
                "wrapped_scene_obs_mae": comparison["step_deltas"][args.step]["wrapped_scene_obs_delta"]["mae"],
                "top_wrapped_residuals": sorted(
                    residuals,
                    key=lambda item: item["wrapped_abs_delta"],
                    reverse=True,
                )[: args.top_k],
            }
        )

    payload = {
        "trace_log": str(Path(args.trace_log).resolve()),
        "scene_semantics_log": str(Path(args.scene_semantics_log).resolve()),
        "step": args.step,
        "top_k": args.top_k,
        "subtask_count": len(results),
        "subtasks": results,
        "aggregate_scene_wrapped_abs_delta": [
            {
                "scene_index": scene_index,
                "semantic": semantics_by_scene_index.get(scene_index, {}),
                "mean_wrapped_abs_delta": sum(values) / len(values),
                "max_wrapped_abs_delta": max(values),
                "count": len(values),
            }
            for scene_index, values in sorted(
                aggregate_scene.items(),
                key=lambda item: (sum(item[1]) / len(item[1])),
                reverse=True,
            )
        ],
        "aggregate_robot_wrapped_abs_delta": [
            {
                "robot_index": robot_index,
                "mean_wrapped_abs_delta": sum(values) / len(values),
                "max_wrapped_abs_delta": max(values),
                "count": len(values),
            }
            for robot_index, values in sorted(
                aggregate_robot.items(),
                key=lambda item: (sum(item[1]) / len(item[1])),
                reverse=True,
            )
        ],
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
