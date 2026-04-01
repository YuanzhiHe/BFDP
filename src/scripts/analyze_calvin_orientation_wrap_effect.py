from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


ANGLE_COMPONENTS = {"roll", "pitch", "yaw"}
ROBOT_EULER_SELECTED_INDICES = {3, 4, 5}


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _mae(values: list[float]) -> float:
    return _mean([abs(value) for value in values])


def _extract_angle_index_sets(semantics_payload: dict) -> tuple[set[int], set[int], set[int]]:
    angle_indices: set[int] = set()
    yaw_indices: set[int] = set()
    roll_pitch_indices: set[int] = set()
    for item in semantics_payload.get("scene_obs_labels", []):
        if item.get("group") != "movable_object":
            continue
        component = item.get("component")
        scene_idx = int(item["scene_index"])
        selected_idx = 9 + scene_idx
        if component in ANGLE_COMPONENTS:
            angle_indices.add(selected_idx)
        if component == "yaw":
            yaw_indices.add(selected_idx)
        if component in {"roll", "pitch"}:
            roll_pitch_indices.add(selected_idx)
    return angle_indices, yaw_indices, roll_pitch_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-step-log", required=True)
    parser.add_argument("--scene-semantics-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-threshold", type=float, default=0.10)
    parser.add_argument("--wrapped-threshold", type=float, default=0.10)
    parser.add_argument("--yaw-only-threshold", type=float, default=0.10)
    args = parser.parse_args()

    first_step_payload = _load_json(args.first_step_log)
    semantics_payload = _load_json(args.scene_semantics_log)
    angle_indices, yaw_indices, roll_pitch_indices = _extract_angle_index_sets(semantics_payload)
    comparisons = first_step_payload.get("comparisons", [])

    raw_maes: list[float] = []
    wrapped_maes: list[float] = []
    yaw_only_maes: list[float] = []
    angle_only_wrapped_maes: list[float] = []
    raw_outlier_task_counter: Counter[str] = Counter()
    resolved_outlier_task_counter: Counter[str] = Counter()
    unresolved_outlier_task_counter: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    wrapped_dominance: defaultdict[int, list[float]] = defaultdict(list)

    for item in comparisons:
        native = item["native_next_selected_obs"]
        dataset = item["dataset_next_selected_obs"]
        count = min(len(native), len(dataset))
        raw_diffs = []
        wrapped_diffs = []
        yaw_only_diffs = []
        angle_wrapped_diffs = []
        per_index_records: list[dict[str, object]] = []
        for idx in range(count):
            raw = float(native[idx]) - float(dataset[idx])
            if idx in ROBOT_EULER_SELECTED_INDICES or idx in angle_indices:
                wrapped = _wrapped_delta(float(native[idx]), float(dataset[idx]))
                angle_wrapped_diffs.append(wrapped)
            else:
                wrapped = raw
            raw_diffs.append(raw)
            wrapped_diffs.append(wrapped)
            if idx in yaw_indices:
                yaw_only_diffs.append(wrapped)
            elif idx not in roll_pitch_indices:
                yaw_only_diffs.append(raw)
            per_index_records.append(
                {
                    "selected_index": idx,
                    "raw_abs_delta": abs(raw),
                    "wrapped_abs_delta": abs(wrapped),
                    "wrap_reduction": abs(raw) - abs(wrapped),
                }
            )
            if idx >= 9:
                wrapped_dominance[idx - 9].append(abs(raw) - abs(wrapped))

        raw_mae = _mae(raw_diffs)
        wrapped_mae = _mae(wrapped_diffs)
        yaw_only_mae = _mae(yaw_only_diffs)
        angle_only_wrapped_mae = _mae(angle_wrapped_diffs)
        raw_maes.append(raw_mae)
        wrapped_maes.append(wrapped_mae)
        yaw_only_maes.append(yaw_only_mae)
        angle_only_wrapped_maes.append(angle_only_wrapped_mae)

        task_name = item.get("task_name", "unknown")
        raw_outlier = raw_mae > args.raw_threshold
        wrapped_outlier = wrapped_mae > args.wrapped_threshold
        if raw_outlier:
            raw_outlier_task_counter[task_name] += 1
            if wrapped_outlier:
                unresolved_outlier_task_counter[task_name] += 1
            else:
                resolved_outlier_task_counter[task_name] += 1

        examples.append(
            {
                "sequence_index": item.get("sequence_index"),
                "task_name": task_name,
                "instruction": item.get("instruction"),
                "raw_selected_obs_mae": raw_mae,
                "wrapped_selected_obs_mae": wrapped_mae,
                "yaw_only_selected_obs_mae": yaw_only_mae,
                "angle_only_wrapped_mae": angle_only_wrapped_mae,
                "raw_minus_wrapped_mae": raw_mae - wrapped_mae,
                "largest_wrap_reductions": sorted(
                    per_index_records,
                    key=lambda record: record["wrap_reduction"],
                    reverse=True,
                )[:5],
            }
        )

    raw_outlier_count = sum(1 for value in raw_maes if value > args.raw_threshold)
    wrapped_outlier_count = sum(1 for value in wrapped_maes if value > args.wrapped_threshold)
    yaw_only_outlier_count = sum(1 for value in yaw_only_maes if value > args.yaw_only_threshold)

    payload = {
        "first_step_log": str(Path(args.first_step_log).resolve()),
        "scene_semantics_log": str(Path(args.scene_semantics_log).resolve()),
        "raw_threshold": args.raw_threshold,
        "wrapped_threshold": args.wrapped_threshold,
        "yaw_only_threshold": args.yaw_only_threshold,
        "episodes_compared": len(comparisons),
        "raw_selected_obs_mae_mean": _mean(raw_maes),
        "raw_selected_obs_mae_median": _median(raw_maes),
        "wrapped_selected_obs_mae_mean": _mean(wrapped_maes),
        "wrapped_selected_obs_mae_median": _median(wrapped_maes),
        "yaw_only_selected_obs_mae_mean": _mean(yaw_only_maes),
        "yaw_only_selected_obs_mae_median": _median(yaw_only_maes),
        "angle_only_wrapped_mae_mean": _mean(angle_only_wrapped_maes),
        "angle_only_wrapped_mae_median": _median(angle_only_wrapped_maes),
        "raw_outlier_count": raw_outlier_count,
        "wrapped_outlier_count": wrapped_outlier_count,
        "yaw_only_outlier_count": yaw_only_outlier_count,
        "resolved_outlier_count": raw_outlier_count - wrapped_outlier_count,
        "raw_outlier_task_count": dict(sorted(raw_outlier_task_counter.items())),
        "resolved_outlier_task_count": dict(sorted(resolved_outlier_task_counter.items())),
        "unresolved_outlier_task_count": dict(sorted(unresolved_outlier_task_counter.items())),
        "scene_dim_wrap_reduction": [
            {
                "scene_index": scene_idx,
                "mean_abs_reduction": _mean(values),
                "median_abs_reduction": _median(values),
            }
            for scene_idx, values in sorted(
                wrapped_dominance.items(),
                key=lambda item: _mean(item[1]),
                reverse=True,
            )
        ],
        "largest_raw_to_wrapped_reductions": sorted(
            examples,
            key=lambda item: item["raw_minus_wrapped_mae"],
            reverse=True,
        )[:12],
        "largest_remaining_wrapped_errors": sorted(
            examples,
            key=lambda item: item["wrapped_selected_obs_mae"],
            reverse=True,
        )[:12],
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
