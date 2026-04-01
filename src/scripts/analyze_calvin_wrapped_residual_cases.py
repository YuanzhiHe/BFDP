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


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-step-log", required=True)
    parser.add_argument("--scene-semantics-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wrapped-threshold", type=float, default=0.10)
    args = parser.parse_args()

    first_step_payload = _load_json(args.first_step_log)
    semantics_payload = _load_json(args.scene_semantics_log)
    scene_label_by_index = {
        int(item["scene_index"]): item for item in semantics_payload.get("scene_obs_labels", [])
    }

    residual_cases: list[dict[str, object]] = []
    for item in first_step_payload.get("comparisons", []):
        native = item["native_next_selected_obs"]
        dataset = item["dataset_next_selected_obs"]
        count = min(len(native), len(dataset))
        wrapped_abs_deltas: list[dict[str, object]] = []
        wrapped_sum = 0.0
        for selected_idx in range(count):
            dataset_value = float(dataset[selected_idx])
            native_value = float(native[selected_idx])
            raw = native_value - dataset_value
            if selected_idx >= 9:
                scene_index = selected_idx - 9
                label = scene_label_by_index.get(scene_index)
                component = None if label is None else label.get("component")
                if component in {"roll", "pitch", "yaw"}:
                    wrapped = _wrapped_delta(native_value, dataset_value)
                else:
                    wrapped = raw
            else:
                scene_index = None
                label = None
                wrapped = raw
            wrapped_abs = abs(wrapped)
            wrapped_sum += wrapped_abs
            wrapped_abs_deltas.append(
                {
                    "selected_index": selected_idx,
                    "scene_index": scene_index,
                    "semantic_label": label,
                    "dataset_value": dataset_value,
                    "native_value": native_value,
                    "raw_abs_delta": abs(raw),
                    "wrapped_abs_delta": wrapped_abs,
                }
            )

        wrapped_mae = wrapped_sum / count if count else 0.0
        if wrapped_mae <= args.wrapped_threshold:
            continue

        residual_cases.append(
            {
                "sequence_index": item.get("sequence_index"),
                "task_name": item.get("task_name"),
                "instruction": item.get("instruction"),
                "wrapped_selected_obs_mae": wrapped_mae,
                "top_wrapped_abs_deltas": sorted(
                    wrapped_abs_deltas,
                    key=lambda record: record["wrapped_abs_delta"],
                    reverse=True,
                )[:10],
            }
        )

    payload = {
        "first_step_log": str(Path(args.first_step_log).resolve()),
        "scene_semantics_log": str(Path(args.scene_semantics_log).resolve()),
        "wrapped_threshold": args.wrapped_threshold,
        "residual_case_count": len(residual_cases),
        "residual_cases": residual_cases,
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
