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


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bucket_label(value: float) -> str:
    if value < 0.10:
        return "<0.10"
    if value < 0.20:
        return "0.10-0.20"
    if value < 0.30:
        return "0.20-0.30"
    if value < 0.40:
        return "0.30-0.40"
    if value < 0.50:
        return "0.40-0.50"
    return ">=0.50"


def _wrap_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selected-obs-threshold",
        type=float,
        default=0.10,
        help="Analyze episodes whose selected_obs MAE exceeds this threshold.",
    )
    parser.add_argument(
        "--top-k-dims",
        type=int,
        default=12,
        help="Number of dominant dimensions to retain in the summary.",
    )
    parser.add_argument(
        "--wrap-epsilon",
        type=float,
        default=0.20,
        help="Mark a delta as 2*pi-like when abs(abs(raw)-2*pi) <= epsilon.",
    )
    args = parser.parse_args()

    payload = _load_json(args.input)
    comparisons = payload.get("comparisons", [])
    outliers = [
        item
        for item in comparisons
        if float(item.get("selected_obs_delta", {}).get("mae", 0.0)) > args.selected_obs_threshold
    ]

    scene_abs_by_dim: defaultdict[int, list[float]] = defaultdict(list)
    scene_raw_by_dim: defaultdict[int, list[float]] = defaultdict(list)
    scene_wrap_by_dim: Counter[int] = Counter()
    per_task_count: Counter[str] = Counter()
    per_task_buckets: defaultdict[str, Counter[str]] = defaultdict(Counter)
    outlier_examples: list[dict[str, object]] = []

    for item in outliers:
        task_name = item.get("task_name", "unknown")
        per_task_count[task_name] += 1
        selected_mae = float(item["selected_obs_delta"]["mae"])
        per_task_buckets[task_name][_bucket_label(selected_mae)] += 1

        dataset_selected = item.get("dataset_next_selected_obs", [])
        native_selected = item.get("native_next_selected_obs", [])
        scene_count = max(0, min(len(dataset_selected), len(native_selected)) - 9)
        per_dim_abs: list[tuple[int, float]] = []
        wrap_dims: list[int] = []
        for scene_idx in range(scene_count):
            selected_idx = 9 + scene_idx
            raw = float(native_selected[selected_idx]) - float(dataset_selected[selected_idx])
            abs_raw = abs(raw)
            wrapped = _wrap_delta(float(native_selected[selected_idx]), float(dataset_selected[selected_idx]))
            wrap_like = abs(abs_raw - (2.0 * math.pi)) <= args.wrap_epsilon
            scene_abs_by_dim[scene_idx].append(abs_raw)
            scene_raw_by_dim[scene_idx].append(raw)
            if wrap_like:
                scene_wrap_by_dim[scene_idx] += 1
                wrap_dims.append(scene_idx)
            per_dim_abs.append((scene_idx, abs_raw))

        top_dims = sorted(per_dim_abs, key=lambda pair: pair[1], reverse=True)[:3]
        outlier_examples.append(
            {
                "sequence_index": item.get("sequence_index"),
                "task_name": task_name,
                "instruction": item.get("instruction"),
                "selected_obs_mae": selected_mae,
                "scene_obs_mae": float(item.get("scene_obs_delta", {}).get("mae", 0.0)),
                "robot_obs_mae": float(item.get("robot_obs_delta", {}).get("mae", 0.0)),
                "top_scene_dims": [
                    {
                        "scene_index": scene_idx,
                        "abs_delta": abs_delta,
                    }
                    for scene_idx, abs_delta in top_dims
                ],
                "wrap_like_scene_dims": wrap_dims,
            }
        )

    dominant_dims: list[dict[str, object]] = []
    for scene_idx, values in scene_abs_by_dim.items():
        dominant_dims.append(
            {
                "scene_index": scene_idx,
                "count": len(values),
                "mean_abs_delta": _mean(values),
                "median_abs_delta": _median(values),
                "max_abs_delta": max(values) if values else 0.0,
                "wrap_like_count": int(scene_wrap_by_dim.get(scene_idx, 0)),
                "wrap_like_rate": (
                    float(scene_wrap_by_dim.get(scene_idx, 0)) / len(values) if values else 0.0
                ),
                "mean_signed_delta": _mean(scene_raw_by_dim[scene_idx]),
            }
        )
    dominant_dims.sort(
        key=lambda item: (
            float(item["mean_abs_delta"]),
            float(item["max_abs_delta"]),
            float(item["wrap_like_count"]),
        ),
        reverse=True,
    )

    output = {
        "input": str(Path(args.input).resolve()),
        "status": "ok",
        "selected_obs_threshold": args.selected_obs_threshold,
        "top_k_dims": args.top_k_dims,
        "wrap_epsilon": args.wrap_epsilon,
        "episodes_total": len(comparisons),
        "outlier_count": len(outliers),
        "outlier_rate": float(len(outliers)) / len(comparisons) if comparisons else 0.0,
        "selected_obs_mae_buckets": dict(
            sorted(Counter(_bucket_label(float(item["selected_obs_delta"]["mae"])) for item in outliers).items())
        ),
        "per_task_outlier_count": dict(sorted(per_task_count.items())),
        "per_task_outlier_mae_buckets": {
            task: dict(sorted(buckets.items()))
            for task, buckets in sorted(per_task_buckets.items())
        },
        "dominant_scene_dims": dominant_dims[: args.top_k_dims],
        "wrap_like_scene_dims": [
            item for item in dominant_dims if int(item["wrap_like_count"]) > 0
        ][: args.top_k_dims],
        "highest_mae_examples": sorted(
            outlier_examples,
            key=lambda item: float(item["selected_obs_mae"]),
            reverse=True,
        )[:10],
    }

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
