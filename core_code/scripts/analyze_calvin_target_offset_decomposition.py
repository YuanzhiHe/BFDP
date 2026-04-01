from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


AXIS_LABELS = {
    0: "x_like",
    1: "y_like",
    2: "z_like",
}


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _abs_mean(values: list[float]) -> float:
    return float(sum(abs(value) for value in values) / len(values))


def _signed_mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _dominant_axis(delta: list[float]) -> str:
    axis_index = max(range(len(delta)), key=lambda idx: abs(delta[idx]))
    return AXIS_LABELS[axis_index]


def _episode_axis_summary(episode: dict[str, object]) -> dict[str, object]:
    summary = episode["episode_summary"]
    best_step = int(summary["best_avg_fingertip_step"])
    best_row = next(row for row in episode["per_step"] if int(row["step"]) == best_step)

    tcp_delta = [float(value) for value in summary["best_tcp_delta_at_best_avg_fingertip_step"]]
    avg_tip_delta = [float(value) for value in summary["best_avg_fingertip_delta"]]
    tcp_abs = [abs(value) for value in tcp_delta]
    avg_tip_abs = [abs(value) for value in avg_tip_delta]

    return {
        "best_step": best_step,
        "target_name": best_row["target_name"],
        "target_link_name": best_row["target_link_name"],
        "best_tcp_distance": float(summary["best_tcp_distance"]),
        "best_avg_fingertip_distance": float(summary["best_avg_fingertip_distance"]),
        "tcp_delta_xyz": tcp_delta,
        "avg_fingertip_delta_xyz": avg_tip_delta,
        "tcp_abs_delta_xyz": tcp_abs,
        "avg_fingertip_abs_delta_xyz": avg_tip_abs,
        "tcp_dominant_axis": _dominant_axis(tcp_delta),
        "avg_fingertip_dominant_axis": _dominant_axis(avg_tip_delta),
    }


def _task_aggregate(task_results: list[dict[str, object]]) -> dict[str, object]:
    if not task_results:
        return {}

    tcp_axis_values = defaultdict(list)
    avg_tip_axis_values = defaultdict(list)
    tcp_dominant_counts = defaultdict(int)
    avg_tip_dominant_counts = defaultdict(int)

    for episode in task_results:
        axis_summary = episode["axis_summary"]
        for axis_index, axis_name in AXIS_LABELS.items():
            tcp_axis_values[axis_name].append(float(axis_summary["tcp_delta_xyz"][axis_index]))
            avg_tip_axis_values[axis_name].append(float(axis_summary["avg_fingertip_delta_xyz"][axis_index]))
        tcp_dominant_counts[str(axis_summary["tcp_dominant_axis"])] += 1
        avg_tip_dominant_counts[str(axis_summary["avg_fingertip_dominant_axis"])] += 1

    return {
        "episodes": len(task_results),
        "tcp_signed_mean_delta_xyz": {
            axis_name: _signed_mean(values) for axis_name, values in sorted(tcp_axis_values.items())
        },
        "tcp_abs_mean_delta_xyz": {
            axis_name: _abs_mean(values) for axis_name, values in sorted(tcp_axis_values.items())
        },
        "avg_fingertip_signed_mean_delta_xyz": {
            axis_name: _signed_mean(values) for axis_name, values in sorted(avg_tip_axis_values.items())
        },
        "avg_fingertip_abs_mean_delta_xyz": {
            axis_name: _abs_mean(values) for axis_name, values in sorted(avg_tip_axis_values.items())
        },
        "tcp_dominant_axis_counts": dict(sorted(tcp_dominant_counts.items())),
        "avg_fingertip_dominant_axis_counts": dict(sorted(avg_tip_dominant_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    payload: dict[str, object] = {
        "input": str(Path(args.input).resolve()),
        "status": "started",
    }

    try:
        source = _load_json(args.input)
        results = []
        per_task: dict[str, list[dict[str, object]]] = defaultdict(list)

        for episode in source.get("results", []):
            task_name = str(episode.get("task_name", "unknown"))
            if task_name not in {"turn_on_led", "open_drawer"}:
                continue

            axis_summary = _episode_axis_summary(episode)
            result = {
                "sequence_index": episode.get("sequence_index"),
                "matched_sequence_index": episode.get("matched_sequence_index"),
                "task_name": task_name,
                "instruction": episode.get("instruction"),
                "axis_summary": axis_summary,
            }
            results.append(result)
            per_task[task_name].append(result)

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
                "task_aggregates": {
                    task_name: _task_aggregate(task_results)
                    for task_name, task_results in sorted(per_task.items())
                },
                "results": results,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
