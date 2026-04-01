from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


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


def _iter_subtasks(payload: dict) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for sequence in payload.get("sequence_results", []):
        for subtask in sequence.get("subtask_results", []):
            rollout_trace = subtask.get("rollout_trace", [])
            first_trace = rollout_trace[0] if rollout_trace else {}
            items.append(
                {
                    "sequence_index": sequence.get("sequence_index"),
                    "initial_state": sequence.get("initial_state"),
                    "eval_sequence": sequence.get("eval_sequence"),
                    "subtask": subtask.get("subtask"),
                    "language_annotation": subtask.get("language_annotation"),
                    "initial_selected_obs": subtask.get("initial_selected_obs", []),
                    "first_raw_action": subtask.get("first_raw_action")
                    or first_trace.get("raw_action", []),
                    "first_action": subtask.get("first_action")
                    or first_trace.get("action", []),
                    "rollout_steps_attempted": subtask.get("rollout_steps_attempted"),
                    "rollout_success": subtask.get("rollout_success"),
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-log", required=True)
    parser.add_argument("--candidate-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = _load_json(args.reference_log)
    candidate = _load_json(args.candidate_log)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    reference_items = _iter_subtasks(reference)
    candidate_items = _iter_subtasks(candidate)
    overlap = min(len(reference_items), len(candidate_items))
    comparisons: list[dict[str, object]] = []

    for idx in range(overlap):
        ref = reference_items[idx]
        cand = candidate_items[idx]
        comparisons.append(
            {
                "pair_index": idx,
                "reference_sequence_index": ref["sequence_index"],
                "candidate_sequence_index": cand["sequence_index"],
                "reference_subtask": ref["subtask"],
                "candidate_subtask": cand["subtask"],
                "same_subtask": ref["subtask"] == cand["subtask"],
                "same_language_annotation": ref["language_annotation"] == cand["language_annotation"],
                "same_initial_state": ref["initial_state"] == cand["initial_state"],
                "same_eval_sequence": ref["eval_sequence"] == cand["eval_sequence"],
                "initial_obs_delta": _vector_stats(
                    ref["initial_selected_obs"],
                    cand["initial_selected_obs"],
                ),
                "first_raw_action_delta": _vector_stats(
                    ref["first_raw_action"],
                    cand["first_raw_action"],
                ),
                "first_action_delta": _vector_stats(
                    ref["first_action"],
                    cand["first_action"],
                ),
                "reference_rollout_success": ref["rollout_success"],
                "candidate_rollout_success": cand["rollout_success"],
                "reference_rollout_steps_attempted": ref["rollout_steps_attempted"],
                "candidate_rollout_steps_attempted": cand["rollout_steps_attempted"],
            }
        )

    payload = {
        "reference_log": str(Path(args.reference_log).resolve()),
        "candidate_log": str(Path(args.candidate_log).resolve()),
        "reference_status": reference.get("status"),
        "candidate_status": candidate.get("status"),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
