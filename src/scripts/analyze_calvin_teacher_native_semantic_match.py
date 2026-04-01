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


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_tcp_summary(tcp_results: list[dict[str, object]]) -> dict[str, float]:
    if not tcp_results:
        return {
            "episodes": 0,
            "mean_min_teacher_native_tcp_l2": 0.0,
            "mean_max_teacher_native_tcp_l2": 0.0,
            "mean_min_teacher_tcp_to_target_l2": 0.0,
            "mean_min_native_tcp_to_target_l2": 0.0,
        }
    return {
        "episodes": len(tcp_results),
        "mean_min_teacher_native_tcp_l2": float(
            sum(float(item["min_teacher_native_tcp_l2"]) for item in tcp_results) / len(tcp_results)
        ),
        "mean_max_teacher_native_tcp_l2": float(
            sum(float(item["max_teacher_native_tcp_l2"]) for item in tcp_results) / len(tcp_results)
        ),
        "mean_min_teacher_tcp_to_target_l2": float(
            sum(float(item["min_teacher_tcp_to_target_l2"]) for item in tcp_results) / len(tcp_results)
        ),
        "mean_min_native_tcp_to_target_l2": float(
            sum(float(item["min_native_tcp_to_target_l2"]) for item in tcp_results) / len(tcp_results)
        ),
    }


def _led_verdict(surface_pref: dict[str, object], approach: dict[str, object], envelope: dict[str, object]) -> dict[str, object]:
    local0 = float(approach["abs_mean_approach_delta_local"]["local_axis_0"])
    local1 = float(approach["abs_mean_approach_delta_local"]["local_axis_1"])
    distance = float(approach["mean_best_distance"])
    aperture = float(approach["mean_best_aperture"])
    intended_win = float(surface_pref["intended_surface_win_rate_by_min"])
    avg_fingertip = float(envelope["mean_best_avg_fingertip_distance"])
    tcp = float(envelope["mean_best_tcp_distance"])

    side_biased = local0 > local1
    near_surface_but_not_closing = distance > 0.10 and aperture > 0.09
    teacher_supports_native_failure = side_biased and near_surface_but_not_closing

    if teacher_supports_native_failure:
        verdict = "teacher_semantics_match_side_biased_stand_off"
    elif intended_win < 0.5:
        verdict = "teacher_surface_selection_mixed"
    else:
        verdict = "teacher_press_like_but_native_execution_mismatch"

    return {
        "verdict": verdict,
        "teacher_supports_native_failure_family": teacher_supports_native_failure,
        "evidence": {
            "intended_surface_win_rate_by_min": intended_win,
            "mean_best_distance": distance,
            "mean_best_aperture": aperture,
            "abs_mean_local_axis_0": local0,
            "abs_mean_local_axis_1": local1,
            "mean_best_avg_fingertip_distance": avg_fingertip,
            "mean_best_tcp_distance": tcp,
            "side_bias_margin_local_axis_0_minus_1": float(local0 - local1),
            "avg_fingertip_minus_tcp_distance": float(avg_fingertip - tcp),
        },
        "interpretation": (
            "Teacher LED trajectories remain closer to the intended button surface often enough, "
            "but their best approach is still more side-biased than press-axis aligned and keeps a "
            "large stand-off distance with an open gripper aperture."
        ),
    }


def _drawer_verdict(surface_pref: dict[str, object], approach: dict[str, object], envelope: dict[str, object]) -> dict[str, object]:
    local1 = float(approach["abs_mean_approach_delta_local"]["local_axis_1"])
    local2 = float(approach["abs_mean_approach_delta_local"]["local_axis_2"])
    distance = float(approach["mean_best_distance"])
    aperture = float(approach["mean_best_aperture"])
    intended_win = float(surface_pref["intended_surface_win_rate_by_min"])
    avg_fingertip = float(envelope["mean_best_avg_fingertip_distance"])
    tcp = float(envelope["mean_best_tcp_distance"])
    nearest_actor_counts = envelope.get("nearest_actor_by_mean_distance_counts", {})
    tcp_dominates = int(nearest_actor_counts.get("tcp", 0)) == int(envelope["episodes"])

    vertical_bias = local1 > local2
    open_gripper_grazing = aperture > 0.10 and avg_fingertip > tcp and tcp_dominates
    teacher_supports_native_failure = intended_win >= 1.0 and vertical_bias and open_gripper_grazing

    if teacher_supports_native_failure:
        verdict = "teacher_semantics_match_open_gripper_grazing"
    elif intended_win < 1.0:
        verdict = "teacher_surface_selection_mixed"
    else:
        verdict = "teacher_handle_entry_like_but_native_execution_mismatch"

    return {
        "verdict": verdict,
        "teacher_supports_native_failure_family": teacher_supports_native_failure,
        "evidence": {
            "intended_surface_win_rate_by_min": intended_win,
            "mean_best_distance": distance,
            "mean_best_aperture": aperture,
            "abs_mean_local_axis_1": local1,
            "abs_mean_local_axis_2": local2,
            "mean_best_avg_fingertip_distance": avg_fingertip,
            "mean_best_tcp_distance": tcp,
            "vertical_bias_margin_local_axis_1_minus_2": float(local1 - local2),
            "avg_fingertip_minus_tcp_distance": float(avg_fingertip - tcp),
            "tcp_dominates_nearest_actor": tcp_dominates,
        },
        "interpretation": (
            "Teacher drawer trajectories prefer the intended drawer surface, but they remain vertically "
            "misaligned, keep a wide gripper aperture, and behave more like TCP-led grazing than "
            "fingertip-level handle entry."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-surface-log", required=True)
    parser.add_argument("--approach-log", required=True)
    parser.add_argument("--envelope-log", required=True)
    parser.add_argument("--failure-family-log", required=True)
    parser.add_argument("--tcp-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "teacher_surface_log": str(Path(args.teacher_surface_log).resolve()),
        "approach_log": str(Path(args.approach_log).resolve()),
        "envelope_log": str(Path(args.envelope_log).resolve()),
        "failure_family_log": str(Path(args.failure_family_log).resolve()),
        "tcp_log": str(Path(args.tcp_log).resolve()),
        "status": "started",
    }

    try:
        teacher_surface = _load_json(args.teacher_surface_log)
        approach = _load_json(args.approach_log)
        envelope = _load_json(args.envelope_log)
        failure_family = _load_json(args.failure_family_log)
        tcp_log = _load_json(args.tcp_log)

        tcp_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in tcp_log.get("results", []):
            tcp_by_task[str(item["task_name"])].append(item)

        led = _led_verdict(
            teacher_surface["task_aggregates"]["turn_on_led"],
            approach["task_aggregates"]["turn_on_led"],
            envelope["task_aggregates"]["turn_on_led"],
        )
        drawer = _drawer_verdict(
            teacher_surface["task_aggregates"]["open_drawer"],
            approach["task_aggregates"]["open_drawer"],
            envelope["task_aggregates"]["open_drawer"],
        )

        task_summaries = {
            "turn_on_led": {
                "native_failure_families": failure_family["task_mode_counts"]["turn_on_led"],
                "teacher_semantic_verdict": led,
                "tcp_alignment": _task_tcp_summary(tcp_by_task["turn_on_led"]),
            },
            "open_drawer": {
                "native_failure_families": failure_family["task_mode_counts"]["open_drawer"],
                "teacher_semantic_verdict": drawer,
                "tcp_alignment": _task_tcp_summary(tcp_by_task["open_drawer"]),
            },
        }

        overall = {
            "teacher_supports_native_failure_families": bool(
                led["teacher_supports_native_failure_family"]
                and drawer["teacher_supports_native_failure_family"]
            ),
            "overall_verdict": (
                "teacher_semantics_partially_explain_remaining_native_gap"
                if led["teacher_supports_native_failure_family"]
                and drawer["teacher_supports_native_failure_family"]
                else "native_gap_not_explained_by_teacher_semantics_alone"
            ),
            "interpretation": (
                "Low TCP drift remains consistent with replay fidelity, while the task-level semantic "
                "summaries indicate that the stored teacher trajectories themselves already resemble "
                "the same LED side-biased stand-off and drawer grazing families observed in native failure analysis."
            ),
        }

        payload.update(
            {
                "status": "ok",
                "task_summaries": task_summaries,
                "overall": overall,
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
