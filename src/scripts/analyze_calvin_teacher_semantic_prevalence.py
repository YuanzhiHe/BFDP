from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _led_signature(surface: dict[str, object], approach: dict[str, object], envelope: dict[str, object]) -> dict[str, object]:
    local0 = float(approach["abs_mean_approach_delta_local"]["local_axis_0"])
    local1 = float(approach["abs_mean_approach_delta_local"]["local_axis_1"])
    distance = float(approach["mean_best_distance"])
    aperture = float(approach["mean_best_aperture"])
    surface_win = float(surface["intended_surface_win_rate_by_min"])
    avg_tip = float(envelope["mean_best_avg_fingertip_distance"])
    tcp = float(envelope["mean_best_tcp_distance"])
    side_biased_stand_off = local0 > local1 and distance > 0.10 and aperture > 0.09
    return {
        "episodes": int(surface["episodes"]),
        "surface_win_rate": surface_win,
        "mean_best_distance": distance,
        "mean_best_aperture": aperture,
        "abs_mean_local_axis_0": local0,
        "abs_mean_local_axis_1": local1,
        "avg_fingertip_minus_tcp_distance": float(avg_tip - tcp),
        "side_bias_margin": float(local0 - local1),
        "semantic_family": "side_biased_stand_off" if side_biased_stand_off else "not_side_biased_stand_off",
    }


def _drawer_signature(surface: dict[str, object], approach: dict[str, object], envelope: dict[str, object]) -> dict[str, object]:
    local1 = float(approach["abs_mean_approach_delta_local"]["local_axis_1"])
    local2 = float(approach["abs_mean_approach_delta_local"]["local_axis_2"])
    distance = float(approach["mean_best_distance"])
    aperture = float(approach["mean_best_aperture"])
    surface_win = float(surface["intended_surface_win_rate_by_min"])
    avg_tip = float(envelope["mean_best_avg_fingertip_distance"])
    tcp = float(envelope["mean_best_tcp_distance"])
    grazing = local1 > local2 and aperture > 0.10 and avg_tip > tcp and surface_win >= 1.0
    return {
        "episodes": int(surface["episodes"]),
        "surface_win_rate": surface_win,
        "mean_best_distance": distance,
        "mean_best_aperture": aperture,
        "abs_mean_local_axis_1": local1,
        "abs_mean_local_axis_2": local2,
        "avg_fingertip_minus_tcp_distance": float(avg_tip - tcp),
        "vertical_bias_margin": float(local1 - local2),
        "semantic_family": "open_gripper_grazing" if grazing else "not_open_gripper_grazing",
    }


def _tcp_summary(tcp_log: dict[str, object], task_name: str) -> dict[str, float]:
    items = [item for item in tcp_log.get("results", []) if str(item["task_name"]) == task_name]
    if not items:
        return {"episodes": 0, "mean_min_teacher_native_tcp_l2": 0.0, "mean_max_teacher_native_tcp_l2": 0.0}
    return {
        "episodes": len(items),
        "mean_min_teacher_native_tcp_l2": float(sum(float(item["min_teacher_native_tcp_l2"]) for item in items) / len(items)),
        "mean_max_teacher_native_tcp_l2": float(sum(float(item["max_teacher_native_tcp_l2"]) for item in items) / len(items)),
    }


def _pool_summary(surface_log: dict[str, object], approach_log: dict[str, object], envelope_log: dict[str, object], tcp_log: dict[str, object]) -> dict[str, object]:
    return {
        "turn_on_led": {
            "semantic_signature": _led_signature(
                surface_log["task_aggregates"]["turn_on_led"],
                approach_log["task_aggregates"]["turn_on_led"],
                envelope_log["task_aggregates"]["turn_on_led"],
            ),
            "tcp_alignment": _tcp_summary(tcp_log, "turn_on_led"),
        },
        "open_drawer": {
            "semantic_signature": _drawer_signature(
                surface_log["task_aggregates"]["open_drawer"],
                approach_log["task_aggregates"]["open_drawer"],
                envelope_log["task_aggregates"]["open_drawer"],
            ),
            "tcp_alignment": _tcp_summary(tcp_log, "open_drawer"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-surface-log", required=True)
    parser.add_argument("--state-approach-log", required=True)
    parser.add_argument("--state-envelope-log", required=True)
    parser.add_argument("--state-tcp-log", required=True)
    parser.add_argument("--mixed-surface-log", required=True)
    parser.add_argument("--mixed-approach-log", required=True)
    parser.add_argument("--mixed-envelope-log", required=True)
    parser.add_argument("--mixed-tcp-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {"status": "started"}

    try:
        state_summary = _pool_summary(
            _load_json(args.state_surface_log),
            _load_json(args.state_approach_log),
            _load_json(args.state_envelope_log),
            _load_json(args.state_tcp_log),
        )
        mixed_summary = _pool_summary(
            _load_json(args.mixed_surface_log),
            _load_json(args.mixed_approach_log),
            _load_json(args.mixed_envelope_log),
            _load_json(args.mixed_tcp_log),
        )

        payload.update(
            {
                "status": "ok",
                "state_matched_pool": state_summary,
                "mixed128_pool": mixed_summary,
                "overall": {
                    "turn_on_led_semantics_persist_from_state_to_mixed": (
                        state_summary["turn_on_led"]["semantic_signature"]["semantic_family"]
                        == "side_biased_stand_off"
                        and mixed_summary["turn_on_led"]["semantic_signature"]["semantic_family"]
                        == "side_biased_stand_off"
                    ),
                    "open_drawer_semantics_persist_from_state_to_mixed": (
                        state_summary["open_drawer"]["semantic_signature"]["semantic_family"]
                        == "open_gripper_grazing"
                        and mixed_summary["open_drawer"]["semantic_signature"]["semantic_family"]
                        == "open_gripper_grazing"
                    ),
                    "interpretation": (
                        "If both task families persist from the state-matched subset to the broader mixed pool, "
                        "the local teacher/export stack is likely encoding these semantics systematically rather than "
                        "only through a small native-matched diagnostic subset."
                    ),
                },
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
