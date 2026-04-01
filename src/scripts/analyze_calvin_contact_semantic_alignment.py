from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _key(item: dict[str, object]) -> tuple[int, str]:
    return int(item["sequence_index"]), str(item["task_name"])


def _led_failure_mode(local_item: dict[str, object], surface_item: dict[str, object], prereq_item: dict[str, object]) -> str:
    dominant = str(local_item["avg_fingertip_dominant_local_axis"])
    contact_pairs = surface_item.get("contact_link_pairs", [])
    contact_targets = {str(pair["target_link_name"]) for pair in contact_pairs}
    control = prereq_item["interaction"]["controls"][0]
    has_any_control_contact = bool(control["robot_contacts_control_uid_any_step"])
    threshold_crossed = bool(control["threshold_crossed"])

    if threshold_crossed:
        return "unexpected_threshold_cross"
    if "slide_link" in contact_targets and "button_link" not in contact_targets:
        return "side_collision_on_slide_link"
    if dominant == "local_axis_0":
        return "side_approach_miss"
    if dominant == "local_axis_1" and not has_any_control_contact:
        return "press_axis_shortfall_without_contact"
    if dominant == "local_axis_2":
        return "cross_axis_misalignment"
    if not has_any_control_contact:
        return "no_control_contact"
    return "unclassified_led_failure"


def _drawer_failure_mode(local_item: dict[str, object], prereq_item: dict[str, object]) -> str:
    dominant = str(local_item["avg_fingertip_dominant_local_axis"])
    interaction = prereq_item["interaction"]
    has_contact = bool(interaction["robot_contacts_drawer_uid_any_step"])

    if has_contact:
        return "unexpected_drawer_contact"
    if dominant == "local_axis_1":
        return "vertical_envelope_miss"
    if dominant == "local_axis_2":
        return "pull_direction_shortfall_after_height_miss"
    if dominant == "local_axis_0":
        return "lateral_handle_offset"
    return "unclassified_drawer_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-axis-log", required=True)
    parser.add_argument("--surface-log", required=True)
    parser.add_argument("--prereq-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "local_axis_log": str(Path(args.local_axis_log).resolve()),
        "surface_log": str(Path(args.surface_log).resolve()),
        "prereq_log": str(Path(args.prereq_log).resolve()),
        "status": "started",
    }

    try:
        local_data = _load_json(args.local_axis_log)
        surface_data = _load_json(args.surface_log)
        prereq_data = _load_json(args.prereq_log)

        surface_by_key = {_key(item): item for item in surface_data.get("results", [])}
        prereq_by_key = {_key(item): item for item in prereq_data.get("results", [])}

        results = []
        per_task_modes: dict[str, list[str]] = defaultdict(list)

        for local_item in local_data.get("results", []):
            task_name = str(local_item["task_name"])
            if task_name not in {"turn_on_led", "open_drawer"}:
                continue
            key = _key(local_item)
            surface_item = surface_by_key[key]
            prereq_item = prereq_by_key[key]

            if task_name == "turn_on_led":
                failure_mode = _led_failure_mode(local_item, surface_item, prereq_item)
            else:
                failure_mode = _drawer_failure_mode(local_item, prereq_item)

            result = {
                "sequence_index": int(local_item["sequence_index"]),
                "matched_sequence_index": int(local_item["matched_sequence_index"]),
                "task_name": task_name,
                "instruction": local_item["instruction"],
                "dominant_local_axis": str(local_item["avg_fingertip_dominant_local_axis"]),
                "contact_link_pairs": surface_item.get("contact_link_pairs", []),
                "failure_mode": failure_mode,
            }
            results.append(result)
            per_task_modes[task_name].append(failure_mode)

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
                "task_mode_counts": {
                    task_name: dict(sorted(Counter(modes).items()))
                    for task_name, modes in sorted(per_task_modes.items())
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
