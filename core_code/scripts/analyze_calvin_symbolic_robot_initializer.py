from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _build_calvin_pythonpath() -> list[str]:
    workspace_root = (
        PROJECT_ROOT / "Experiment" / "code_references" / "calvin_workspace" / "calvin"
    ).resolve()
    return [
        str(workspace_root),
        str((workspace_root / "calvin_env").resolve()),
        str((workspace_root / "calvin_models").resolve()),
    ]


for path in reversed(_build_calvin_pythonpath()):
    if path not in sys.path:
        sys.path.insert(0, path)

from calvin_agent.evaluation.utils import get_env_state_for_initial_condition


ROBOT_LABELS = [
    "tcp_pos_x",
    "tcp_pos_y",
    "tcp_pos_z",
    "tcp_orn_x",
    "tcp_orn_y",
    "tcp_orn_z",
    "gripper_opening_width",
    "robot_joint_0",
    "robot_joint_1",
    "robot_joint_2",
    "robot_joint_3",
    "robot_joint_4",
    "robot_joint_5",
    "robot_joint_6",
    "gripper_action",
]

JOINT_BANDS = {
    "joint_prefix_7_10": slice(7, 10),
    "joint_prefix_7_11": slice(7, 11),
    "joint_prefix_7_12": slice(7, 12),
    "joint_band_6_14": slice(6, 14),
}


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_initial_conditions() -> list[dict[str, object]]:
    sliders = ["left", "right"]
    drawers = ["closed", "open"]
    lightbulbs = [0, 1]
    leds = [0, 1]
    grasped = [0, 1]
    placements = ["table", "slider_left", "slider_right"]
    sampled = []
    for slider, drawer, lightbulb, led, grasped_flag, red_block, blue_block, pink_block in itertools.product(
        sliders,
        drawers,
        lightbulbs,
        leds,
        grasped,
        placements,
        placements,
        placements,
    ):
        sampled.append(
            {
                "slider": slider,
                "drawer": drawer,
                "lightbulb": lightbulb,
                "led": led,
                "grasped": grasped_flag,
                "red_block": red_block,
                "blue_block": blue_block,
                "pink_block": pink_block,
            }
        )
        if len(sampled) >= 64:
            break
    return sampled


def _initializer_consistency_report() -> dict[str, object]:
    conditions = _sample_initial_conditions()
    robot_states = []
    for condition in conditions:
        robot_obs, _ = get_env_state_for_initial_condition(condition)
        robot_states.append(np.asarray(robot_obs, dtype=np.float64))
    stacked = np.stack(robot_states, axis=0)
    first = stacked[0]
    max_abs_delta = float(np.max(np.abs(stacked - first)))
    return {
        "sampled_initial_condition_count": len(conditions),
        "unique_robot_obs_count": len({tuple(np.round(row, 8).tolist()) for row in stacked}),
        "max_abs_delta_across_conditions": max_abs_delta,
        "canonical_robot_obs": [float(value) for value in first.tolist()],
        "canonical_robot_labels": ROBOT_LABELS,
        "sampled_initial_conditions": conditions[:8],
    }


def _band_report(exact_robot_stack: np.ndarray, canonical_robot: np.ndarray, band_name: str) -> dict[str, object]:
    band = JOINT_BANDS[band_name]
    exact = exact_robot_stack[:, band]
    canonical = canonical_robot[band]
    deltas = exact - canonical
    mean_abs_delta_by_dim = np.mean(np.abs(deltas), axis=0)
    return {
        "labels": ROBOT_LABELS[band],
        "canonical_values": [float(value) for value in canonical.tolist()],
        "teacher_mean": [float(value) for value in np.mean(exact, axis=0).tolist()],
        "teacher_std": [float(value) for value in np.std(exact, axis=0).tolist()],
        "mean_abs_delta_vs_canonical": [float(value) for value in mean_abs_delta_by_dim.tolist()],
        "episode_l2_to_canonical": [float(value) for value in np.linalg.norm(deltas, axis=1).tolist()],
        "mean_l2_to_canonical": float(np.mean(np.linalg.norm(deltas, axis=1))),
    }


def _episode_prefix_report(episode: dict[str, object], canonical_robot: np.ndarray) -> dict[str, object]:
    exact_robot = np.asarray(episode["steps"][0]["robot_obs"], dtype=np.float64)
    return {
        "sequence_index": episode.get("sequence_index"),
        "source_sequence_index": episode.get("source_sequence_index"),
        "instruction": episode.get("instruction"),
        "task_name": episode.get("task_name"),
        "joint_prefix_7_10_exact": [float(value) for value in exact_robot[7:10].tolist()],
        "joint_prefix_7_10_delta_vs_canonical": [float(value) for value in (exact_robot[7:10] - canonical_robot[7:10]).tolist()],
        "joint_prefix_7_11_exact": [float(value) for value in exact_robot[7:11].tolist()],
        "joint_prefix_7_11_delta_vs_canonical": [float(value) for value in (exact_robot[7:11] - canonical_robot[7:11]).tolist()],
    }


def _rollout_report(rollout_payload: dict[str, object], canonical_robot: np.ndarray) -> dict[str, object]:
    episodes = rollout_payload.get("episodes", [])
    exact_robot_stack = np.stack(
        [np.asarray(episode["steps"][0]["robot_obs"], dtype=np.float64) for episode in episodes],
        axis=0,
    )
    return {
        "episode_count": len(episodes),
        "task_names": sorted({str(episode.get("task_name")) for episode in episodes}),
        "joint_bands": {
            name: _band_report(exact_robot_stack, canonical_robot, name)
            for name in JOINT_BANDS
        },
        "episodes": [_episode_prefix_report(episode, canonical_robot) for episode in episodes],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-on-led-rollout", required=True)
    parser.add_argument("--open-drawer-rollout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "turn_on_led_rollout": str(Path(args.turn_on_led_rollout).resolve()),
        "open_drawer_rollout": str(Path(args.open_drawer_rollout).resolve()),
        "status": "started",
    }

    try:
        initializer_report = _initializer_consistency_report()
        canonical_robot = np.asarray(initializer_report["canonical_robot_obs"], dtype=np.float64)
        payload.update(
            {
                "status": "ok",
                "initializer_consistency": initializer_report,
                "turn_on_led_exact_rollout": _rollout_report(
                    _load_json(args.turn_on_led_rollout), canonical_robot
                ),
                "open_drawer_exact_rollout": _rollout_report(
                    _load_json(args.open_drawer_rollout), canonical_robot
                ),
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
