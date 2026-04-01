from __future__ import annotations

import argparse
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

import hydra
from calvin_agent.evaluation.evaluate_policy import make_env
from omegaconf import OmegaConf


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_env_without_egl(dataset_path: str):
    render_conf = OmegaConf.load(Path(dataset_path) / "validation" / ".hydra" / "merged_config.yaml")
    if "tactile" in render_conf.cameras:
        del render_conf.cameras["tactile"]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(
            str(
                (
                    PROJECT_ROOT
                    / "Experiment"
                    / "code_references"
                    / "calvin_workspace"
                    / "calvin"
                    / "calvin_env"
                    / "calvin_env"
                    / "envs"
                ).resolve()
            )
        )
    env = hydra.utils.instantiate(
        render_conf.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
        use_egl=False,
    )
    return env


def _robot_contact_uids(info: dict) -> set[int]:
    return {int(contact[2]) for contact in info["robot_info"]["contacts"]}


def _button_switch_specs(env, effect_name: str) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for button in getattr(env.scene, "buttons", []):
        if button.effect == effect_name:
            specs.append(
                {
                    "kind": "button",
                    "name": button.name,
                    "uid": int(button.uid),
                    "joint_index": int(button.joint_index),
                    "trigger_threshold": float(button.trigger_threshold),
                    "initial_state": float(button.initial_state),
                }
            )
    for switch in getattr(env.scene, "switches", []):
        if switch.effect == effect_name:
            specs.append(
                {
                    "kind": "switch",
                    "name": switch.name,
                    "uid": int(switch.uid),
                    "joint_index": int(switch.joint_index),
                    "trigger_threshold": float(switch.trigger_threshold),
                    "initial_state": float(switch.initial_state),
                }
            )
    return specs


def _joint_crossed_threshold(initial_state: float, trigger_threshold: float, values: list[float]) -> bool:
    if initial_state <= trigger_threshold:
        return any(value > trigger_threshold for value in values)
    return any(value < trigger_threshold for value in values)


def _analyze_turn_on_led(env, start_info: dict, history: list[dict]) -> dict[str, object]:
    specs = _button_switch_specs(env, "led")
    robot_contact_history = [_robot_contact_uids(item["info"]) for item in history]
    controls = []
    for spec in specs:
        joint_values = [
            float(item["info"]["scene_info"][f"{spec['kind']}s"][spec["name"]]["joint_state"]) for item in history
        ]
        controls.append(
            {
                **spec,
                "start_joint_state": float(start_info["scene_info"][f"{spec['kind']}s"][spec["name"]]["joint_state"]),
                "max_joint_state": max(joint_values) if joint_values else None,
                "min_joint_state": min(joint_values) if joint_values else None,
                "threshold_crossed": _joint_crossed_threshold(
                    spec["initial_state"],
                    spec["trigger_threshold"],
                    joint_values,
                ),
                "robot_contacts_control_uid_any_step": any(spec["uid"] in contacts for contacts in robot_contact_history),
            }
        )
    led_states = [int(item["info"]["scene_info"]["lights"]["led"]["logical_state"]) for item in history]
    return {
        "controls": controls,
        "max_led_state": max(led_states) if led_states else int(start_info["scene_info"]["lights"]["led"]["logical_state"]),
    }


def _analyze_open_drawer(env, start_info: dict, history: list[dict]) -> dict[str, object]:
    drawer_door = next(door for door in env.scene.doors if door.name == "base__drawer")
    robot_contact_history = [_robot_contact_uids(item["info"]) for item in history]
    joint_values = [float(item["info"]["scene_info"]["doors"]["base__drawer"]["current_state"]) for item in history]
    return {
        "drawer_uid": int(drawer_door.uid),
        "drawer_joint_index": int(drawer_door.joint_index),
        "start_joint_state": float(start_info["scene_info"]["doors"]["base__drawer"]["current_state"]),
        "max_joint_state": max(joint_values) if joint_values else None,
        "min_joint_state": min(joint_values) if joint_values else None,
        "robot_contacts_drawer_uid_any_step": any(int(drawer_door.uid) in contacts for contacts in robot_contact_history),
        "robot_contacts_drawer_uid_steps": [
            idx for idx, contacts in enumerate(robot_contact_history) if int(drawer_door.uid) in contacts
        ],
    }


def _analyze_push_blue_block_right(start_info: dict, history: list[dict]) -> dict[str, object]:
    robot_uid = int(start_info["robot_info"]["uid"])
    contact_steps = []
    blue_positions = []
    for idx, item in enumerate(history):
        blue_info = item["info"]["scene_info"]["movable_objects"]["block_blue"]
        blue_positions.append(list(blue_info["current_pos"]))
        contacts = {int(contact[2]) for contact in blue_info["contacts"]}
        if robot_uid in contacts:
            contact_steps.append(idx)
    start_pos = np.asarray(start_info["scene_info"]["movable_objects"]["block_blue"]["current_pos"], dtype=np.float64)
    deltas = [float(np.asarray(pos, dtype=np.float64)[0] - start_pos[0]) for pos in blue_positions]
    return {
        "robot_contacts_blue_block_any_step": len(contact_steps) > 0,
        "robot_contacts_blue_block_steps": contact_steps,
        "max_x_delta": max(deltas) if deltas else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    rollout_payload = _load_json(args.rollout_path)
    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "max_episodes": args.max_episodes,
        "status": "started",
    }

    try:
        try:
            env = make_env(args.dataset_path)
            payload["env_mode"] = "official_make_env"
        except BaseException as env_exc:
            payload["env_error_type"] = type(env_exc).__name__
            payload["env_error_message"] = str(env_exc)
            env = build_env_without_egl(args.dataset_path)
            payload["env_mode"] = "hydra_no_egl_fallback"

        teacher_episodes = rollout_payload.get("episodes", [])
        if args.max_episodes > 0:
            teacher_episodes = teacher_episodes[: args.max_episodes]

        results: list[dict[str, object]] = []
        for episode in teacher_episodes:
            steps = episode.get("steps", [])
            if not steps:
                continue
            task_name = episode.get("task_name", "unknown")
            first_step = steps[0]
            env.reset(
                robot_obs=np.asarray(first_step["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(first_step.get("scene_obs", []), dtype=np.float32),
            )
            start_info = env.get_info()
            history: list[dict[str, object]] = []
            for step_idx, step in enumerate(steps):
                _, _, _, info = env.step(np.asarray(step["action"], dtype=np.float32))
                history.append({"step": step_idx, "info": info})

            interaction = {}
            if task_name == "turn_on_led":
                interaction = _analyze_turn_on_led(env, start_info, history)
            elif task_name == "open_drawer":
                interaction = _analyze_open_drawer(env, start_info, history)
            elif task_name == "push_blue_block_right":
                interaction = _analyze_push_blue_block_right(start_info, history)

            results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "interaction": interaction,
                }
            )

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
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
