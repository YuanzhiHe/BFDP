from __future__ import annotations

import argparse
import json
import math
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
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
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


def _distance(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _match_teacher_episode(episodes: list[dict], subtask: str, instruction: str) -> dict | None:
    for episode in episodes:
        if episode.get("task_name") == subtask and episode.get("instruction") == instruction:
            return episode
    for episode in episodes:
        if episode.get("task_name") == subtask:
            return episode
    return None


def _filtered_contacts(robot_contacts: list, target_uid: int) -> list[dict[str, int]]:
    filtered = []
    for contact in robot_contacts:
        if int(contact[2]) != target_uid:
            continue
        filtered.append(
            {
                "robot_link_index": int(contact[3]),
                "target_link_index": int(contact[4]),
            }
        )
    return filtered


def _task_target(env, task_name: str) -> tuple[int, int | None]:
    if task_name == "turn_on_led":
        button = next(button for button in env.scene.buttons if button.effect == "led")
        return int(button.uid), int(button.joint_index)
    if task_name == "open_drawer":
        door = next(door for door in env.scene.doors if door.name == "base__drawer")
        return int(door.uid), int(door.joint_index)
    return -1, None


def _task_effect_snapshot(task_name: str, info: dict) -> dict[str, float | int]:
    if task_name == "turn_on_led":
        return {
            "led_state": int(info["scene_info"]["lights"]["led"]["logical_state"]),
        }
    if task_name == "open_drawer":
        return {
            "drawer_joint_state": float(info["scene_info"]["doors"]["base__drawer"]["current_state"]),
        }
    return {}


def _summarize_task_effect(task_name: str, start_info: dict, history: list[dict]) -> dict[str, object]:
    if task_name == "turn_on_led":
        start_led = int(start_info["scene_info"]["lights"]["led"]["logical_state"])
        led_states = [int(item["info"]["scene_info"]["lights"]["led"]["logical_state"]) for item in history]
        success_step = next((item["step"] for item in history if int(item["info"]["scene_info"]["lights"]["led"]["logical_state"]) == 1), None)
        return {
            "start_led_state": start_led,
            "max_led_state": max(led_states) if led_states else start_led,
            "final_led_state": led_states[-1] if led_states else start_led,
            "success_step": success_step,
        }
    if task_name == "open_drawer":
        start_joint = float(start_info["scene_info"]["doors"]["base__drawer"]["current_state"])
        joint_states = [float(item["info"]["scene_info"]["doors"]["base__drawer"]["current_state"]) for item in history]
        deltas = [value - start_joint for value in joint_states]
        success_step = next((item["step"] for item, delta in zip(history, deltas) if delta > 0.12), None)
        return {
            "start_joint_state": start_joint,
            "max_joint_state": max(joint_states) if joint_states else start_joint,
            "final_joint_state": joint_states[-1] if joint_states else start_joint,
            "max_joint_delta": max(deltas) if deltas else 0.0,
            "final_joint_delta": deltas[-1] if deltas else 0.0,
            "success_step": success_step,
        }
    return {}


def _run_episode(env, task_name: str, robot_obs: list[float], scene_obs: list[float], actions: list[list[float]]) -> dict[str, object]:
    env.reset(
        robot_obs=np.asarray(robot_obs, dtype=np.float32),
        scene_obs=np.asarray(scene_obs, dtype=np.float32),
    )
    start_info = env.get_info()
    target_uid, target_link_index = _task_target(env, task_name)
    history: list[dict[str, object]] = []
    min_tcp_distance = math.inf
    first_target_contact_step = None

    for step_idx, action in enumerate(actions):
        _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        step_record: dict[str, object] = {
            "step": step_idx,
            "task_effect": _task_effect_snapshot(task_name, info),
        }
        if target_uid >= 0 and target_link_index is not None:
            tcp_pos = info["robot_info"]["tcp_pos"]
            target_pos = env.p.getLinkState(target_uid, target_link_index, physicsClientId=env.cid)[0]
            tcp_distance = _distance(tcp_pos, target_pos)
            min_tcp_distance = min(min_tcp_distance, tcp_distance)
            contact_pairs = _filtered_contacts(info["robot_info"]["contacts"], target_uid)
            has_target_contact = len(contact_pairs) > 0
            if has_target_contact and first_target_contact_step is None:
                first_target_contact_step = step_idx
            step_record.update(
                {
                    "tcp_distance_to_target": tcp_distance,
                    "target_contact_pairs": contact_pairs,
                }
            )
        history.append(
            {
                "step": step_idx,
                "info": info,
                "step_record": step_record,
            }
        )

    return {
        "task_effect_summary": _summarize_task_effect(task_name, start_info, history),
        "min_tcp_distance_to_target": None if min_tcp_distance is math.inf else min_tcp_distance,
        "first_target_contact_step": first_target_contact_step,
        "history": [item["step_record"] for item in history],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--sequence-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "sequence_log": str(Path(args.sequence_log).resolve()),
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

        rollout_payload = _load_json(args.rollout_path)
        sequence_payload = _load_json(args.sequence_log)
        teacher_episodes = rollout_payload.get("episodes", [])
        comparisons: list[dict[str, object]] = []

        for sequence in sequence_payload.get("sequence_results", []):
            if not sequence.get("subtask_results"):
                continue
            subtask_result = sequence["subtask_results"][0]
            task_name = subtask_result["subtask"]
            instruction = subtask_result["language_annotation"]
            teacher_episode = _match_teacher_episode(teacher_episodes, task_name, instruction)
            if teacher_episode is None or not teacher_episode.get("steps"):
                continue

            teacher_steps = teacher_episode["steps"]
            actions = [step["action"] for step in teacher_steps]
            exact_first_step = teacher_steps[0]
            symbolic_robot_obs, symbolic_scene_obs = get_env_state_for_initial_condition(sequence["initial_state"])

            exact_result = _run_episode(
                env=env,
                task_name=task_name,
                robot_obs=exact_first_step["robot_obs"],
                scene_obs=exact_first_step.get("scene_obs", []),
                actions=actions,
            )
            symbolic_result = _run_episode(
                env=env,
                task_name=task_name,
                robot_obs=symbolic_robot_obs,
                scene_obs=symbolic_scene_obs,
                actions=actions,
            )

            comparisons.append(
                {
                    "task_name": task_name,
                    "language_annotation": instruction,
                    "native_sequence_index": sequence.get("sequence_index"),
                    "teacher_sequence_index": teacher_episode.get("sequence_index"),
                    "teacher_source_sequence_index": teacher_episode.get("source_sequence_index"),
                    "initial_state": sequence.get("initial_state"),
                    "exact_reset": exact_result,
                    "symbolic_reset": symbolic_result,
                }
            )

        payload.update(
            {
                "status": "ok",
                "comparison_count": len(comparisons),
                "comparisons": comparisons,
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
