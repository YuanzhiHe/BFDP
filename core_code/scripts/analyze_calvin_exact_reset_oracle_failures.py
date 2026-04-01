from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
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


def _non_robot_contacts(obj_info: dict, robot_uid: int) -> set[tuple[int, int]]:
    return {(contact[2], contact[4]) for contact in obj_info["contacts"] if contact[2] != robot_uid}


def _analyze_turn_on_led(start_info: dict, end_info: dict, history: list[dict]) -> dict[str, object]:
    led_start = start_info["scene_info"]["lights"]["led"]["logical_state"]
    led_end = end_info["scene_info"]["lights"]["led"]["logical_state"]
    max_led = max(
        item["info"]["scene_info"]["lights"]["led"]["logical_state"] for item in history
    ) if history else led_start
    return {
        "oracle_family": "toggle_light",
        "led_start": led_start,
        "led_end": led_end,
        "max_led_state_during_replay": max_led,
        "required_end_state": 1,
        "criterion_satisfied": led_start == 0 and led_end == 1,
    }


def _analyze_open_drawer(start_info: dict, end_info: dict, history: list[dict]) -> dict[str, object]:
    start_joint = float(start_info["scene_info"]["doors"]["base__drawer"]["current_state"])
    end_joint = float(end_info["scene_info"]["doors"]["base__drawer"]["current_state"])
    joint_history = [
        float(item["info"]["scene_info"]["doors"]["base__drawer"]["current_state"]) for item in history
    ]
    max_joint = max(joint_history) if joint_history else start_joint
    return {
        "oracle_family": "move_door_rel",
        "joint_name": "base__drawer",
        "start_joint_state": start_joint,
        "end_joint_state": end_joint,
        "max_joint_state_during_replay": max_joint,
        "delta_final": end_joint - start_joint,
        "delta_max": max_joint - start_joint,
        "required_delta": 0.12,
        "criterion_satisfied": (end_joint - start_joint) > 0.12,
    }


def _analyze_push_blue_block_right(start_info: dict, end_info: dict, history: list[dict]) -> dict[str, object]:
    robot_uid = int(start_info["robot_info"]["uid"])
    start_obj = start_info["scene_info"]["movable_objects"]["block_blue"]
    end_obj = end_info["scene_info"]["movable_objects"]["block_blue"]
    start_pos = np.asarray(start_obj["current_pos"], dtype=np.float64)
    end_pos = np.asarray(end_obj["current_pos"], dtype=np.float64)
    x_delta_final = float(end_pos[0] - start_pos[0])
    x_positions = [
        float(item["info"]["scene_info"]["movable_objects"]["block_blue"]["current_pos"][0]) for item in history
    ]
    max_x_delta = max((value - start_pos[0]) for value in x_positions) if x_positions else x_delta_final
    start_contacts = _non_robot_contacts(start_obj, robot_uid)
    end_contacts = _non_robot_contacts(end_obj, robot_uid)
    history_surface_contact = [
        _non_robot_contacts(item["info"]["scene_info"]["movable_objects"]["block_blue"], robot_uid)
        for item in history
    ]
    preserved_surface_contact = len(start_contacts) > 0 and len(end_contacts) > 0 and start_contacts <= end_contacts
    preserved_surface_contact_any = any(
        len(start_contacts) > 0 and len(step_contacts) > 0 and start_contacts <= step_contacts
        for step_contacts in history_surface_contact
    )
    return {
        "oracle_family": "push_object",
        "object_name": "block_blue",
        "start_pos": start_pos.tolist(),
        "end_pos": end_pos.tolist(),
        "x_delta_final": x_delta_final,
        "x_delta_max": float(max_x_delta),
        "required_x_delta": 0.1,
        "start_contacts_excluding_robot": sorted(list(start_contacts)),
        "end_contacts_excluding_robot": sorted(list(end_contacts)),
        "surface_contact_preserved_final": preserved_surface_contact,
        "surface_contact_preserved_any_step": preserved_surface_contact_any,
        "criterion_satisfied": preserved_surface_contact and x_delta_final > 0.1,
    }


def _analyze_episode(task_name: str, start_info: dict, end_info: dict, history: list[dict]) -> dict[str, object]:
    if task_name == "turn_on_led":
        return _analyze_turn_on_led(start_info, end_info, history)
    if task_name == "open_drawer":
        return _analyze_open_drawer(start_info, end_info, history)
    if task_name == "push_blue_block_right":
        return _analyze_push_blue_block_right(start_info, end_info, history)
    return {
        "oracle_family": "unsupported",
        "criterion_satisfied": False,
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

        per_task_counter: Counter[str] = Counter()
        analysis_results: list[dict[str, object]] = []
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
            end_info = start_info
            for step_idx, step in enumerate(steps):
                _, _, _, end_info = env.step(np.asarray(step["action"], dtype=np.float32))
                history.append(
                    {
                        "step": step_idx,
                        "info": end_info,
                    }
                )
            oracle_analysis = _analyze_episode(task_name, start_info, end_info, history)
            analysis_results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "oracle_analysis": oracle_analysis,
                }
            )
            per_task_counter[task_name] += 1

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(analysis_results),
                "task_counts": dict(per_task_counter),
                "analysis_results": analysis_results,
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
