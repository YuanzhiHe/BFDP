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


def _unique_pairs(contact_steps: list[dict[str, object]]) -> list[dict[str, int]]:
    pairs = {
        (pair["robot_link_index"], pair["target_link_index"])
        for step in contact_steps
        for pair in step["contact_pairs"]
    }
    return [
        {"robot_link_index": robot_link_index, "target_link_index": target_link_index}
        for robot_link_index, target_link_index in sorted(pairs)
    ]


def _analyze_turn_on_led(env, history: list[dict[str, object]]) -> dict[str, object]:
    button = next(button for button in env.scene.buttons if button.effect == "led")
    tcp_distances = []
    contact_steps = []
    for item in history:
        tcp_pos = item["info"]["robot_info"]["tcp_pos"]
        button_pos = env.p.getLinkState(button.uid, button.joint_index, physicsClientId=env.cid)[0]
        tcp_distances.append(_distance(tcp_pos, button_pos))
        pairs = _filtered_contacts(item["info"]["robot_info"]["contacts"], int(button.uid))
        if pairs:
            contact_steps.append({"step": item["step"], "contact_pairs": pairs})
    return {
        "target_kind": "button",
        "target_name": button.name,
        "target_uid": int(button.uid),
        "target_joint_index": int(button.joint_index),
        "min_tcp_distance_to_target": min(tcp_distances) if tcp_distances else math.inf,
        "contact_steps": [item["step"] for item in contact_steps],
        "unique_contact_link_pairs": _unique_pairs(contact_steps),
    }


def _analyze_open_drawer(env, history: list[dict[str, object]]) -> dict[str, object]:
    drawer = next(door for door in env.scene.doors if door.name == "base__drawer")
    tcp_distances = []
    contact_steps = []
    for item in history:
        tcp_pos = item["info"]["robot_info"]["tcp_pos"]
        drawer_pos = env.p.getLinkState(drawer.uid, drawer.joint_index, physicsClientId=env.cid)[0]
        tcp_distances.append(_distance(tcp_pos, drawer_pos))
        pairs = _filtered_contacts(item["info"]["robot_info"]["contacts"], int(drawer.uid))
        if pairs:
            contact_steps.append({"step": item["step"], "contact_pairs": pairs})
    return {
        "target_kind": "drawer_joint_link",
        "target_name": drawer.name,
        "target_uid": int(drawer.uid),
        "target_joint_index": int(drawer.joint_index),
        "min_tcp_distance_to_target": min(tcp_distances) if tcp_distances else math.inf,
        "contact_steps": [item["step"] for item in contact_steps],
        "unique_contact_link_pairs": _unique_pairs(contact_steps),
    }


def _analyze_push_blue_block_right(history: list[dict[str, object]]) -> dict[str, object]:
    tcp_distances = []
    contact_steps = []
    target_uid = None
    for item in history:
        tcp_pos = item["info"]["robot_info"]["tcp_pos"]
        block_info = item["info"]["scene_info"]["movable_objects"]["block_blue"]
        target_uid = int(block_info["uid"])
        block_pos = block_info["current_pos"]
        tcp_distances.append(_distance(tcp_pos, block_pos))
        pairs = _filtered_contacts(item["info"]["robot_info"]["contacts"], target_uid)
        if pairs:
            contact_steps.append({"step": item["step"], "contact_pairs": pairs})
    return {
        "target_kind": "movable_object",
        "target_name": "block_blue",
        "target_uid": target_uid,
        "min_tcp_distance_to_target": min(tcp_distances) if tcp_distances else math.inf,
        "contact_steps": [item["step"] for item in contact_steps],
        "unique_contact_link_pairs": _unique_pairs(contact_steps),
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
            history: list[dict[str, object]] = []
            for step_idx, step in enumerate(steps):
                _, _, _, info = env.step(np.asarray(step["action"], dtype=np.float32))
                history.append({"step": step_idx, "info": info})

            geometry = {}
            if task_name == "turn_on_led":
                geometry = _analyze_turn_on_led(env, history)
            elif task_name == "open_drawer":
                geometry = _analyze_open_drawer(env, history)
            elif task_name == "push_blue_block_right":
                geometry = _analyze_push_blue_block_right(history)

            results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "geometry": geometry,
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
