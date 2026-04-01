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


def _link_name_map(body_uid: int, physics, cid: int) -> dict[int, str]:
    names = {-1: "base_link"}
    for idx in range(physics.getNumJoints(body_uid, physicsClientId=cid)):
        joint_info = physics.getJointInfo(body_uid, idx, physicsClientId=cid)
        names[idx] = joint_info[12].decode("utf-8")
    return names


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


def _pairs_with_names(contact_steps: list[dict[str, object]], robot_links: dict[int, str], target_links: dict[int, str]):
    pairs = {
        (pair["robot_link_index"], pair["target_link_index"])
        for step in contact_steps
        for pair in step["contact_pairs"]
    }
    result = []
    for robot_link_index, target_link_index in sorted(pairs):
        result.append(
            {
                "robot_link_index": robot_link_index,
                "robot_link_name": robot_links.get(robot_link_index, f"unknown_{robot_link_index}"),
                "target_link_index": target_link_index,
                "target_link_name": target_links.get(target_link_index, f"unknown_{target_link_index}"),
            }
        )
    return result


def _target_details(env, task_name: str) -> tuple[int, int, str]:
    if task_name == "turn_on_led":
        button = next(button for button in env.scene.buttons if button.effect == "led")
        return int(button.uid), int(button.joint_index), button.name
    if task_name == "open_drawer":
        drawer = next(door for door in env.scene.doors if door.name == "base__drawer")
        return int(drawer.uid), int(drawer.joint_index), drawer.name
    if task_name == "push_blue_block_right":
        block = next(obj for obj in env.scene.movable_objects if obj.name == "block_blue")
        return int(block.uid), -1, block.name
    raise ValueError(f"Unsupported task_name={task_name}")


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

        robot_links = _link_name_map(env.robot.robot_uid, env.p, env.cid)
        results: list[dict[str, object]] = []

        for episode in teacher_episodes:
            steps = episode.get("steps", [])
            if not steps:
                continue
            task_name = episode.get("task_name", "unknown")
            if task_name not in {"turn_on_led", "open_drawer", "push_blue_block_right"}:
                continue

            first_step = steps[0]
            env.reset(
                robot_obs=np.asarray(first_step["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(first_step.get("scene_obs", []), dtype=np.float32),
            )
            target_uid, target_joint_index, target_name = _target_details(env, task_name)
            target_links = _link_name_map(target_uid, env.p, env.cid)

            contact_steps = []
            for step_idx, step in enumerate(steps):
                _, _, _, info = env.step(np.asarray(step["action"], dtype=np.float32))
                pairs = _filtered_contacts(info["robot_info"]["contacts"], target_uid)
                if pairs:
                    contact_steps.append({"step": step_idx, "contact_pairs": pairs})

            results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "target_uid": target_uid,
                    "target_name": target_name,
                    "target_joint_index": target_joint_index,
                    "target_joint_link_name": target_links.get(target_joint_index, f"unknown_{target_joint_index}"),
                    "target_links": target_links,
                    "contact_steps": [item["step"] for item in contact_steps],
                    "contact_link_pairs": _pairs_with_names(contact_steps, robot_links, target_links),
                }
            )

        payload.update(
            {
                "status": "ok",
                "robot_links": robot_links,
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
