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


def _target_position(env, task_name: str) -> tuple[str, list[float]]:
    if task_name == "turn_on_led":
        button = next(button for button in env.scene.buttons if button.effect == "led")
        position = env.p.getLinkState(button.uid, button.joint_index, physicsClientId=env.cid)[0]
        return button.name, list(position)
    if task_name == "open_drawer":
        drawer = next(door for door in env.scene.doors if door.name == "base__drawer")
        position = env.p.getLinkState(drawer.uid, drawer.joint_index, physicsClientId=env.cid)[0]
        return drawer.name, list(position)
    if task_name == "push_blue_block_right":
        block = next(obj for obj in env.scene.movable_objects if obj.name == "block_blue")
        position = env.p.getBasePositionAndOrientation(block.uid, physicsClientId=env.cid)[0]
        return block.name, list(position)
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
            per_step = []
            target_name = None
            for step_idx, step in enumerate(steps):
                target_name, target_pos = _target_position(env, task_name)
                teacher_tcp = list(step["robot_obs"][:3])
                _, _, _, info = env.step(np.asarray(step["action"], dtype=np.float32))
                native_tcp = list(info["robot_info"]["tcp_pos"])
                per_step.append(
                    {
                        "step": step_idx,
                        "teacher_tcp": teacher_tcp,
                        "native_tcp": native_tcp,
                        "teacher_native_tcp_mae": sum(abs(a - b) for a, b in zip(teacher_tcp, native_tcp)) / 3.0,
                        "teacher_native_tcp_l2": _distance(teacher_tcp, native_tcp),
                        "target_pos": target_pos,
                        "teacher_tcp_to_target_l2": _distance(teacher_tcp, target_pos),
                        "native_tcp_to_target_l2": _distance(native_tcp, target_pos),
                    }
                )

            results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "target_name": target_name,
                    "min_teacher_native_tcp_l2": min(item["teacher_native_tcp_l2"] for item in per_step),
                    "max_teacher_native_tcp_l2": max(item["teacher_native_tcp_l2"] for item in per_step),
                    "min_native_tcp_to_target_l2": min(item["native_tcp_to_target_l2"] for item in per_step),
                    "min_teacher_tcp_to_target_l2": min(item["teacher_tcp_to_target_l2"] for item in per_step),
                    "per_step": per_step,
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
