from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
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


TASK_TARGETS = {
    "turn_on_led": ("base__button", 0, "button_link"),
    "open_drawer": ("base__drawer", 3, "drawer_link"),
}

ROBOT_LINKS = {
    "finger_left_tip": 10,
    "finger_right_tip": 12,
    "tcp": 15,
}


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


def _robot_link_position(env, link_index: int) -> list[float]:
    position = env.p.getLinkState(env.robot.robot_uid, link_index, physicsClientId=env.cid)[0]
    return list(position)


def _target_info(env, task_name: str) -> tuple[int, int, str]:
    target_name, joint_index, _ = TASK_TARGETS[task_name]
    if task_name == "turn_on_led":
        obj = next(button for button in env.scene.buttons if button.name == target_name)
    else:
        obj = next(door for door in env.scene.doors if door.name == target_name)
    return int(obj.uid), int(joint_index), target_name


def _target_position(env, target_uid: int, link_index: int) -> list[float]:
    position = env.p.getLinkState(target_uid, link_index, physicsClientId=env.cid)[0]
    return list(position)


def _step_geometry(env, task_name: str) -> dict[str, object]:
    target_uid, target_link_index, target_name = _target_info(env, task_name)
    target_pos = _target_position(env, target_uid, target_link_index)

    left_tip = _robot_link_position(env, ROBOT_LINKS["finger_left_tip"])
    right_tip = _robot_link_position(env, ROBOT_LINKS["finger_right_tip"])
    tcp = _robot_link_position(env, ROBOT_LINKS["tcp"])
    avg_tip = list(((np.asarray(left_tip) + np.asarray(right_tip)) / 2.0).tolist())

    positions = {
        "target": target_pos,
        "tcp": tcp,
        "finger_left_tip": left_tip,
        "finger_right_tip": right_tip,
        "avg_fingertip": avg_tip,
    }

    deltas = {
        key: [float(a - b) for a, b in zip(value, target_pos)]
        for key, value in positions.items()
        if key != "target"
    }
    distances = {
        key: _distance(value, target_pos)
        for key, value in positions.items()
        if key != "target"
    }

    nearest_actor = min(distances, key=distances.get)
    return {
        "target_name": target_name,
        "target_link_name": TASK_TARGETS[task_name][2],
        "positions": positions,
        "deltas": deltas,
        "distances": distances,
        "nearest_actor": nearest_actor,
    }


def _episode_summary(step_rows: list[dict[str, object]]) -> dict[str, object]:
    actor_names = list(step_rows[0]["distances"].keys())
    min_distance = {
        actor_name: float(min(row["distances"][actor_name] for row in step_rows))
        for actor_name in actor_names
    }
    mean_distance = {
        actor_name: float(sum(row["distances"][actor_name] for row in step_rows) / len(step_rows))
        for actor_name in actor_names
    }
    nearest_step_counts = {
        actor_name: int(sum(row["nearest_actor"] == actor_name for row in step_rows))
        for actor_name in actor_names
    }
    best_avg_step = min(step_rows, key=lambda row: row["distances"]["avg_fingertip"])
    return {
        "min_distance_by_actor": min_distance,
        "mean_distance_by_actor": mean_distance,
        "nearest_actor_by_min_distance": min(min_distance, key=min_distance.get),
        "nearest_actor_by_mean_distance": min(mean_distance, key=mean_distance.get),
        "nearest_actor_step_counts": nearest_step_counts,
        "best_avg_fingertip_step": int(best_avg_step["step"]),
        "best_avg_fingertip_distance": float(best_avg_step["distances"]["avg_fingertip"]),
        "best_avg_fingertip_delta": best_avg_step["deltas"]["avg_fingertip"],
        "best_tcp_distance": float(min_distance["tcp"]),
        "best_tcp_delta_at_best_avg_fingertip_step": best_avg_step["deltas"]["tcp"],
    }


def _aggregate(task_results: list[dict[str, object]]) -> dict[str, object]:
    if not task_results:
        return {}
    min_avg_tip = [item["episode_summary"]["best_avg_fingertip_distance"] for item in task_results]
    min_tcp = [item["episode_summary"]["best_tcp_distance"] for item in task_results]
    nearest_mean_counts = defaultdict(int)
    for item in task_results:
        nearest_mean_counts[item["episode_summary"]["nearest_actor_by_mean_distance"]] += 1
    return {
        "episodes": len(task_results),
        "mean_best_avg_fingertip_distance": float(sum(min_avg_tip) / len(min_avg_tip)),
        "mean_best_tcp_distance": float(sum(min_tcp) / len(min_tcp)),
        "nearest_actor_by_mean_distance_counts": dict(sorted(nearest_mean_counts.items())),
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

        results = []
        per_task = defaultdict(list)
        for episode in teacher_episodes:
            steps = episode.get("steps", [])
            task_name = str(episode.get("task_name", "unknown"))
            if not steps or task_name not in TASK_TARGETS:
                continue

            env.reset(
                robot_obs=np.asarray(steps[0]["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(steps[0].get("scene_obs", []), dtype=np.float32),
            )

            step_rows = []
            for step_idx, step in enumerate(steps):
                geometry = _step_geometry(env, task_name)
                geometry["step"] = step_idx
                step_rows.append(geometry)
                env.step(np.asarray(step["action"], dtype=np.float32))

            episode_result = {
                "sequence_index": episode.get("sequence_index"),
                "matched_sequence_index": episode.get("matched_sequence_index"),
                "task_name": task_name,
                "instruction": episode.get("instruction"),
                "episode_summary": _episode_summary(step_rows),
                "per_step": step_rows,
            }
            results.append(episode_result)
            per_task[task_name].append(episode_result)

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
                "task_aggregates": {
                    task_name: _aggregate(task_results)
                    for task_name, task_results in sorted(per_task.items())
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
