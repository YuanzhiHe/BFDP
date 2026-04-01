from __future__ import annotations

import argparse
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap on number of teacher episodes to replay. 0 means replay all episodes.",
    )
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
            payload["env_created"] = True
            payload["env_mode"] = "official_make_env"
        except BaseException as env_exc:
            payload["env_created"] = False
            payload["env_error_type"] = type(env_exc).__name__
            payload["env_error_message"] = str(env_exc)
            env = build_env_without_egl(args.dataset_path)
            payload["env_created"] = True
            payload["env_mode"] = "hydra_no_egl_fallback"

        conf_dir = (
            PROJECT_ROOT
            / "Experiment"
            / "code_references"
            / "calvin_workspace"
            / "calvin"
            / "calvin_models"
            / "conf"
        )
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        payload["task_oracle_created"] = True

        teacher_episodes = rollout_payload.get("episodes", [])
        if args.max_episodes > 0:
            teacher_episodes = teacher_episodes[: args.max_episodes]

        per_task_total: Counter[str] = Counter()
        per_task_success: Counter[str] = Counter()
        episode_results: list[dict[str, object]] = []

        for episode in teacher_episodes:
            steps = episode.get("steps", [])
            if not steps:
                continue
            first_step = steps[0]
            task_name = episode.get("task_name", "unknown")
            env.reset(
                robot_obs=np.asarray(first_step["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(first_step.get("scene_obs", []), dtype=np.float32),
            )
            start_info = env.get_info()
            success = False
            final_task_info: list[str] = []
            rollout_trace: list[dict[str, object]] = []
            for step_idx, step in enumerate(steps):
                _, _, _, next_info = env.step(step["action"])
                task_info = sorted(task_oracle.get_task_info_for_set(start_info, next_info, {task_name}))
                rollout_trace.append(
                    {
                        "step": step_idx,
                        "action": step["action"],
                        "task_info": task_info,
                    }
                )
                if task_info:
                    success = True
                    final_task_info = task_info
                    break
            per_task_total[task_name] += 1
            if success:
                per_task_success[task_name] += 1
            episode_results.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "episode_steps": len(steps),
                    "replayed_steps": len(rollout_trace),
                    "oracle_success": success,
                    "final_task_info": final_task_info,
                    "rollout_trace": rollout_trace,
                }
            )

        total_episodes = len(episode_results)
        total_successes = sum(1 for item in episode_results if item["oracle_success"])
        payload.update(
            {
                "status": "ok",
                "selection_mode": rollout_payload.get("selection_mode"),
                "episodes_replayed": total_episodes,
                "oracle_successes": total_successes,
                "oracle_success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
                "per_task_success": [
                    {
                        "task_name": task_name,
                        "success": per_task_success[task_name],
                        "total": total,
                        "success_rate": per_task_success[task_name] / total if total else 0.0,
                    }
                    for task_name, total in sorted(per_task_total.items())
                ],
                "episode_results": episode_results,
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
