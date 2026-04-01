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


def _vector_stats(a: list[float], b: list[float]) -> dict[str, float | int]:
    count = min(len(a), len(b))
    if count == 0:
        return {"count": 0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    diffs = [float(a[idx]) - float(b[idx]) for idx in range(count)]
    abs_diffs = [abs(value) for value in diffs]
    return {
        "count": count,
        "mae": sum(abs_diffs) / count,
        "mse": sum(value * value for value in diffs) / count,
        "max_abs": max(abs_diffs),
    }


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


def _selected_obs(step: dict) -> list[float]:
    return list(step["robot_obs"][:9]) + list(step.get("scene_obs", [])[:24])


def _obs_dict_to_lists(obs: dict) -> dict[str, list[float]]:
    return {
        "robot_obs": list(np.asarray(obs["robot_obs"], dtype=np.float32).tolist()),
        "scene_obs": list(np.asarray(obs.get("scene_obs", []), dtype=np.float32).tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap on number of episodes to compare. 0 means all usable episodes.",
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

        episodes = [
            episode for episode in rollout_payload.get("episodes", []) if len(episode.get("steps", [])) >= 2
        ]
        if args.max_episodes > 0:
            episodes = episodes[: args.max_episodes]

        comparisons: list[dict[str, object]] = []
        task_counts: Counter[str] = Counter()
        over_0p05_counts: Counter[str] = Counter()
        over_0p10_counts: Counter[str] = Counter()

        for episode in episodes:
            first_step = episode["steps"][0]
            second_step = episode["steps"][1]
            env.reset(
                robot_obs=np.asarray(first_step["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(first_step.get("scene_obs", []), dtype=np.float32),
            )
            next_obs, _, _, _ = env.step(first_step["action"])
            next_obs_lists = _obs_dict_to_lists(next_obs)
            dataset_next_selected_obs = _selected_obs(second_step)
            native_next_selected_obs = next_obs_lists["robot_obs"][:9] + next_obs_lists["scene_obs"][:24]

            selected_stats = _vector_stats(native_next_selected_obs, dataset_next_selected_obs)
            robot_stats = _vector_stats(next_obs_lists["robot_obs"], second_step["robot_obs"])
            scene_stats = _vector_stats(next_obs_lists["scene_obs"], second_step.get("scene_obs", []))

            task_name = episode.get("task_name", "unknown")
            task_counts[task_name] += 1
            if selected_stats["mae"] > 0.05:
                over_0p05_counts[task_name] += 1
            if selected_stats["mae"] > 0.10:
                over_0p10_counts[task_name] += 1

            comparisons.append(
                {
                    "sequence_index": episode.get("sequence_index"),
                    "matched_sequence_index": episode.get("matched_sequence_index"),
                    "task_name": task_name,
                    "instruction": episode.get("instruction"),
                    "first_action": first_step["action"],
                    "dataset_next_selected_obs": dataset_next_selected_obs,
                    "native_next_selected_obs": native_next_selected_obs,
                    "selected_obs_delta": selected_stats,
                    "robot_obs_delta": robot_stats,
                    "scene_obs_delta": scene_stats,
                }
            )

        payload.update(
            {
                "status": "ok",
                "selection_mode": rollout_payload.get("selection_mode"),
                "episodes_compared": len(comparisons),
                "selected_obs_mae_mean": (
                    sum(item["selected_obs_delta"]["mae"] for item in comparisons) / len(comparisons)
                    if comparisons
                    else 0.0
                ),
                "selected_obs_mae_median_proxy_sorted": sorted(
                    item["selected_obs_delta"]["mae"] for item in comparisons
                ),
                "per_task_selected_obs_over_0p05_rate": {
                    task: over_0p05_counts[task] / total if total else 0.0
                    for task, total in sorted(task_counts.items())
                },
                "per_task_selected_obs_over_0p10_rate": {
                    task: over_0p10_counts[task] / total if total else 0.0
                    for task, total in sorted(task_counts.items())
                },
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
