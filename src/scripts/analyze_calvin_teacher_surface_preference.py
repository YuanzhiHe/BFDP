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


TABLE_SURFACE_LINKS = {
    "button_link": 0,
    "switch_link": 1,
    "slide_link": 2,
    "drawer_link": 3,
}

TASK_SURFACES = {
    "turn_on_led": {
        "intended_surface": "button_link",
        "candidate_surfaces": ["button_link", "switch_link", "slide_link", "drawer_link"],
    },
    "open_drawer": {
        "intended_surface": "drawer_link",
        "candidate_surfaces": ["drawer_link", "slide_link", "button_link", "switch_link"],
    },
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


def _table_uid(env) -> int:
    # Buttons, switch, slider, and drawer are all links on the same table body.
    return int(next(button for button in env.scene.buttons if button.effect == "led").uid)


def _surface_position(env, table_uid: int, link_index: int) -> list[float]:
    position = env.p.getLinkState(table_uid, link_index, physicsClientId=env.cid)[0]
    return list(position)


def _step_surface_distances(env, teacher_tcp: list[float], candidate_surfaces: list[str]) -> dict[str, float]:
    table_uid = _table_uid(env)
    distances = {}
    for surface_name in candidate_surfaces:
        link_index = TABLE_SURFACE_LINKS[surface_name]
        distances[surface_name] = _distance(teacher_tcp, _surface_position(env, table_uid, link_index))
    return distances


def _surface_summary(step_distances: list[dict[str, float]], intended_surface: str) -> dict[str, object]:
    mins = {
        surface_name: float(min(item[surface_name] for item in step_distances))
        for surface_name in step_distances[0]
    }
    means = {
        surface_name: float(sum(item[surface_name] for item in step_distances) / len(step_distances))
        for surface_name in step_distances[0]
    }
    per_step_nearest = [
        min(item, key=item.get)
        for item in step_distances
    ]
    nearest_surface_by_min = min(mins, key=mins.get)
    nearest_surface_by_mean = min(means, key=means.get)
    return {
        "min_distance_by_surface": mins,
        "mean_distance_by_surface": means,
        "nearest_surface_by_min": nearest_surface_by_min,
        "nearest_surface_by_mean": nearest_surface_by_mean,
        "intended_surface": intended_surface,
        "intended_surface_is_nearest_by_min": nearest_surface_by_min == intended_surface,
        "intended_surface_is_nearest_by_mean": nearest_surface_by_mean == intended_surface,
        "nearest_surface_step_counts": {
            surface_name: int(per_step_nearest.count(surface_name))
            for surface_name in step_distances[0]
        },
    }


def _aggregate_task(task_results: list[dict[str, object]]) -> dict[str, object]:
    if not task_results:
        return {}

    intended_surface = str(task_results[0]["surface_summary"]["intended_surface"])
    nearest_min_counts = defaultdict(int)
    nearest_mean_counts = defaultdict(int)
    intended_min_wins = 0
    intended_mean_wins = 0
    margin_by_min = []
    margin_by_mean = []

    for episode in task_results:
        summary = episode["surface_summary"]
        nearest_min = summary["nearest_surface_by_min"]
        nearest_mean = summary["nearest_surface_by_mean"]
        nearest_min_counts[nearest_min] += 1
        nearest_mean_counts[nearest_mean] += 1
        intended_min_wins += int(summary["intended_surface_is_nearest_by_min"])
        intended_mean_wins += int(summary["intended_surface_is_nearest_by_mean"])

        mins = summary["min_distance_by_surface"]
        means = summary["mean_distance_by_surface"]
        sorted_mins = sorted(mins.items(), key=lambda item: item[1])
        sorted_means = sorted(means.items(), key=lambda item: item[1])
        best_other_min = next(value for name, value in sorted_mins if name != intended_surface)
        best_other_mean = next(value for name, value in sorted_means if name != intended_surface)
        margin_by_min.append(float(mins[intended_surface] - best_other_min))
        margin_by_mean.append(float(means[intended_surface] - best_other_mean))

    return {
        "episodes": len(task_results),
        "intended_surface": intended_surface,
        "nearest_surface_by_min_counts": dict(sorted(nearest_min_counts.items())),
        "nearest_surface_by_mean_counts": dict(sorted(nearest_mean_counts.items())),
        "intended_surface_win_rate_by_min": float(intended_min_wins / len(task_results)),
        "intended_surface_win_rate_by_mean": float(intended_mean_wins / len(task_results)),
        "mean_intended_minus_best_other_min_margin": float(sum(margin_by_min) / len(margin_by_min)),
        "mean_intended_minus_best_other_mean_margin": float(sum(margin_by_mean) / len(margin_by_mean)),
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
        per_task: dict[str, list[dict[str, object]]] = defaultdict(list)

        for episode in teacher_episodes:
            steps = episode.get("steps", [])
            if not steps:
                continue

            task_name = str(episode.get("task_name", "unknown"))
            if task_name not in TASK_SURFACES:
                continue

            config = TASK_SURFACES[task_name]
            env.reset(
                robot_obs=np.asarray(steps[0]["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(steps[0].get("scene_obs", []), dtype=np.float32),
            )

            step_rows = []
            for step_idx, step in enumerate(steps):
                teacher_tcp = list(step["robot_obs"][:3])
                surface_distances = _step_surface_distances(env, teacher_tcp, config["candidate_surfaces"])
                step_rows.append(
                    {
                        "step": step_idx,
                        "teacher_tcp": teacher_tcp,
                        "surface_distances": surface_distances,
                        "nearest_surface": min(surface_distances, key=surface_distances.get),
                    }
                )
                env.step(np.asarray(step["action"], dtype=np.float32))

            surface_summary = _surface_summary(
                [row["surface_distances"] for row in step_rows],
                config["intended_surface"],
            )
            episode_result = {
                "sequence_index": episode.get("sequence_index"),
                "matched_sequence_index": episode.get("matched_sequence_index"),
                "task_name": task_name,
                "instruction": episode.get("instruction"),
                "surface_summary": surface_summary,
                "per_step": step_rows,
            }
            results.append(episode_result)
            per_task[task_name].append(episode_result)

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
                "task_aggregates": {
                    task_name: _aggregate_task(task_results)
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
