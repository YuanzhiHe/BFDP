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
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_agent.evaluation.evaluate_policy import make_env
from omegaconf import OmegaConf


ROBOT_EULER_SELECTED_INDICES = {3, 4, 5}
SCENE_ORIENTATION_SELECTED_INDICES = {
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
}


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_env_without_egl(dataset_path: str):
    render_conf = OmegaConf.load(Path(dataset_path) / "validation" / ".hydra" / "merged_config.yaml")
    if "tactile" in render_conf.cameras:
        del render_conf.cameras["tactile"]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(str((PROJECT_ROOT / "Experiment" / "code_references" / "calvin_workspace" / "calvin" / "calvin_env" / "calvin_env" / "envs").resolve()))
    env = hydra.utils.instantiate(
        render_conf.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
        use_egl=False,
    )
    return env


def _wrapped_delta(a: float, b: float) -> float:
    raw = a - b
    return ((raw + math.pi) % (2.0 * math.pi)) - math.pi


def _wrapped_vector_stats(a: list[float], b: list[float]) -> dict[str, float | int]:
    count = min(len(a), len(b))
    if count == 0:
        return {"count": 0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    diffs = []
    for idx in range(count):
        if idx in ROBOT_EULER_SELECTED_INDICES or idx in SCENE_ORIENTATION_SELECTED_INDICES:
            diffs.append(_wrapped_delta(float(a[idx]), float(b[idx])))
        else:
            diffs.append(float(a[idx]) - float(b[idx]))
    abs_diffs = [abs(value) for value in diffs]
    return {
        "count": count,
        "mae": sum(abs_diffs) / count,
        "mse": sum(value * value for value in diffs) / count,
        "max_abs": max(abs_diffs),
    }


def _extract_selected_obs(obs: dict) -> list[float]:
    return list(obs["robot_obs"][:9]) + list(obs.get("scene_obs", [])[:24])


def _match_teacher_episode(episodes: list[dict], subtask: str, instruction: str) -> dict | None:
    for episode in episodes:
        if episode.get("task_name") == subtask and episode.get("instruction") == instruction:
            return episode
    for episode in episodes:
        if episode.get("task_name") == subtask:
            return episode
    return None


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
        except BaseException:
            env = build_env_without_egl(args.dataset_path)
            payload["env_mode"] = "hydra_no_egl_fallback"

        rollout_payload = _load_json(args.rollout_path)
        sequence_payload = _load_json(args.sequence_log)
        teacher_episodes = rollout_payload.get("episodes", [])
        comparisons: list[dict[str, object]] = []

        for sequence in sequence_payload.get("sequence_results", []):
            if not sequence.get("subtask_results"):
                continue
            subtask = sequence["subtask_results"][0]
            teacher_episode = _match_teacher_episode(
                teacher_episodes,
                subtask["subtask"],
                subtask["language_annotation"],
            )
            if teacher_episode is None or not teacher_episode.get("steps"):
                continue

            teacher_initial_step = teacher_episode["steps"][0]
            teacher_selected_obs = _extract_selected_obs(
                {
                    "robot_obs": teacher_initial_step["robot_obs"],
                    "scene_obs": teacher_initial_step.get("scene_obs", []),
                }
            )

            env.reset(
                robot_obs=np.asarray(teacher_initial_step["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(teacher_initial_step.get("scene_obs", []), dtype=np.float32),
            )
            exact_reset_obs = env.get_obs()
            exact_reset_selected_obs = _extract_selected_obs(exact_reset_obs)
            zero_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            exact_next_obs, _, _, _ = env.step(zero_action)
            exact_next_selected_obs = _extract_selected_obs(exact_next_obs)

            symbolic_robot_obs, symbolic_scene_obs = get_env_state_for_initial_condition(sequence["initial_state"])
            env.reset(
                robot_obs=np.asarray(symbolic_robot_obs, dtype=np.float32),
                scene_obs=np.asarray(symbolic_scene_obs, dtype=np.float32),
            )
            symbolic_reset_obs = env.get_obs()
            symbolic_reset_selected_obs = _extract_selected_obs(symbolic_reset_obs)
            symbolic_next_obs, _, _, _ = env.step(zero_action)
            symbolic_next_selected_obs = _extract_selected_obs(symbolic_next_obs)

            comparisons.append(
                {
                    "subtask": subtask["subtask"],
                    "language_annotation": subtask["language_annotation"],
                    "teacher_sequence_index": teacher_episode.get("sequence_index"),
                    "native_sequence_index": sequence["sequence_index"],
                    "exact_reset_vs_teacher_initial": _wrapped_vector_stats(
                        exact_reset_selected_obs,
                        teacher_selected_obs,
                    ),
                    "exact_next_vs_teacher_initial": _wrapped_vector_stats(
                        exact_next_selected_obs,
                        teacher_selected_obs,
                    ),
                    "symbolic_reset_vs_teacher_initial": _wrapped_vector_stats(
                        symbolic_reset_selected_obs,
                        teacher_selected_obs,
                    ),
                    "symbolic_next_vs_teacher_initial": _wrapped_vector_stats(
                        symbolic_next_selected_obs,
                        teacher_selected_obs,
                    ),
                    "exact_next_vs_exact_reset": _wrapped_vector_stats(
                        exact_next_selected_obs,
                        exact_reset_selected_obs,
                    ),
                    "symbolic_next_vs_symbolic_reset": _wrapped_vector_stats(
                        symbolic_next_selected_obs,
                        symbolic_reset_selected_obs,
                    ),
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
