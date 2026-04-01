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


def _target_info(env, task_name: str) -> tuple[int, int, str]:
    target_name, joint_index, link_name = TASK_TARGETS[task_name]
    if task_name == "turn_on_led":
        obj = next(button for button in env.scene.buttons if button.name == target_name)
    else:
        obj = next(door for door in env.scene.doors if door.name == target_name)
    return int(obj.uid), int(joint_index), link_name


def _rotation_matrix_from_quat(quat: list[float] | tuple[float, ...]) -> np.ndarray:
    x, y, z, w = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_to_local(delta_world: list[float], quat_world: list[float]) -> list[float]:
    rotation = _rotation_matrix_from_quat(quat_world)
    local = rotation.T @ np.asarray(delta_world, dtype=np.float64)
    return [float(value) for value in local.tolist()]


def _link_state(env, uid: int, link_index: int) -> tuple[list[float], list[float]]:
    position, orientation = env.p.getLinkState(uid, link_index, physicsClientId=env.cid)[:2]
    return list(position), list(orientation)


def _axis_name(idx: int) -> str:
    return f"local_axis_{idx}"


def _abs_mean(values: list[float]) -> float:
    return float(sum(abs(value) for value in values) / len(values))


def _signed_mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--envelope-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "envelope_log": str(Path(args.envelope_log).resolve()),
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

        rollout = _load_json(args.rollout_path)
        envelope = _load_json(args.envelope_log)
        envelope_by_sequence = {
            (int(item["sequence_index"]), str(item["task_name"])): item
            for item in envelope.get("results", [])
        }

        results = []
        per_task: dict[str, list[dict[str, object]]] = defaultdict(list)

        for episode in rollout.get("episodes", []):
            task_name = str(episode.get("task_name", "unknown"))
            if task_name not in TASK_TARGETS:
                continue

            key = (int(episode["sequence_index"]), task_name)
            envelope_episode = envelope_by_sequence.get(key)
            if envelope_episode is None:
                continue

            best_step = int(envelope_episode["episode_summary"]["best_avg_fingertip_step"])
            steps = episode.get("steps", [])
            if best_step >= len(steps):
                continue

            env.reset(
                robot_obs=np.asarray(steps[0]["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(steps[0].get("scene_obs", []), dtype=np.float32),
            )

            target_uid, target_link_index, target_link_name = _target_info(env, task_name)
            row_at_best = None
            for step_idx, step in enumerate(steps):
                target_pos, target_quat = _link_state(env, target_uid, target_link_index)
                if step_idx == best_step:
                    row_at_best = next(
                        row for row in envelope_episode["per_step"] if int(row["step"]) == best_step
                    )
                    tcp_delta_world = row_at_best["deltas"]["tcp"]
                    avg_tip_delta_world = row_at_best["deltas"]["avg_fingertip"]
                    tcp_delta_local = _world_to_local(tcp_delta_world, target_quat)
                    avg_tip_delta_local = _world_to_local(avg_tip_delta_world, target_quat)

                    tcp_abs_local = [abs(value) for value in tcp_delta_local]
                    avg_tip_abs_local = [abs(value) for value in avg_tip_delta_local]
                    tcp_dominant_idx = max(range(3), key=lambda idx: tcp_abs_local[idx])
                    avg_tip_dominant_idx = max(range(3), key=lambda idx: avg_tip_abs_local[idx])

                    result = {
                        "sequence_index": int(episode["sequence_index"]),
                        "matched_sequence_index": int(episode.get("matched_sequence_index", -1)),
                        "task_name": task_name,
                        "instruction": episode.get("instruction"),
                        "best_step": best_step,
                        "target_link_name": target_link_name,
                        "target_quaternion_world": target_quat,
                        "tcp_delta_world": tcp_delta_world,
                        "avg_fingertip_delta_world": avg_tip_delta_world,
                        "tcp_delta_local": tcp_delta_local,
                        "avg_fingertip_delta_local": avg_tip_delta_local,
                        "tcp_abs_delta_local": tcp_abs_local,
                        "avg_fingertip_abs_delta_local": avg_tip_abs_local,
                        "tcp_dominant_local_axis": _axis_name(tcp_dominant_idx),
                        "avg_fingertip_dominant_local_axis": _axis_name(avg_tip_dominant_idx),
                    }
                    results.append(result)
                    per_task[task_name].append(result)
                    break
                env.step(np.asarray(step["action"], dtype=np.float32))

            if row_at_best is None:
                continue

        task_aggregates = {}
        for task_name, task_results in sorted(per_task.items()):
            tcp_values = defaultdict(list)
            avg_tip_values = defaultdict(list)
            tcp_dom = defaultdict(int)
            avg_tip_dom = defaultdict(int)
            for item in task_results:
                for idx in range(3):
                    axis = _axis_name(idx)
                    tcp_values[axis].append(float(item["tcp_delta_local"][idx]))
                    avg_tip_values[axis].append(float(item["avg_fingertip_delta_local"][idx]))
                tcp_dom[str(item["tcp_dominant_local_axis"])] += 1
                avg_tip_dom[str(item["avg_fingertip_dominant_local_axis"])] += 1
            task_aggregates[task_name] = {
                "episodes": len(task_results),
                "tcp_signed_mean_delta_local": {
                    axis: _signed_mean(values) for axis, values in sorted(tcp_values.items())
                },
                "tcp_abs_mean_delta_local": {
                    axis: _abs_mean(values) for axis, values in sorted(tcp_values.items())
                },
                "avg_fingertip_signed_mean_delta_local": {
                    axis: _signed_mean(values) for axis, values in sorted(avg_tip_values.items())
                },
                "avg_fingertip_abs_mean_delta_local": {
                    axis: _abs_mean(values) for axis, values in sorted(avg_tip_values.items())
                },
                "tcp_dominant_local_axis_counts": dict(sorted(tcp_dom.items())),
                "avg_fingertip_dominant_local_axis_counts": dict(sorted(avg_tip_dom.items())),
            }

        payload.update(
            {
                "status": "ok",
                "episodes_analyzed": len(results),
                "task_aggregates": task_aggregates,
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
