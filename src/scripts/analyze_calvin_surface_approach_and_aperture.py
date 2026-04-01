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


TARGETS = {
    "turn_on_led": ("base__button", 0, "button_link"),
    "open_drawer": ("base__drawer", 3, "drawer_link"),
}

FINGERTIP_LINKS = {
    "finger_left_tip": 10,
    "finger_right_tip": 12,
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


def _target_info(env, task_name: str) -> tuple[int, int]:
    target_name, joint_index, _ = TARGETS[task_name]
    if task_name == "turn_on_led":
        obj = next(button for button in env.scene.buttons if button.name == target_name)
    else:
        obj = next(door for door in env.scene.doors if door.name == target_name)
    return int(obj.uid), int(joint_index)


def _link_world_pos(env, uid: int, link_index: int) -> list[float]:
    return list(env.p.getLinkState(uid, link_index, physicsClientId=env.cid)[0])


def _best_candidate(env, target_uid: int, target_link_index: int) -> dict[str, object] | None:
    target_pos, target_quat = env.p.getLinkState(target_uid, target_link_index, physicsClientId=env.cid)[:2]
    target_pos = list(target_pos)
    target_quat = list(target_quat)
    left_tip = _link_world_pos(env, env.robot.robot_uid, FINGERTIP_LINKS["finger_left_tip"])
    right_tip = _link_world_pos(env, env.robot.robot_uid, FINGERTIP_LINKS["finger_right_tip"])
    aperture = float(np.linalg.norm(np.asarray(left_tip) - np.asarray(right_tip)))

    best = None
    for finger_name, finger_link_index in FINGERTIP_LINKS.items():
        points = env.p.getClosestPoints(
            bodyA=env.robot.robot_uid,
            bodyB=target_uid,
            distance=0.5,
            linkIndexA=finger_link_index,
            linkIndexB=target_link_index,
            physicsClientId=env.cid,
        )
        if not points:
            continue
        candidate = min(points, key=lambda item: float(item[8]))
        pos_on_finger = list(candidate[5])
        pos_on_target = list(candidate[6])
        approach_world = [float(a - b) for a, b in zip(pos_on_finger, pos_on_target)]
        target_patch_world = [float(a - b) for a, b in zip(pos_on_target, target_pos)]
        info = {
            "finger_name": finger_name,
            "finger_link_index": finger_link_index,
            "distance": float(candidate[8]),
            "position_on_finger_world": pos_on_finger,
            "position_on_target_world": pos_on_target,
            "approach_delta_world": approach_world,
            "approach_delta_local": _world_to_local(approach_world, target_quat),
            "target_patch_delta_world": target_patch_world,
            "target_patch_delta_local": _world_to_local(target_patch_world, target_quat),
            "contact_normal_on_target_world": [float(x) for x in candidate[7]],
            "contact_normal_on_target_local": _world_to_local(list(candidate[7]), target_quat),
            "target_link_quaternion_world": target_quat,
            "fingertip_aperture": aperture,
        }
        if best is None or info["distance"] < best["distance"]:
            best = info
    return best


def _abs_mean(values: list[float]) -> float:
    return float(sum(abs(v) for v in values) / len(values))


def _signed_mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--rollout-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "status": "started",
    }

    try:
        rollout = _load_json(args.rollout_path)
        try:
            env = make_env(args.dataset_path)
            payload["env_mode"] = "official_make_env"
        except BaseException as env_exc:
            payload["env_error_type"] = type(env_exc).__name__
            payload["env_error_message"] = str(env_exc)
            env = build_env_without_egl(args.dataset_path)
            payload["env_mode"] = "hydra_no_egl_fallback"

        results = []
        per_task = defaultdict(list)

        for episode in rollout.get("episodes", []):
            task_name = str(episode.get("task_name", "unknown"))
            if task_name not in TARGETS:
                continue
            steps = episode.get("steps", [])
            if not steps:
                continue

            env.reset(
                robot_obs=np.asarray(steps[0]["robot_obs"], dtype=np.float32),
                scene_obs=np.asarray(steps[0].get("scene_obs", []), dtype=np.float32),
            )
            target_uid, target_link_index = _target_info(env, task_name)

            best = None
            for step_idx, step in enumerate(steps):
                candidate = _best_candidate(env, target_uid, target_link_index)
                if candidate is not None:
                    candidate["step"] = step_idx
                    if best is None or candidate["distance"] < best["distance"]:
                        best = candidate
                env.step(np.asarray(step["action"], dtype=np.float32))

            if best is None:
                continue

            result = {
                "sequence_index": int(episode["sequence_index"]),
                "matched_sequence_index": int(episode.get("matched_sequence_index", -1)),
                "task_name": task_name,
                "instruction": episode.get("instruction"),
                "best_surface_relation": best,
            }
            results.append(result)
            per_task[task_name].append(result)

        task_aggregates = {}
        for task_name, task_results in sorted(per_task.items()):
            approach_values = defaultdict(list)
            patch_values = defaultdict(list)
            apertures = []
            distances = []
            for item in task_results:
                best = item["best_surface_relation"]
                apertures.append(float(best["fingertip_aperture"]))
                distances.append(float(best["distance"]))
                for idx, value in enumerate(best["approach_delta_local"]):
                    approach_values[f"local_axis_{idx}"].append(float(value))
                for idx, value in enumerate(best["target_patch_delta_local"]):
                    patch_values[f"local_axis_{idx}"].append(float(value))
            task_aggregates[task_name] = {
                "episodes": len(task_results),
                "mean_best_distance": float(sum(distances) / len(distances)),
                "mean_best_aperture": float(sum(apertures) / len(apertures)),
                "signed_mean_approach_delta_local": {
                    axis: _signed_mean(values) for axis, values in sorted(approach_values.items())
                },
                "abs_mean_approach_delta_local": {
                    axis: _abs_mean(values) for axis, values in sorted(approach_values.items())
                },
                "signed_mean_target_patch_delta_local": {
                    axis: _signed_mean(values) for axis, values in sorted(patch_values.items())
                },
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
