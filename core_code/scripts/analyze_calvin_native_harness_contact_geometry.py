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
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
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


def _link_world_state(env, uid: int, link_index: int) -> tuple[list[float], list[float]]:
    state = env.p.getLinkState(uid, link_index, physicsClientId=env.cid)
    return list(state[0]), list(state[1])


def _best_candidate(env, target_uid: int, target_link_index: int) -> dict[str, object] | None:
    target_pos, target_quat = _link_world_state(env, target_uid, target_link_index)
    left_tip = _link_world_state(env, env.robot.robot_uid, FINGERTIP_LINKS["finger_left_tip"])[0]
    right_tip = _link_world_state(env, env.robot.robot_uid, FINGERTIP_LINKS["finger_right_tip"])[0]
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
        patch_world = [float(a - b) for a, b in zip(pos_on_target, target_pos)]
        info = {
            "finger_name": finger_name,
            "finger_link_index": finger_link_index,
            "distance": float(candidate[8]),
            "position_on_target_world": pos_on_target,
            "approach_delta_local": _world_to_local(approach_world, target_quat),
            "target_patch_delta_local": _world_to_local(patch_world, target_quat),
            "target_link_quaternion_world": target_quat,
            "fingertip_aperture": aperture,
        }
        if best is None or info["distance"] < best["distance"]:
            best = info
    return best


def _contact_pairs_with_target(env, target_uid: int) -> list[dict[str, int]]:
    pairs = []
    for contact in env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=target_uid, physicsClientId=env.cid):
        pairs.append(
            {
                "robot_link_index": int(contact[3]),
                "target_link_index": int(contact[4]),
            }
        )
    return pairs


def _target_link_name(target_uid: int, link_index: int, env, task_name: str) -> str:
    target_name, target_joint_index, target_link_name = TARGETS[task_name]
    if link_index == target_joint_index:
        return target_link_name
    if link_index == -1:
        return "base_link"
    for obj in list(env.scene.buttons) + list(env.scene.doors):
        if int(obj.uid) == int(target_uid) and obj.name == target_name:
            if link_index == target_joint_index:
                return target_link_name
    return f"link_{link_index}"


def _classify(task_name: str, best_candidate: dict[str, object] | None, all_pairs: list[dict[str, object]]) -> str:
    if best_candidate is None:
        return "no_surface_candidate"
    axis_magnitudes = [abs(float(value)) for value in best_candidate["approach_delta_local"]]
    dominant_axis = max(range(3), key=lambda idx: axis_magnitudes[idx])
    target_link_names = {str(pair["target_link_name"]) for pair in all_pairs}
    if task_name == "turn_on_led":
        if not target_link_names and float(best_candidate["distance"]) >= 0.12:
            return "button_stand_off_without_contact"
        if "slide_link" in target_link_names and "button_link" not in target_link_names:
            return "side_collision_on_slide_link"
        if dominant_axis == 0:
            return "side_biased_stand_off"
        if dominant_axis == 1:
            return "press_axis_shortfall"
        return "cross_axis_button_miss"
    if "drawer_link" in target_link_names:
        return "drawer_link_contact_without_opening"
    if dominant_axis == 1:
        return "vertical_handle_miss"
    if dominant_axis == 2:
        return "pull_direction_shortfall"
    return "lateral_handle_offset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--harness-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "harness_log": str(Path(args.harness_log).resolve()),
        "status": "started",
    }

    try:
        harness = _load_json(args.harness_log)
        try:
            env = make_env(args.dataset_path)
            payload["env_mode"] = "official_make_env"
        except BaseException as env_exc:
            payload["env_error_type"] = type(env_exc).__name__
            payload["env_error_message"] = str(env_exc)
            env = build_env_without_egl(args.dataset_path)
            payload["env_mode"] = "hydra_no_egl_fallback"

        results = []
        for sequence in harness.get("sequence_results", []):
            initial_state = sequence.get("initial_state")
            for subtask_result in sequence.get("subtask_results", []):
                task_name = str(subtask_result.get("subtask"))
                if task_name not in TARGETS:
                    continue
                robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                target_uid, target_link_index = _target_info(env, task_name)

                best = None
                all_pairs: list[dict[str, object]] = []
                gripper_prefix = []
                drawer_states = []
                led_states = []
                trace = subtask_result.get("rollout_trace", [])

                for step_idx, step in enumerate(trace):
                    candidate = _best_candidate(env, target_uid, target_link_index)
                    if candidate is not None:
                        candidate["step"] = step_idx
                        if best is None or float(candidate["distance"]) < float(best["distance"]):
                            best = candidate
                    action = np.asarray(step["action"], dtype=np.float32)
                    gripper_prefix.append(float(action[-1]))
                    _, _, _, info = env.step(action)
                    for pair in _contact_pairs_with_target(env, target_uid):
                        pair["step"] = step_idx
                        pair["target_link_name"] = _target_link_name(
                            target_uid=target_uid,
                            link_index=int(pair["target_link_index"]),
                            env=env,
                            task_name=task_name,
                        )
                        all_pairs.append(pair)
                    scene_info = info.get("scene_info", {})
                    if task_name == "open_drawer":
                        drawer_states.append(
                            float(scene_info.get("doors", {}).get("base__drawer", {}).get("current_state", 0.0))
                        )
                    elif task_name == "turn_on_led":
                        led_states.append(
                            int(scene_info.get("lights", {}).get("led", {}).get("logical_state", 0))
                        )

                unique_pairs = []
                seen = set()
                for pair in all_pairs:
                    key = (pair["robot_link_index"], pair["target_link_index"], pair["target_link_name"])
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_pairs.append(
                        {
                            "robot_link_index": pair["robot_link_index"],
                            "target_link_index": pair["target_link_index"],
                            "target_link_name": pair["target_link_name"],
                        }
                    )

                result = {
                    "task_name": task_name,
                    "language_annotation": subtask_result.get("language_annotation"),
                    "native_success": bool(subtask_result.get("rollout_success")),
                    "steps": len(trace),
                    "gripper_prefix": gripper_prefix[:8],
                    "best_surface_relation": best,
                    "unique_contact_link_pairs": unique_pairs,
                    "failure_family": _classify(task_name, best, all_pairs),
                }
                if drawer_states:
                    result["drawer_state_delta_final"] = float(drawer_states[-1] - drawer_states[0])
                    result["drawer_state_max"] = float(max(drawer_states))
                if led_states:
                    result["led_state_max"] = int(max(led_states))
                results.append(result)

        payload.update(
            {
                "status": "ok",
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
