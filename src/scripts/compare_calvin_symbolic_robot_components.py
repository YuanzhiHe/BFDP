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
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
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

ROBOT_COMPONENTS = {
    "swap_xyz_0_3": slice(0, 3),
    "swap_rpy_3_6": slice(3, 6),
    "swap_hand_6_9": slice(6, 9),
    "swap_arm_joints_9_14": slice(9, 14),
    "swap_gripper_14_15": slice(14, 15),
    "swap_pose_0_6": slice(0, 6),
    "swap_front_0_9": slice(0, 9),
    "swap_gripper_width_6_7": slice(6, 7),
    "swap_joint0_7_8": slice(7, 8),
    "swap_joint1_8_9": slice(8, 9),
    "swap_joints01_7_9": slice(7, 9),
    "swap_hand_to_arm_6_14": slice(6, 14),
    "swap_hand_to_gripper_action_6_15": slice(6, 15),
    "swap_arm_and_gripper_action_9_15": slice(9, 15),
    "swap_arm_prefix_7_12": slice(7, 12),
    "swap_joints012_7_10": slice(7, 10),
    "swap_joints0123_7_11": slice(7, 11),
    "swap_joints123_8_11": slice(8, 11),
    "swap_joints1234_8_12": slice(8, 12),
    "swap_joints23_9_11": slice(9, 11),
    "swap_joints234_9_12": slice(9, 12),
    "swap_joints34_10_12": slice(10, 12),
    "swap_all_robot_0_15": slice(0, 15),
}

ROBOT_COMPONENT_LABELS = {
    "swap_xyz_0_3": ["tcp_pos_x", "tcp_pos_y", "tcp_pos_z"],
    "swap_rpy_3_6": ["tcp_orn_x", "tcp_orn_y", "tcp_orn_z"],
    "swap_hand_6_9": ["gripper_opening_width", "robot_joint_0", "robot_joint_1"],
    "swap_arm_joints_9_14": ["robot_joint_2", "robot_joint_3", "robot_joint_4", "robot_joint_5", "robot_joint_6"],
    "swap_gripper_14_15": ["gripper_action"],
    "swap_pose_0_6": [
        "tcp_pos_x",
        "tcp_pos_y",
        "tcp_pos_z",
        "tcp_orn_x",
        "tcp_orn_y",
        "tcp_orn_z",
    ],
    "swap_front_0_9": [
        "tcp_pos_x",
        "tcp_pos_y",
        "tcp_pos_z",
        "tcp_orn_x",
        "tcp_orn_y",
        "tcp_orn_z",
        "gripper_opening_width",
        "robot_joint_0",
        "robot_joint_1",
    ],
    "swap_gripper_width_6_7": ["gripper_opening_width"],
    "swap_joint0_7_8": ["robot_joint_0"],
    "swap_joint1_8_9": ["robot_joint_1"],
    "swap_joints01_7_9": ["robot_joint_0", "robot_joint_1"],
    "swap_hand_to_arm_6_14": [
        "gripper_opening_width",
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
        "robot_joint_5",
        "robot_joint_6",
    ],
    "swap_hand_to_gripper_action_6_15": [
        "gripper_opening_width",
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
        "robot_joint_5",
        "robot_joint_6",
        "gripper_action",
    ],
    "swap_arm_and_gripper_action_9_15": [
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
        "robot_joint_5",
        "robot_joint_6",
        "gripper_action",
    ],
    "swap_arm_prefix_7_12": [
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
    ],
    "swap_joints012_7_10": [
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
    ],
    "swap_joints0123_7_11": [
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
    ],
    "swap_joints123_8_11": [
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
    ],
    "swap_joints1234_8_12": [
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
    ],
    "swap_joints23_9_11": [
        "robot_joint_2",
        "robot_joint_3",
    ],
    "swap_joints234_9_12": [
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
    ],
    "swap_joints34_10_12": [
        "robot_joint_3",
        "robot_joint_4",
    ],
    "swap_all_robot_0_15": [
        "tcp_pos_x",
        "tcp_pos_y",
        "tcp_pos_z",
        "tcp_orn_x",
        "tcp_orn_y",
        "tcp_orn_z",
        "gripper_opening_width",
        "robot_joint_0",
        "robot_joint_1",
        "robot_joint_2",
        "robot_joint_3",
        "robot_joint_4",
        "robot_joint_5",
        "robot_joint_6",
        "gripper_action",
    ],
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


def _robot_link_position(env, link_index: int) -> list[float]:
    position = env.p.getLinkState(env.robot.robot_uid, link_index, physicsClientId=env.cid)[0]
    return list(position)


def _target_info(env, task_name: str) -> tuple[int, int, str]:
    target_name, link_index, link_name = TASK_TARGETS[task_name]
    if task_name == "turn_on_led":
        obj = next(button for button in env.scene.buttons if button.name == target_name)
    else:
        obj = next(door for door in env.scene.doors if door.name == target_name)
    return int(obj.uid), int(link_index), link_name


def _target_link_state(env, uid: int, link_index: int) -> tuple[list[float], list[float]]:
    position, orientation = env.p.getLinkState(uid, link_index, physicsClientId=env.cid)[:2]
    return list(position), list(orientation)


def _filtered_contacts(robot_contacts: list, target_uid: int, target_link_index: int) -> list[dict[str, int]]:
    filtered = []
    for contact in robot_contacts:
        if int(contact[2]) != target_uid:
            continue
        if int(contact[4]) != target_link_index:
            continue
        filtered.append(
            {
                "robot_link_index": int(contact[3]),
                "target_link_index": int(contact[4]),
            }
        )
    return filtered


def _step_geometry(env, task_name: str, step_idx: int, info: dict) -> dict[str, object]:
    target_uid, target_link_index, _ = _target_info(env, task_name)
    target_pos, target_quat = _target_link_state(env, target_uid, target_link_index)
    tcp_pos = list(info["robot_info"]["tcp_pos"])
    left_tip = _robot_link_position(env, ROBOT_LINKS["finger_left_tip"])
    right_tip = _robot_link_position(env, ROBOT_LINKS["finger_right_tip"])
    avg_tip = list(((np.asarray(left_tip, dtype=np.float64) + np.asarray(right_tip, dtype=np.float64)) / 2.0).tolist())
    contacts = _filtered_contacts(info["robot_info"]["contacts"], target_uid, target_link_index)

    def build(pos: list[float]) -> dict[str, object]:
        delta_world = [float(a - b) for a, b in zip(pos, target_pos)]
        delta_local = _world_to_local(delta_world, target_quat)
        return {
            "distance": _distance(pos, target_pos),
            "delta_local": delta_local,
        }

    return {
        "step": step_idx,
        "tcp": build(tcp_pos),
        "avg_fingertip": build(avg_tip),
        "target_contact_pairs": contacts,
    }


def _summarize(history: list[dict[str, object]]) -> dict[str, object]:
    initial = history[0]
    min_tcp = min(history, key=lambda row: row["tcp"]["distance"])
    min_tip = min(history, key=lambda row: row["avg_fingertip"]["distance"])
    return {
        "initial": {
            "tcp_distance": float(initial["tcp"]["distance"]),
            "avg_fingertip_distance": float(initial["avg_fingertip"]["distance"]),
            "tcp_delta_local": initial["tcp"]["delta_local"],
            "avg_fingertip_delta_local": initial["avg_fingertip"]["delta_local"],
        },
        "min_tcp": {
            "step": int(min_tcp["step"]),
            "distance": float(min_tcp["tcp"]["distance"]),
            "delta_local": min_tcp["tcp"]["delta_local"],
        },
        "min_avg_fingertip": {
            "step": int(min_tip["step"]),
            "distance": float(min_tip["avg_fingertip"]["distance"]),
            "delta_local": min_tip["avg_fingertip"]["delta_local"],
        },
        "first_target_contact_step": next((int(row["step"]) for row in history if row["target_contact_pairs"]), None),
    }


def _run_path(env, task_name: str, robot_obs: list[float], scene_obs: list[float], actions: list[list[float]]) -> dict[str, object]:
    env.reset(
        robot_obs=np.asarray(robot_obs, dtype=np.float32),
        scene_obs=np.asarray(scene_obs, dtype=np.float32),
    )
    history = []
    start_info = env.get_info()
    history.append(_step_geometry(env, task_name, -1, start_info))
    for step_idx, action in enumerate(actions):
        _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        history.append(_step_geometry(env, task_name, step_idx, info))
    return {"summary": _summarize(history)}


def _match_teacher_episode(episodes: list[dict], subtask: str, instruction: str) -> dict | None:
    for episode in episodes:
        if episode.get("task_name") == subtask and episode.get("instruction") == instruction:
            return episode
    for episode in episodes:
        if episode.get("task_name") == subtask:
            return episode
    return None


def _distance_gap(summary: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    return {
        "initial_tcp_distance_gap": float(summary["initial"]["tcp_distance"] - baseline["initial"]["tcp_distance"]),
        "initial_avg_fingertip_distance_gap": float(summary["initial"]["avg_fingertip_distance"] - baseline["initial"]["avg_fingertip_distance"]),
        "min_tcp_distance_gap": float(summary["min_tcp"]["distance"] - baseline["min_tcp"]["distance"]),
        "min_avg_fingertip_distance_gap": float(summary["min_avg_fingertip"]["distance"] - baseline["min_avg_fingertip"]["distance"]),
    }


def _swap_component(symbolic_robot_obs: list[float], exact_robot_obs: list[float], component: slice) -> list[float]:
    merged = list(symbolic_robot_obs)
    merged[component] = exact_robot_obs[component]
    return merged


def _component_value_report(
    symbolic_robot_obs: list[float], exact_robot_obs: list[float], component_name: str
) -> dict[str, object]:
    component = ROBOT_COMPONENTS[component_name]
    labels = ROBOT_COMPONENT_LABELS[component_name]
    symbolic_values = symbolic_robot_obs[component]
    exact_values = exact_robot_obs[component]
    return {
        "labels": labels,
        "symbolic_values": [float(value) for value in symbolic_values],
        "exact_values": [float(value) for value in exact_values],
        "delta": [float(exact - symbolic) for exact, symbolic in zip(exact_values, symbolic_values)],
    }


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
        except BaseException as env_exc:
            payload["env_error_type"] = type(env_exc).__name__
            payload["env_error_message"] = str(env_exc)
            env = build_env_without_egl(args.dataset_path)
            payload["env_mode"] = "hydra_no_egl_fallback"

        rollout_payload = _load_json(args.rollout_path)
        sequence_payload = _load_json(args.sequence_log)
        teacher_episodes = rollout_payload.get("episodes", [])
        results = []

        for sequence in sequence_payload.get("sequence_results", []):
            if not sequence.get("subtask_results"):
                continue
            subtask = sequence["subtask_results"][0]
            task_name = str(subtask["subtask"])
            instruction = str(subtask["language_annotation"])
            teacher_episode = _match_teacher_episode(teacher_episodes, task_name, instruction)
            if teacher_episode is None or not teacher_episode.get("steps"):
                continue

            steps = teacher_episode["steps"]
            actions = [step["action"] for step in steps]
            exact_robot_obs = list(steps[0]["robot_obs"])
            exact_scene_obs = list(steps[0].get("scene_obs", []))
            symbolic_robot_obs, _ = get_env_state_for_initial_condition(sequence["initial_state"])
            symbolic_robot_obs = list(symbolic_robot_obs)

            variants = {
                "exact_exact": _run_path(env, task_name, exact_robot_obs, exact_scene_obs, actions),
                "symbolic_robot_exact_scene": _run_path(env, task_name, symbolic_robot_obs, exact_scene_obs, actions),
            }
            for name, component in ROBOT_COMPONENTS.items():
                robot_obs = _swap_component(symbolic_robot_obs, exact_robot_obs, component)
                variants[name] = _run_path(env, task_name, robot_obs, exact_scene_obs, actions)

            baseline = variants["exact_exact"]["summary"]
            results.append(
                {
                    "task_name": task_name,
                    "language_annotation": instruction,
                    "native_sequence_index": sequence.get("sequence_index"),
                    "teacher_sequence_index": teacher_episode.get("sequence_index"),
                    "teacher_source_sequence_index": teacher_episode.get("source_sequence_index"),
                    "initial_state": sequence.get("initial_state"),
                    "exact_robot_obs": exact_robot_obs,
                    "symbolic_robot_obs": symbolic_robot_obs,
                    "robot_obs_euler_semantics": {
                        "0_3": ["tcp_pos_x", "tcp_pos_y", "tcp_pos_z"],
                        "3_6": ["tcp_orn_x", "tcp_orn_y", "tcp_orn_z"],
                        "6_7": ["gripper_opening_width"],
                        "7_14": [
                            "robot_joint_0",
                            "robot_joint_1",
                            "robot_joint_2",
                            "robot_joint_3",
                            "robot_joint_4",
                            "robot_joint_5",
                            "robot_joint_6",
                        ],
                        "14_15": ["gripper_action"],
                        "6_9": ["gripper_opening_width", "robot_joint_0", "robot_joint_1"],
                    },
                    "component_value_report": {
                        key: _component_value_report(symbolic_robot_obs, exact_robot_obs, key)
                        for key in ROBOT_COMPONENTS
                    },
                    "variants": variants,
                    "distance_gaps_vs_exact_exact": {
                        key: _distance_gap(value["summary"], baseline)
                        for key, value in variants.items()
                        if key != "exact_exact"
                    },
                }
            )

        payload.update(
            {
                "status": "ok",
                "comparison_count": len(results),
                "robot_components": {key: [value.start, value.stop] for key, value in ROBOT_COMPONENTS.items()},
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
