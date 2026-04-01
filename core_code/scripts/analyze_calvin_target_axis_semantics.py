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
from omegaconf import OmegaConf


TARGETS = {
    "turn_on_led": ("base__button", 0, "button_link"),
    "open_drawer": ("base__drawer", 3, "drawer_link"),
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


def _target_info(env, task_name: str) -> tuple[int, int, str]:
    target_name, joint_index, link_name = TARGETS[task_name]
    if task_name == "turn_on_led":
        obj = next(button for button in env.scene.buttons if button.name == target_name)
    else:
        obj = next(door for door in env.scene.doors if door.name == target_name)
    return int(obj.uid), int(joint_index), link_name


def _joint_type_name(joint_type: int) -> str:
    mapping = {
        0: "revolute",
        1: "prismatic",
        2: "spherical",
        3: "planar",
        4: "fixed",
        5: "point2point",
        6: "gear",
    }
    return mapping.get(joint_type, f"unknown_{joint_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
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

        env.reset()
        results = {}
        for task_name in ["turn_on_led", "open_drawer"]:
            uid, joint_index, link_name = _target_info(env, task_name)
            joint_info = env.p.getJointInfo(uid, joint_index, physicsClientId=env.cid)
            link_state = env.p.getLinkState(uid, joint_index, physicsClientId=env.cid)
            link_pos = list(link_state[0])
            link_quat = list(link_state[1])
            rotation = _rotation_matrix_from_quat(link_quat)
            local_axes_in_world = {
                "local_axis_0": [float(value) for value in rotation[:, 0].tolist()],
                "local_axis_1": [float(value) for value in rotation[:, 1].tolist()],
                "local_axis_2": [float(value) for value in rotation[:, 2].tolist()],
            }
            joint_axis_parent = [float(value) for value in joint_info[13]]
            parent_frame_orn = [float(value) for value in joint_info[15]]
            parent_rotation = _rotation_matrix_from_quat(parent_frame_orn)
            joint_axis_world = [float(value) for value in (parent_rotation @ np.asarray(joint_axis_parent)).tolist()]
            alignment = {
                axis_name: float(np.dot(np.asarray(axis_world), np.asarray(joint_axis_world)))
                for axis_name, axis_world in local_axes_in_world.items()
            }
            results[task_name] = {
                "uid": int(uid),
                "joint_index": int(joint_index),
                "joint_name": joint_info[1].decode("utf-8"),
                "link_name": joint_info[12].decode("utf-8"),
                "joint_type": _joint_type_name(int(joint_info[2])),
                "joint_lower_limit": float(joint_info[8]),
                "joint_upper_limit": float(joint_info[9]),
                "joint_axis_parent": joint_axis_parent,
                "parent_frame_orientation": parent_frame_orn,
                "joint_axis_world_from_parent": joint_axis_world,
                "target_link_position_world": link_pos,
                "target_link_quaternion_world": link_quat,
                "local_axes_in_world": local_axes_in_world,
                "joint_axis_alignment_with_local_axes": alignment,
                "most_aligned_local_axis": max(alignment, key=lambda key: abs(alignment[key])),
            }

        payload.update({"status": "ok", "results": results})
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
