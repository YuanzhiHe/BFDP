from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_scene_labels(scene_cfg_path: Path) -> list[dict[str, object]]:
    scene_cfg = OmegaConf.load(scene_cfg_path)
    scene_cfg_raw = OmegaConf.to_container(scene_cfg, resolve=False)
    object_cfg = scene_cfg_raw["objects"]

    labels: list[dict[str, object]] = []
    index = 0

    for fixed_name, fixed_cfg in object_cfg.get("fixed_objects", {}).items():
        for joint_name in fixed_cfg.get("joints", {}).keys():
            labels.append(
                {
                    "scene_index": index,
                    "group": "door",
                    "object_name": fixed_name,
                    "state_name": joint_name,
                    "component": "joint_state",
                }
            )
            index += 1
        for button_name in fixed_cfg.get("buttons", {}).keys():
            labels.append(
                {
                    "scene_index": index,
                    "group": "button",
                    "object_name": fixed_name,
                    "state_name": button_name,
                    "component": "joint_state",
                }
            )
            index += 1
        for switch_name in fixed_cfg.get("switches", {}).keys():
            labels.append(
                {
                    "scene_index": index,
                    "group": "switch",
                    "object_name": fixed_name,
                    "state_name": switch_name,
                    "component": "joint_state",
                }
            )
            index += 1
        for light_name in fixed_cfg.get("lights", {}).keys():
            labels.append(
                {
                    "scene_index": index,
                    "group": "light",
                    "object_name": fixed_name,
                    "state_name": light_name,
                    "component": "logical_state",
                }
            )
            index += 1

    euler_raw = scene_cfg_raw.get("euler_obs", True)
    euler_obs = True if isinstance(euler_raw, str) else bool(euler_raw)
    orientation_components = ["roll", "pitch", "yaw"] if euler_obs else ["qx", "qy", "qz", "qw"]
    pose_components = ["x", "y", "z"] + orientation_components

    for movable_name in object_cfg.get("movable_objects", {}).keys():
        for component in pose_components:
            labels.append(
                {
                    "scene_index": index,
                    "group": "movable_object",
                    "object_name": movable_name,
                    "state_name": "pose",
                    "component": component,
                }
            )
            index += 1

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-config",
        default=str(
            PROJECT_ROOT
            / "Experiment"
            / "code_references"
            / "calvin_workspace"
            / "calvin"
            / "calvin_env"
            / "conf"
            / "scene"
            / "calvin_scene_D_eval.yaml"
        ),
    )
    parser.add_argument(
        "--outlier-analysis",
        default="",
        help="Optional outlier analysis log to annotate with semantic labels.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = _build_scene_labels(Path(args.scene_config))
    label_by_index = {item["scene_index"]: item for item in labels}

    payload: dict[str, object] = {
        "scene_config": str(Path(args.scene_config).resolve()),
        "scene_obs_dim": len(labels),
        "scene_obs_labels": labels,
    }

    if args.outlier_analysis:
        outlier_payload = _load_json(args.outlier_analysis)
        payload["outlier_analysis"] = str(Path(args.outlier_analysis).resolve())
        payload["dominant_outlier_scene_dims_annotated"] = [
            {
                **item,
                "semantic_label": label_by_index.get(item["scene_index"]),
            }
            for item in outlier_payload.get("dominant_scene_dims", [])
        ]
        payload["wrap_like_scene_dims_annotated"] = [
            {
                **item,
                "semantic_label": label_by_index.get(item["scene_index"]),
            }
            for item in outlier_payload.get("wrap_like_scene_dims", [])
        ]

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
