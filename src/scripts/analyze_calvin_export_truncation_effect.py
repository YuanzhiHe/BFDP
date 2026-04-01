from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


FOCUS_TASKS = {"turn_on_led", "open_drawer", "push_blue_block_right"}


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


def _lookup_naming_pattern(dataset_dir: Path, suffix: str = ".npz") -> tuple[str, int]:
    for path in sorted(dataset_dir.glob(f"*{suffix}")):
        match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
        if match:
            return path.name[: match.start()], len(match.group(1))
    raise FileNotFoundError(f"could not find any {suffix} files in {dataset_dir}")


def _load_scene_obs(dataset_dir: Path, prefix: str, n_digits: int, frame_index: int) -> np.ndarray:
    path = dataset_dir / f"{prefix}{frame_index:0{n_digits}d}.npz"
    with np.load(path, allow_pickle=True) as payload:
        return np.asarray(payload["scene_obs"], dtype=np.float64)


def _top_changes(delta: np.ndarray, labels: list[dict[str, object]], top_k: int = 5) -> list[dict[str, object]]:
    ordering = np.argsort(np.abs(delta))[::-1][:top_k]
    rows = []
    for idx in ordering.tolist():
        rows.append(
            {
                "scene_index": int(idx),
                "abs_delta": float(abs(delta[idx])),
                "signed_delta": float(delta[idx]),
                "semantic_label": labels[idx] if idx < len(labels) else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-path", required=True)
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload: dict[str, object] = {
        "rollout_path": str(Path(args.rollout_path).resolve()),
        "scene_config": str(Path(args.scene_config).resolve()),
        "status": "started",
    }

    try:
        rollout = _load_json(args.rollout_path)
        labels = _build_scene_labels(Path(args.scene_config))
        dataset_dir = Path(rollout["dataset_path"]) / rollout["split"]
        prefix, n_digits = _lookup_naming_pattern(dataset_dir)

        results = []
        per_task: dict[str, list[dict[str, object]]] = {}

        for episode in rollout.get("episodes", []):
            task_name = str(episode.get("task_name", "unknown"))
            if task_name not in FOCUS_TASKS:
                continue
            start_idx = int(episode["start_frame_index"])
            effective_start_idx = int(episode.get("effective_start_frame_index", start_idx))
            effective_end_idx = int(episode["effective_end_frame_index"])
            full_end_idx = int(episode["end_frame_index"])
            export_mode = str(episode.get("export_mode", rollout.get("export_mode", "prefix")))

            start_scene = _load_scene_obs(dataset_dir, prefix, n_digits, start_idx)
            effective_start_scene = _load_scene_obs(dataset_dir, prefix, n_digits, effective_start_idx)
            effective_end_scene = _load_scene_obs(dataset_dir, prefix, n_digits, effective_end_idx)
            full_end_scene = _load_scene_obs(dataset_dir, prefix, n_digits, full_end_idx)

            exported_delta = effective_end_scene - effective_start_scene
            full_delta = full_end_scene - start_scene
            if export_mode == "tail":
                omitted_delta = effective_start_scene - start_scene
            else:
                omitted_delta = full_end_scene - effective_end_scene
            truncated = effective_start_idx > start_idx or effective_end_idx < full_end_idx

            result = {
                "sequence_index": int(episode["sequence_index"]),
                "task_name": task_name,
                "instruction": episode.get("instruction"),
                "export_mode": export_mode,
                "full_len": int(full_end_idx - start_idx + 1),
                "exported_len": int(effective_end_idx - effective_start_idx + 1),
                "truncation_gap": int((effective_start_idx - start_idx) + (full_end_idx - effective_end_idx)),
                "is_truncated": truncated,
                "effective_start_frame_index": effective_start_idx,
                "effective_end_frame_index": effective_end_idx,
                "top_exported_window_changes": _top_changes(exported_delta, labels),
                "top_full_changes": _top_changes(full_delta, labels),
                "top_omitted_changes": _top_changes(omitted_delta, labels),
            }
            results.append(result)
            per_task.setdefault(task_name, []).append(result)

        task_aggregates = {}
        for task_name, task_results in sorted(per_task.items()):
            task_aggregates[task_name] = {
                "episodes": len(task_results),
                "all_truncated": all(int(item["truncation_gap"]) > 0 for item in task_results),
                "mean_full_len": float(sum(item["full_len"] for item in task_results) / len(task_results)),
                "mean_exported_len": float(sum(item["exported_len"] for item in task_results) / len(task_results)),
                "mean_truncation_gap": float(sum(item["truncation_gap"] for item in task_results) / len(task_results)),
                "dominant_exported_window_scene_index_counts": {},
                "dominant_full_scene_index_counts": {},
                "dominant_omitted_scene_index_counts": {},
            }
            for key_name, field_name in [
                ("dominant_exported_window_scene_index_counts", "top_exported_window_changes"),
                ("dominant_full_scene_index_counts", "top_full_changes"),
                ("dominant_omitted_scene_index_counts", "top_omitted_changes"),
            ]:
                counts: dict[str, int] = {}
                for item in task_results:
                    dominant = item[field_name][0]
                    semantic = dominant["semantic_label"]
                    label = (
                        f"{semantic['group']}:{semantic['object_name']}:{semantic['component']}"
                        if semantic
                        else str(dominant["scene_index"])
                    )
                    counts[label] = counts.get(label, 0) + 1
                task_aggregates[task_name][key_name] = counts

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
