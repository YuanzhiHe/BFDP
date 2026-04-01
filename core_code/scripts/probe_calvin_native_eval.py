from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.config import load_config
from svh_dp.data.calvin_rollout import _stable_instruction_id
from svh_dp.models.system import SVHDPModel
from svh_dp.utils.common import ensure_dir, set_seed, write_json


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

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from omegaconf import OmegaConf
import hydra
from calvin_agent.evaluation.evaluate_policy import make_env


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_obs_snapshot(adapter_cfg: dict, obs: dict) -> dict[str, object]:
    robot_obs = list(obs["robot_obs"])
    scene_obs = list(obs.get("scene_obs", []))
    obs_slice = adapter_cfg["obs_slice"]
    proprio_slice = adapter_cfg["proprio_slice"]
    scene_obs_slice = adapter_cfg.get("scene_obs_slice")

    selected_robot_obs = robot_obs[obs_slice["start"] : obs_slice["end"]]
    selected_scene_obs = []
    if scene_obs_slice is not None:
        selected_scene_obs = scene_obs[scene_obs_slice["start"] : scene_obs_slice["end"]]

    return {
        "robot_obs_dim": len(robot_obs),
        "scene_obs_dim": len(scene_obs),
        "selected_robot_obs": selected_robot_obs,
        "selected_scene_obs": selected_scene_obs,
        "selected_obs": selected_robot_obs + selected_scene_obs,
        "selected_proprio": robot_obs[proprio_slice["start"] : proprio_slice["end"]],
    }


def _selected_obs_from_episode_step(step: dict, adapter_cfg: dict) -> list[float]:
    robot_obs = list(step["robot_obs"])
    scene_obs = list(step.get("scene_obs", []))
    obs_slice = adapter_cfg["obs_slice"]
    scene_obs_slice = adapter_cfg.get("scene_obs_slice")
    selected_obs = list(robot_obs[obs_slice["start"] : obs_slice["end"]])
    if scene_obs_slice is not None:
        selected_obs.extend(scene_obs[scene_obs_slice["start"] : scene_obs_slice["end"]])
    return selected_obs


def _mean_squared_error(a: list[float], b: list[float]) -> float:
    count = min(len(a), len(b))
    if count == 0:
        return math.inf
    return sum((a[idx] - b[idx]) ** 2 for idx in range(count)) / count


def _load_teacher_episodes(path: str | None) -> list[dict[str, object]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("episodes", [])


def _load_probe_sequences(path: str | None) -> list[tuple[dict[str, object], list[str]]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sequences: list[tuple[dict[str, object], list[str]]] = []
    for sequence in payload.get("sequence_results", []):
        sequences.append((sequence["initial_state"], list(sequence["eval_sequence"])))
    return sequences


def _parse_sequence_indices(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    indices: list[int] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices or None


def _parse_slice_spec(spec: str) -> tuple[int, int]:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid slice spec {spec!r}; expected start:end")
    start = int(parts[0].strip())
    end = int(parts[1].strip())
    if start < 0 or end <= start:
        raise ValueError(f"invalid slice bounds {spec!r}; expected 0 <= start < end")
    return start, end


def _parse_task_slice_overrides(spec: str | None) -> dict[str, tuple[int, int]]:
    if not spec:
        return {}
    overrides: dict[str, tuple[int, int]] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        task_name, separator, slice_spec = item.partition("=")
        if not separator:
            raise ValueError(
                f"invalid task slice override {item!r}; expected task_name=start:end"
            )
        overrides[task_name.strip()] = _parse_slice_spec(slice_spec.strip())
    return overrides


def _resolve_symbolic_robot_override_slice(
    subtask: str,
    default_slice: tuple[int, int] | None,
    task_slices: dict[str, tuple[int, int]],
) -> tuple[int, int] | None:
    if subtask in task_slices:
        return task_slices[subtask]
    return default_slice


def _apply_symbolic_robot_override(
    symbolic_robot_obs: list[float] | np.ndarray,
    teacher_robot_obs: list[float] | np.ndarray,
    robot_slice: tuple[int, int] | None,
) -> tuple[np.ndarray, dict[str, object] | None]:
    symbolic_array = np.asarray(symbolic_robot_obs, dtype=np.float32).copy()
    teacher_array = np.asarray(teacher_robot_obs, dtype=np.float32)
    if robot_slice is None:
        return symbolic_array, None
    start, end = robot_slice
    if end > symbolic_array.shape[0] or end > teacher_array.shape[0]:
        raise ValueError(
            f"symbolic robot override slice {start}:{end} exceeds robot_obs dims "
            f"symbolic={symbolic_array.shape[0]} teacher={teacher_array.shape[0]}"
        )
    symbolic_before = symbolic_array[start:end].tolist()
    teacher_values = teacher_array[start:end].tolist()
    symbolic_array[start:end] = teacher_array[start:end]
    return symbolic_array, {
        "mode": "teacher_robot_prefix",
        "slice_start": start,
        "slice_end": end,
        "symbolic_values_before": symbolic_before,
        "teacher_values_applied": teacher_values,
    }


def _select_teacher_episode(
    teacher_episodes: list[dict[str, object]],
    subtask: str,
    initial_selected_obs: list[float],
    adapter_cfg: dict,
    preferred_sequence_index: int | None = None,
) -> dict[str, object] | None:
    matching = [
        episode for episode in teacher_episodes if episode.get("task_name") == subtask and episode.get("steps")
    ]
    if not matching:
        return None
    if preferred_sequence_index is not None:
        sequence_matching = [
            episode
            for episode in matching
            if episode.get("matched_sequence_index") == preferred_sequence_index
            or episode.get("source_sequence_index") == preferred_sequence_index
            or episode.get("sequence_index") == preferred_sequence_index
        ]
        if sequence_matching:
            matching = sequence_matching
    best_episode = None
    best_distance = math.inf
    for episode in matching:
        if initial_selected_obs:
            teacher_initial_selected_obs = _selected_obs_from_episode_step(episode["steps"][0], adapter_cfg)
            distance = _mean_squared_error(initial_selected_obs, teacher_initial_selected_obs)
        else:
            distance = 0.0
        if distance < best_distance:
            best_distance = distance
            best_episode = episode
    if best_episode is None:
        return None
    return {
        "episode": best_episode,
        "distance_mse": best_distance,
        "teacher_sequence_index": best_episode.get("sequence_index"),
        "teacher_matched_sequence_index": best_episode.get("matched_sequence_index"),
        "teacher_source_sequence_index": best_episode.get("source_sequence_index"),
        "teacher_instruction": best_episode.get("instruction"),
        "teacher_motion_pattern": [step["action"][:6] for step in best_episode.get("steps", [])],
        "teacher_gripper_pattern": [step["action"][-1] for step in best_episode.get("steps", [])],
    }


class NativeSVHDPWrapper:
    def __init__(self, config: dict, checkpoint_path: Path | None, variant: str) -> None:
        self.config = config
        self.adapter_cfg = config["benchmarks"]["calvin"]["adapter"]
        payload = None
        if checkpoint_path is not None:
            payload = torch.load(checkpoint_path, map_location="cpu")
        obs_dim = self.adapter_cfg["obs_slice"]["end"] - self.adapter_cfg["obs_slice"]["start"]
        scene_obs_slice = self.adapter_cfg.get("scene_obs_slice")
        if scene_obs_slice is not None:
            obs_dim += scene_obs_slice["end"] - scene_obs_slice["start"]
        checkpoint_dataset_cfg = {} if payload is None else dict(payload.get("dataset_cfg", {}))
        checkpoint_task_name_to_id = {}
        if payload is not None and payload.get("task_name_to_id"):
            checkpoint_task_name_to_id = {
                str(key): int(value) for key, value in payload["task_name_to_id"].items()
            }
        dataset_cfg = {
            "obs_dim": int(checkpoint_dataset_cfg.get("obs_dim", obs_dim)),
            "proprio_dim": int(checkpoint_dataset_cfg.get("proprio_dim", 6)),
            "vocab_size": int(checkpoint_dataset_cfg.get("vocab_size", config["dataset"]["vocab_size"])),
            "num_phases": int(checkpoint_dataset_cfg.get("num_phases", config["dataset"]["num_phases"])),
            "num_tasks": int(
                checkpoint_dataset_cfg.get(
                    "num_tasks",
                    max(1, len(checkpoint_task_name_to_id)),
                )
            ),
            "action_dim": int(checkpoint_dataset_cfg.get("action_dim", self.adapter_cfg["action_dim"])),
            "chunk_len": int(checkpoint_dataset_cfg.get("chunk_len", config["dataset"]["chunk_len"])),
        }
        model_cfg = dict(config["model"])
        if payload is not None and payload.get("model_cfg"):
            model_cfg.update(payload["model_cfg"])
        self.task_name_to_id = checkpoint_task_name_to_id
        self.model = SVHDPModel(dataset_cfg, model_cfg, variant=variant)
        if checkpoint_path is not None:
            self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.variant = variant
        self.step_counter = 0

    def reset(self) -> None:
        self.step_counter = 0

    def predict_action(self, obs: dict, goal: str, task_name: str | None = None) -> list[float]:
        robot_obs = obs["robot_obs"]
        scene_obs = obs.get("scene_obs", [])
        obs_slice = self.adapter_cfg["obs_slice"]
        scene_obs_slice = self.adapter_cfg.get("scene_obs_slice")
        proprio_slice = self.adapter_cfg["proprio_slice"]
        obs_values = list(robot_obs[obs_slice["start"] : obs_slice["end"]])
        if scene_obs_slice is not None:
            obs_values.extend(scene_obs[scene_obs_slice["start"] : scene_obs_slice["end"]])
        obs_tensor = torch.tensor(obs_values, dtype=torch.float32).unsqueeze(0)
        proprio_tensor = torch.tensor(
            robot_obs[proprio_slice["start"] : proprio_slice["end"]],
            dtype=torch.float32,
        ).unsqueeze(0)
        instruction_id = _stable_instruction_id(
            goal,
            self.config["dataset"]["vocab_size"],
        )
        phase_bucket = min(
            self.config["dataset"]["num_phases"] - 1,
            int((self.step_counter / max(1, 7)) * self.config["dataset"]["num_phases"]),
        )
        batch = {
            "obs": obs_tensor,
            "proprio": proprio_tensor,
            "instruction": torch.tensor([instruction_id], dtype=torch.long),
            "phase": torch.tensor([phase_bucket], dtype=torch.long),
            "step_index": torch.tensor([self.step_counter], dtype=torch.long),
        }
        if task_name is not None:
            batch["task_id"] = torch.tensor(
                [self.task_name_to_id.get(task_name, 0)],
                dtype=torch.long,
            )
        with torch.no_grad():
            outputs = self.model(batch)
        self.step_counter += 1
        action = outputs.action[0, 0].detach().cpu().tolist()
        native_cfg = self.config["benchmarks"]["calvin"].get("native_gripper_decode", {})
        if (
            outputs.gripper_logits is not None
            and native_cfg.get("enabled", False)
            and native_cfg.get("mode", "discrete_sign") == "discrete_sign"
        ):
            gripper_logit = float(outputs.gripper_logits[0, 0].detach().cpu().item())
            threshold = float(native_cfg.get("threshold", 0.0))
            action[-1] = 1.0 if gripper_logit >= threshold else -1.0
        return action


def _apply_gripper_override(
    raw_action: list[float],
    override_mode: str,
    step_idx: int,
    teacher_gripper_pattern: list[float] | None,
    teacher_prefix_steps: int,
) -> tuple[list[float], dict[str, object]]:
    action = list(raw_action)
    raw_gripper = float(raw_action[-1]) if raw_action else 0.0
    discretized_model_gripper = 1.0 if raw_gripper >= 0 else -1.0
    applied_gripper = discretized_model_gripper

    if override_mode == "force-open":
        applied_gripper = 1.0
    elif override_mode == "force-close":
        applied_gripper = -1.0
    elif override_mode == "teacher-pattern" and teacher_gripper_pattern:
        teacher_idx = min(step_idx, len(teacher_gripper_pattern) - 1)
        applied_gripper = float(teacher_gripper_pattern[teacher_idx])
    elif (
        override_mode == "teacher-prefix"
        and teacher_gripper_pattern
        and step_idx < teacher_prefix_steps
    ):
        teacher_idx = min(step_idx, len(teacher_gripper_pattern) - 1)
        applied_gripper = float(teacher_gripper_pattern[teacher_idx])

    if action:
        action[-1] = applied_gripper
    return action, {
        "raw_model_gripper": raw_gripper,
        "discretized_model_gripper": discretized_model_gripper,
        "applied_gripper": applied_gripper,
        "gripper_override_mode": override_mode,
    }


def _apply_motion_override(
    raw_action: list[float],
    override_mode: str,
    step_idx: int,
    teacher_motion_pattern: list[list[float]] | None,
    translation_scale: float,
    rotation_scale: float,
    teacher_prefix_steps: int,
) -> tuple[list[float], dict[str, object]]:
    action = list(raw_action)
    applied_motion = list(raw_action[:6])

    if override_mode == "teacher-pattern" and teacher_motion_pattern:
        teacher_idx = min(step_idx, len(teacher_motion_pattern) - 1)
        applied_motion = list(teacher_motion_pattern[teacher_idx][:6])
    elif (
        override_mode == "teacher-prefix"
        and teacher_motion_pattern
        and step_idx < teacher_prefix_steps
    ):
        teacher_idx = min(step_idx, len(teacher_motion_pattern) - 1)
        applied_motion = list(teacher_motion_pattern[teacher_idx][:6])
    elif override_mode == "zero":
        applied_motion = [0.0 for _ in applied_motion]
    else:
        for idx in range(min(3, len(applied_motion))):
            applied_motion[idx] *= translation_scale
        for idx in range(3, min(6, len(applied_motion))):
            applied_motion[idx] *= rotation_scale

    for idx, value in enumerate(applied_motion):
        action[idx] = value
    return action, {
        "raw_model_motion": list(raw_action[:6]),
        "applied_motion": applied_motion,
        "motion_override_mode": override_mode,
        "translation_scale": translation_scale,
        "rotation_scale": rotation_scale,
    }


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


def rollout_subtask(
    env,
    wrapper: NativeSVHDPWrapper,
    task_oracle,
    start_info: dict,
    subtask: str,
    lang_annotation: str,
    max_rollout_steps: int,
    motion_override_mode: str,
    gripper_override_mode: str,
    translation_scale: float,
    rotation_scale: float,
    teacher_match: dict[str, object] | None,
    teacher_prefix_steps: int,
    include_step_info: bool,
) -> tuple[list[dict[str, object]], set[str]]:
    rollout_trace: list[dict[str, object]] = []
    task_info: set[str] = set()
    next_obs = env.get_obs()
    next_info = start_info
    teacher_motion_pattern = None if teacher_match is None else teacher_match.get("teacher_motion_pattern")
    teacher_gripper_pattern = None if teacher_match is None else teacher_match.get("teacher_gripper_pattern")
    for step_idx in range(max_rollout_steps):
        current_obs_snapshot = _extract_obs_snapshot(wrapper.adapter_cfg, next_obs)
        raw_action = wrapper.predict_action(next_obs, lang_annotation, subtask)
        motion_action, motion_info = _apply_motion_override(
            raw_action=raw_action,
            override_mode=motion_override_mode,
            step_idx=step_idx,
            teacher_motion_pattern=teacher_motion_pattern,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            teacher_prefix_steps=teacher_prefix_steps,
        )
        action, gripper_info = _apply_gripper_override(
            raw_action=motion_action,
            override_mode=gripper_override_mode,
            step_idx=step_idx,
            teacher_gripper_pattern=teacher_gripper_pattern,
            teacher_prefix_steps=teacher_prefix_steps,
        )
        next_obs, _, _, next_info = env.step(action)
        next_obs_snapshot = _extract_obs_snapshot(wrapper.adapter_cfg, next_obs)
        task_info = task_oracle.get_task_info_for_set(start_info, next_info, {subtask})
        rollout_trace.append(
            {
                "step": step_idx,
                "raw_action": raw_action,
                "action": action,
                "motion_info": motion_info,
                "gripper_info": gripper_info,
                "task_info": sorted(task_info),
                "current_obs": current_obs_snapshot,
                "next_obs": next_obs_snapshot,
                "next_info": _json_safe(next_info) if include_step_info else None,
            }
        )
        if task_info:
            break
    return rollout_trace, task_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "Experiment" / "datasets" / "calvin_real" / "task_D_D"),
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=["diffusion_only", "vla_only", "modular", "full"],
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Optional checkpoint to load into the native-eval wrapper.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "Experiment" / "core_code" / "logs" / "calvin_native_eval_probe.json"),
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=16,
        help="How many native env steps to attempt for the first CALVIN subtask.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1,
        help="How many official CALVIN evaluation sequences to probe.",
    )
    parser.add_argument(
        "--max-subtasks",
        type=int,
        default=1,
        help="How many subtasks to evaluate from each selected sequence.",
    )
    parser.add_argument(
        "--subtask-reset-mode",
        default="per-subtask",
        choices=["per-subtask", "sequence-once"],
        help=(
            "Whether to reset before every evaluated subtask or initialize once at the "
            "sequence start and carry native state forward across subtasks."
        ),
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help=(
            "Keep evaluating later subtasks in the same selected sequence even if an "
            "earlier subtask does not satisfy the task oracle."
        ),
    )
    parser.add_argument(
        "--motion-override-mode",
        default="model",
        choices=["model", "teacher-pattern", "teacher-prefix", "zero"],
        help="How to set the first 6 action dimensions before native stepping.",
    )
    parser.add_argument(
        "--gripper-override-mode",
        default="model",
        choices=["model", "force-open", "force-close", "teacher-pattern", "teacher-prefix"],
        help="How to set the gripper channel before stepping the native CALVIN env.",
    )
    parser.add_argument(
        "--teacher-prefix-steps",
        type=int,
        default=0,
        help="Number of initial rollout steps that should use matched teacher motion/gripper when using teacher-prefix modes.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale applied to action dimensions 0:3 when motion override is model.",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Scale applied to action dimensions 3:6 when motion override is model.",
    )
    parser.add_argument(
        "--teacher-rollout-path",
        help="Optional rollout export used to source a nearest-neighbor teacher gripper pattern.",
    )
    parser.add_argument(
        "--initial-state-source",
        default="symbolic",
        choices=["symbolic", "teacher-exact"],
        help="How to initialize the native environment before each evaluated subtask.",
    )
    parser.add_argument(
        "--symbolic-robot-override-mode",
        default="none",
        choices=["none", "teacher-joint-prefix"],
        help=(
            "Optional symbolic-reset robot_obs override. "
            "'teacher-joint-prefix' keeps symbolic scene_obs but replaces a robot_obs slice "
            "with the matched teacher step-0 values."
        ),
    )
    parser.add_argument(
        "--symbolic-robot-override-default-slice",
        help="Default symbolic robot_obs slice to override, formatted start:end.",
    )
    parser.add_argument(
        "--symbolic-robot-override-task-slices",
        help=(
            "Optional comma-separated task-specific symbolic robot_obs slices, formatted "
            "task_name=start:end,task_name=start:end."
        ),
    )
    parser.add_argument(
        "--sequence-source-log",
        help="Optional existing native probe log used to reuse fixed initial states and eval sequences without calling CALVIN's multiprocessing sequence generator.",
    )
    parser.add_argument(
        "--sequence-indices",
        help=(
            "Optional comma-separated sequence indices to keep after loading probe "
            "sequences or generating official sequences."
        ),
    )
    parser.add_argument(
        "--include-step-info",
        action="store_true",
        help="Store a JSON-safe snapshot of native env info for each rollout step.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    set_seed(config["seed"])
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    symbolic_robot_override_default_slice = (
        None
        if not args.symbolic_robot_override_default_slice
        else _parse_slice_spec(args.symbolic_robot_override_default_slice)
    )
    symbolic_robot_override_task_slices = _parse_task_slice_overrides(
        args.symbolic_robot_override_task_slices
    )
    requested_sequence_indices = _parse_sequence_indices(args.sequence_indices)

    payload: dict[str, object] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "variant": args.variant,
        "checkpoint_path": args.checkpoint_path,
        "motion_override_mode": args.motion_override_mode,
        "gripper_override_mode": args.gripper_override_mode,
        "translation_scale": args.translation_scale,
        "rotation_scale": args.rotation_scale,
        "teacher_rollout_path": args.teacher_rollout_path,
        "initial_state_source": args.initial_state_source,
        "subtask_reset_mode": args.subtask_reset_mode,
        "continue_after_failure": args.continue_after_failure,
        "symbolic_robot_override_mode": args.symbolic_robot_override_mode,
        "symbolic_robot_override_default_slice": (
            None
            if symbolic_robot_override_default_slice is None
            else list(symbolic_robot_override_default_slice)
        ),
        "symbolic_robot_override_task_slices": {
            task_name: list(robot_slice)
            for task_name, robot_slice in symbolic_robot_override_task_slices.items()
        },
        "sequence_source_log": args.sequence_source_log,
        "sequence_indices": requested_sequence_indices,
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

        conf_dir = (
            PROJECT_ROOT
            / "Experiment"
            / "code_references"
            / "calvin_workspace"
            / "calvin"
            / "calvin_models"
            / "conf"
        )
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        payload["task_oracle_created"] = True

        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        wrapper = NativeSVHDPWrapper(
            config=config,
            checkpoint_path=None if not args.checkpoint_path else Path(args.checkpoint_path),
            variant=args.variant,
        )
        teacher_episodes = _load_teacher_episodes(args.teacher_rollout_path)
        selected_sequences = _load_probe_sequences(args.sequence_source_log)
        if selected_sequences:
            indexed_sequences = list(enumerate(selected_sequences))
            if requested_sequence_indices is not None:
                indexed_sequences = [
                    (idx, sequence)
                    for idx, sequence in indexed_sequences
                    if idx in requested_sequence_indices
                ]
            else:
                indexed_sequences = indexed_sequences[: args.num_sequences]
            payload["sequence_source_mode"] = "log_reuse"
        else:
            indexed_sequences = list(enumerate(get_sequences(args.num_sequences)))
            if requested_sequence_indices is not None:
                indexed_sequences = [
                    (idx, sequence)
                    for idx, sequence in indexed_sequences
                    if idx in requested_sequence_indices
                ]
            payload["sequence_source_mode"] = "calvin_generator"
        sequence_results: list[dict[str, object]] = []
        subtask_total_counter: Counter[str] = Counter()
        subtask_success_counter: Counter[str] = Counter()
        successful_subtasks_per_sequence: list[int] = []

        for sequence_result_index, (sequence_index, sequence_payload) in enumerate(indexed_sequences):
            initial_state, eval_sequence = sequence_payload
            evaluated_subtasks = list(eval_sequence[: args.max_subtasks])
            subtask_results: list[dict[str, object]] = []
            successful_subtasks = 0
            sequence_initialized = False
            sequence_initialization: dict[str, object] | None = None

            for subtask_index, subtask in enumerate(evaluated_subtasks):
                lang_annotation = val_annotations[subtask][0]
                wrapper.reset()
                teacher_match = None
                symbolic_initial_snapshot = None
                symbolic_robot_override = None
                needs_reset = (
                    args.subtask_reset_mode == "per-subtask" or not sequence_initialized
                )
                if needs_reset:
                    if args.initial_state_source == "symbolic":
                        if args.symbolic_robot_override_mode == "teacher-joint-prefix":
                            if not teacher_episodes:
                                raise ValueError(
                                    "--symbolic-robot-override-mode teacher-joint-prefix requires "
                                    "--teacher-rollout-path"
                                )
                            teacher_match = _select_teacher_episode(
                                teacher_episodes=teacher_episodes,
                                subtask=subtask,
                                initial_selected_obs=[],
                                adapter_cfg=wrapper.adapter_cfg,
                                preferred_sequence_index=sequence_index,
                            )
                            if teacher_match is None:
                                raise ValueError(
                                    "No teacher episode available for symbolic override "
                                    f"subtask={subtask} sequence_index={sequence_index}"
                                )
                        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
                        if args.symbolic_robot_override_mode == "teacher-joint-prefix":
                            teacher_initial_step = teacher_match["episode"]["steps"][0]
                            robot_obs, symbolic_robot_override = _apply_symbolic_robot_override(
                                symbolic_robot_obs=robot_obs,
                                teacher_robot_obs=teacher_initial_step["robot_obs"],
                                robot_slice=_resolve_symbolic_robot_override_slice(
                                    subtask=subtask,
                                    default_slice=symbolic_robot_override_default_slice,
                                    task_slices=symbolic_robot_override_task_slices,
                                ),
                            )
                        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                        initial_obs = env.get_obs()
                        initial_snapshot = _extract_obs_snapshot(wrapper.adapter_cfg, initial_obs)
                        symbolic_initial_snapshot = initial_snapshot
                        sequence_initialization = {
                            "mode": "symbolic",
                            "subtask": subtask,
                            "symbolic_robot_override": symbolic_robot_override,
                            "initial_selected_obs": initial_snapshot["selected_obs"],
                        }
                    else:
                        if not teacher_episodes:
                            raise ValueError(
                                "--initial-state-source teacher-exact requires --teacher-rollout-path"
                            )
                        teacher_match = _select_teacher_episode(
                            teacher_episodes=teacher_episodes,
                            subtask=subtask,
                            initial_selected_obs=[],
                            adapter_cfg=wrapper.adapter_cfg,
                            preferred_sequence_index=sequence_index,
                        )
                        if teacher_match is None:
                            raise ValueError(
                                f"No teacher episode available for subtask={subtask} sequence_index={sequence_index}"
                            )
                        teacher_steps = teacher_match["episode"]["steps"]
                        teacher_initial_step = teacher_steps[0]
                        env.reset(
                            robot_obs=np.asarray(teacher_initial_step["robot_obs"], dtype=np.float32),
                            scene_obs=np.asarray(
                                teacher_initial_step.get("scene_obs", []), dtype=np.float32
                            ),
                        )
                        initial_obs = env.get_obs()
                        initial_snapshot = _extract_obs_snapshot(wrapper.adapter_cfg, initial_obs)
                        sequence_initialization = {
                            "mode": "teacher-exact",
                            "subtask": subtask,
                            "teacher_sequence_index": teacher_match.get("teacher_sequence_index"),
                            "teacher_source_sequence_index": teacher_match.get(
                                "teacher_source_sequence_index"
                            ),
                            "initial_selected_obs": initial_snapshot["selected_obs"],
                        }
                    sequence_initialized = True
                else:
                    initial_obs = env.get_obs()
                    initial_snapshot = _extract_obs_snapshot(wrapper.adapter_cfg, initial_obs)
                    if args.initial_state_source == "symbolic":
                        symbolic_initial_snapshot = initial_snapshot
                start_info = env.get_info()
                if teacher_match is None and (
                    args.gripper_override_mode in {"teacher-pattern", "teacher-prefix"}
                    or args.motion_override_mode in {"teacher-pattern", "teacher-prefix"}
                ):
                    teacher_match = _select_teacher_episode(
                        teacher_episodes=teacher_episodes,
                        subtask=subtask,
                        initial_selected_obs=initial_snapshot["selected_obs"],
                        adapter_cfg=wrapper.adapter_cfg,
                        preferred_sequence_index=sequence_index,
                    )
                rollout_trace, task_info = rollout_subtask(
                    env=env,
                    wrapper=wrapper,
                    task_oracle=task_oracle,
                    start_info=start_info,
                    subtask=subtask,
                    lang_annotation=lang_annotation,
                    max_rollout_steps=args.max_rollout_steps,
                    motion_override_mode=args.motion_override_mode,
                    gripper_override_mode=args.gripper_override_mode,
                    translation_scale=args.translation_scale,
                    rotation_scale=args.rotation_scale,
                    teacher_match=teacher_match,
                    teacher_prefix_steps=args.teacher_prefix_steps,
                    include_step_info=args.include_step_info,
                )
                success = bool(task_info)
                subtask_total_counter[subtask] += 1
                if success:
                    subtask_success_counter[subtask] += 1
                    successful_subtasks += 1
                subtask_results.append(
                    {
                        "subtask_index": subtask_index,
                        "subtask": subtask,
                        "language_annotation": lang_annotation,
                        "initial_state_source": args.initial_state_source,
                        "motion_override_mode": args.motion_override_mode,
                        "gripper_override_mode": args.gripper_override_mode,
                        "rollout_steps_attempted": len(rollout_trace),
                        "rollout_success": success,
                        "teacher_gripper_match": teacher_match,
                        "symbolic_robot_override": symbolic_robot_override,
                        "symbolic_initial_selected_obs": (
                            symbolic_initial_snapshot["selected_obs"]
                            if symbolic_initial_snapshot is not None
                            else None
                        ),
                        "task_info_after_one_step": sorted(rollout_trace[0]["task_info"]) if rollout_trace else [],
                        "task_info_final": sorted(task_info),
                        "first_raw_action": rollout_trace[0]["raw_action"] if rollout_trace else [],
                        "first_action": rollout_trace[0]["action"] if rollout_trace else [],
                        "final_raw_action": rollout_trace[-1]["raw_action"] if rollout_trace else [],
                        "final_action": rollout_trace[-1]["action"] if rollout_trace else [],
                        "initial_selected_obs": (
                            rollout_trace[0]["current_obs"]["selected_obs"] if rollout_trace else []
                        ),
                        "start_info": _json_safe(start_info) if args.include_step_info else None,
                        "final_selected_obs": (
                            rollout_trace[-1]["next_obs"]["selected_obs"] if rollout_trace else []
                        ),
                        "next_robot_obs_dim": (
                            rollout_trace[-1]["next_obs"]["robot_obs_dim"] if rollout_trace else None
                        ),
                        "next_scene_obs_dim": (
                            rollout_trace[-1]["next_obs"]["scene_obs_dim"] if rollout_trace else None
                        ),
                        "rollout_trace": rollout_trace,
                    }
                )
                if not success:
                    if not args.continue_after_failure:
                        break

            successful_subtasks_per_sequence.append(successful_subtasks)
            sequence_results.append(
                {
                    "sequence_result_index": sequence_result_index,
                    "sequence_index": sequence_index,
                    "initial_state": initial_state,
                    "eval_sequence": list(eval_sequence),
                    "evaluated_subtasks": evaluated_subtasks,
                    "successful_subtasks": successful_subtasks,
                    "subtask_reset_mode": args.subtask_reset_mode,
                    "sequence_initialization": sequence_initialization,
                    "subtask_results": subtask_results,
                }
            )

        total_subtasks = sum(subtask_total_counter.values())
        total_successes = sum(subtask_success_counter.values())
        avg_successful_subtasks = (
            sum(successful_subtasks_per_sequence) / len(successful_subtasks_per_sequence)
            if successful_subtasks_per_sequence
            else 0.0
        )
        per_task_success = [
            {
                "subtask": subtask,
                "success": subtask_success_counter[subtask],
                "total": total,
                "success_rate": subtask_success_counter[subtask] / total if total else 0.0,
            }
            for subtask, total in sorted(subtask_total_counter.items())
        ]
        first_subtask_result = (
            sequence_results[0]["subtask_results"][0]
            if sequence_results and sequence_results[0]["subtask_results"]
            else {}
        )

        payload.update(
            {
                "status": "ok",
                "num_sequences": len(sequence_results),
                "max_subtasks": args.max_subtasks,
                "max_rollout_steps": args.max_rollout_steps,
                "total_subtasks_evaluated": total_subtasks,
                "total_subtask_successes": total_successes,
                "native_subtask_success_rate": (total_successes / total_subtasks) if total_subtasks else 0.0,
                "avg_successful_subtasks_per_sequence": avg_successful_subtasks,
                "per_task_success": per_task_success,
                "sequence_results": sequence_results,
                "initial_subtask": first_subtask_result.get("subtask"),
                "language_annotation": first_subtask_result.get("language_annotation"),
                "action_dim": len(first_subtask_result.get("first_action", [])),
                "first_action": first_subtask_result.get("first_action", []),
                "task_info_after_one_step": first_subtask_result.get("task_info_after_one_step", []),
                "task_info_final": first_subtask_result.get("task_info_final", []),
                "rollout_steps_attempted": first_subtask_result.get("rollout_steps_attempted", 0),
                "rollout_success": first_subtask_result.get("rollout_success", False),
                "teacher_gripper_match": first_subtask_result.get("teacher_gripper_match"),
                "first_raw_action": first_subtask_result.get("first_raw_action", []),
                "initial_selected_obs": first_subtask_result.get("initial_selected_obs", []),
                "final_selected_obs": first_subtask_result.get("final_selected_obs", []),
                "rollout_trace": first_subtask_result.get("rollout_trace", []),
                "next_robot_obs_dim": first_subtask_result.get("next_robot_obs_dim"),
                "next_scene_obs_dim": first_subtask_result.get("next_scene_obs_dim"),
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
    print(f"saved_probe={output_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
