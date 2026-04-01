from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class CALVINRolloutDataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]], obs_noise_scale: float = 0.0) -> None:
        super().__init__()
        self.samples = samples
        self.obs_noise_scale = obs_noise_scale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        obs = sample["obs"].clone()
        if self.obs_noise_scale > 0.0:
            obs = obs + torch.randn_like(obs) * self.obs_noise_scale
        return {
            "obs": obs,
            "proprio": sample["proprio"].clone(),
            "instruction": sample["instruction"].clone(),
            "task_id": sample["task_id"].clone(),
            "phase": sample["phase"].clone(),
            "target_action": sample["target_action"].clone(),
            "episode_id": sample["episode_id"].clone(),
            "step_index": sample["step_index"].clone(),
            "episode_length": sample["episode_length"].clone(),
            "gripper_change": sample["gripper_change"].clone(),
            "turn_on_led_flag": sample["turn_on_led_flag"].clone(),
            "turn_on_led_early_flag": sample["turn_on_led_early_flag"].clone(),
        }


def _stable_instruction_id(text: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % vocab_size


def _phase_bucket(step_index: int, total_steps: int, num_phases: int) -> int:
    if num_phases <= 1 or total_steps <= 1:
        return 0
    progress = step_index / max(1, total_steps - 1)
    return min(num_phases - 1, int(progress * num_phases))


def _is_turn_on_led_instruction(text: str) -> bool:
    normalized = text.strip().lower()
    return "turn on" in normalized and ("led" in normalized or "light" in normalized)


def _fit_vector(values: list[float], size: int) -> torch.Tensor:
    vector = torch.tensor(values[:size], dtype=torch.float32)
    if vector.shape[0] == size:
        return vector
    padded = torch.zeros(size, dtype=torch.float32)
    padded[: vector.shape[0]] = vector
    return padded


def _build_obs_vector(
    step: dict,
    obs_slice: tuple[int, int],
    scene_obs_slice: tuple[int, int] | None,
) -> torch.Tensor:
    robot_obs = step["robot_obs"]
    obs_values = list(robot_obs[obs_slice[0] : obs_slice[1]])
    if scene_obs_slice is not None:
        scene_obs = step.get("scene_obs", [])
        obs_values.extend(scene_obs[scene_obs_slice[0] : scene_obs_slice[1]])
    return torch.tensor(obs_values, dtype=torch.float32)


def _build_action_chunk(
    steps: list[dict],
    start_index: int,
    chunk_len: int,
    action_dim: int,
) -> torch.Tensor:
    actions = [_fit_vector(step["action"], action_dim) for step in steps]
    tail = actions[min(start_index, len(actions) - 1)]
    chunk = []
    for offset in range(chunk_len):
        idx = start_index + offset
        chunk.append(actions[idx] if idx < len(actions) else tail)
    return torch.stack(chunk, dim=0)


def _oversample_turn_on_led_heads(
    samples: list[dict[str, torch.Tensor]],
    factor: int,
) -> tuple[list[dict[str, torch.Tensor]], int]:
    if factor <= 1:
        return samples, 0
    duplicated = [
        sample
        for sample in samples
        if float(sample["turn_on_led_early_flag"].item()) > 0.5
        for _ in range(factor - 1)
    ]
    return samples + duplicated, len(duplicated)


def _flatten_episode_samples(
    episode: dict,
    episode_id: int,
    task_id: int,
    dataset_cfg: dict,
    obs_slice: tuple[int, int],
    scene_obs_slice: tuple[int, int] | None,
    proprio_slice: tuple[int, int],
    action_dim: int,
) -> list[dict[str, torch.Tensor]]:
    steps = episode["steps"]
    if not steps:
        return []
    instruction_text = episode.get("instruction", "calvin rollout")
    instruction_id = _stable_instruction_id(instruction_text, dataset_cfg["vocab_size"])
    turn_on_led_flag = 1.0 if _is_turn_on_led_instruction(instruction_text) else 0.0
    samples = []
    for step_index, step in enumerate(steps):
        robot_obs = step["robot_obs"]
        progress = step_index / max(1, len(steps) - 1)
        current_gripper = float(_fit_vector(step["action"], action_dim)[-1].item())
        if step_index > 0:
            previous_gripper = float(_fit_vector(steps[step_index - 1]["action"], action_dim)[-1].item())
            gripper_change = 1.0 if (current_gripper >= 0.0) != (previous_gripper >= 0.0) else 0.0
        else:
            gripper_change = 0.0
        turn_on_led_early_flag = 1.0 if turn_on_led_flag > 0.5 and progress <= 0.2 else 0.0
        samples.append(
            {
                "obs": _build_obs_vector(
                    step=step,
                    obs_slice=obs_slice,
                    scene_obs_slice=scene_obs_slice,
                ),
                "proprio": torch.tensor(
                    robot_obs[proprio_slice[0] : proprio_slice[1]],
                    dtype=torch.float32,
                ),
                "instruction": torch.tensor(instruction_id, dtype=torch.long),
                "task_id": torch.tensor(task_id, dtype=torch.long),
                "phase": torch.tensor(
                    _phase_bucket(step_index, len(steps), dataset_cfg["num_phases"]),
                    dtype=torch.long,
                ),
                "episode_id": torch.tensor(episode_id, dtype=torch.long),
                "step_index": torch.tensor(step_index, dtype=torch.long),
                "episode_length": torch.tensor(len(steps), dtype=torch.long),
                "gripper_change": torch.tensor(gripper_change, dtype=torch.float32),
                "turn_on_led_flag": torch.tensor(turn_on_led_flag, dtype=torch.float32),
                "turn_on_led_early_flag": torch.tensor(turn_on_led_early_flag, dtype=torch.float32),
                "target_action": _build_action_chunk(
                    steps=steps,
                    start_index=step_index,
                    chunk_len=dataset_cfg["chunk_len"],
                    action_dim=action_dim,
                ),
            }
        )
    return samples


def _load_rollout_payload(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_data_bundle(
    dataset_cfg: dict,
    training_cfg: dict,
    benchmark_cfg: dict,
    seed: int,
) -> dict:
    rollout_cfg = dataset_cfg["calvin_export"]
    rollout_path = Path(rollout_cfg["rollout_path"])
    payload = _load_rollout_payload(rollout_path)
    adapter_cfg = benchmark_cfg["adapter"]
    obs_slice = (
        adapter_cfg["obs_slice"]["start"],
        adapter_cfg["obs_slice"]["end"],
    )
    scene_obs_slice = None
    if "scene_obs_slice" in adapter_cfg:
        scene_obs_slice = (
            adapter_cfg["scene_obs_slice"]["start"],
            adapter_cfg["scene_obs_slice"]["end"],
        )
    proprio_slice = (
        adapter_cfg["proprio_slice"]["start"],
        adapter_cfg["proprio_slice"]["end"],
    )
    action_dim = adapter_cfg["action_dim"]
    task_names = sorted(
        {
            episode.get("task_name")
            for episode in payload["episodes"]
            if episode.get("task_name")
        }
    )
    task_name_to_id = {task_name: idx for idx, task_name in enumerate(task_names)}

    episode_samples = [
        _flatten_episode_samples(
            episode=episode,
            episode_id=episode_index,
            task_id=task_name_to_id.get(episode.get("task_name"), 0),
            dataset_cfg=dataset_cfg,
            obs_slice=obs_slice,
            scene_obs_slice=scene_obs_slice,
            proprio_slice=proprio_slice,
            action_dim=action_dim,
        )
        for episode_index, episode in enumerate(payload["episodes"])
    ]
    episode_indices = [idx for idx, samples in enumerate(episode_samples) if samples]
    if not episode_indices:
        raise ValueError(f"no usable CALVIN rollout steps found in {rollout_path}")

    rng = random.Random(seed)
    rng.shuffle(episode_indices)
    if len(episode_indices) == 1:
        train_episode_ids = episode_indices
        val_episode_ids = episode_indices
    else:
        requested_val = max(1, int(round(len(episode_indices) * rollout_cfg["val_ratio"])))
        val_count = min(len(episode_indices) - 1, requested_val)
        val_episode_ids = episode_indices[:val_count]
        train_episode_ids = episode_indices[val_count:]

    train_samples = [
        sample
        for episode_idx in train_episode_ids
        for sample in episode_samples[episode_idx]
    ]
    val_samples = [
        sample
        for episode_idx in val_episode_ids
        for sample in episode_samples[episode_idx]
    ]
    train_samples, oversampled_turn_on_led_head_samples = _oversample_turn_on_led_heads(
        train_samples,
        int(training_cfg.get("turn_on_led_head_oversample_factor", 1)),
    )

    batch_size = training_cfg["batch_size"]
    loaders = {
        "train": DataLoader(
            CALVINRolloutDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val_nominal": DataLoader(
            CALVINRolloutDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
        ),
        "val_perturbed": DataLoader(
            CALVINRolloutDataset(
                val_samples,
                obs_noise_scale=rollout_cfg["obs_perturb_scale"],
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    effective_dataset_cfg = {
        "obs_dim": (obs_slice[1] - obs_slice[0])
        + (0 if scene_obs_slice is None else scene_obs_slice[1] - scene_obs_slice[0]),
        "proprio_dim": proprio_slice[1] - proprio_slice[0],
        "vocab_size": dataset_cfg["vocab_size"],
        "num_phases": dataset_cfg["num_phases"],
        "num_tasks": max(1, len(task_name_to_id)),
        "task_name_to_id": task_name_to_id,
        "action_dim": action_dim,
        "chunk_len": dataset_cfg["chunk_len"],
    }
    summary = {
        "rollout_path": str(rollout_path),
        "dataset_path": payload["dataset_path"],
        "split": payload["split"],
        "lang_folder": payload["lang_folder"],
        "action_key": payload["action_key"],
        "horizon": payload["horizon"],
        "sequences": len(payload["episodes"]),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "oversampled_turn_on_led_head_samples": oversampled_turn_on_led_head_samples,
        "obs_dim": effective_dataset_cfg["obs_dim"],
        "proprio_dim": effective_dataset_cfg["proprio_dim"],
        "action_dim": action_dim,
        "chunk_len": effective_dataset_cfg["chunk_len"],
        "task_names": task_names,
        "task_name_to_id": task_name_to_id,
    }
    return {
        "source": "calvin_export",
        "dataset_cfg": effective_dataset_cfg,
        "loaders": loaders,
        "summary": summary,
    }
