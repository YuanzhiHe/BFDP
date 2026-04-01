from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class RolloutSlices:
    obs_start: int
    obs_end: int
    proprio_start: int
    proprio_end: int


class RLBenchRolloutDataset(Dataset):
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
            "phase": sample["phase"].clone(),
            "target_action": sample["target_action"].clone(),
            "episode_id": sample["episode_id"].clone(),
            "step_index": sample["step_index"].clone(),
            "episode_length": sample["episode_length"].clone(),
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


def _fit_vector(values: list[float], size: int) -> torch.Tensor:
    vector = torch.tensor(values[:size], dtype=torch.float32)
    if vector.shape[0] == size:
        return vector
    padded = torch.zeros(size, dtype=torch.float32)
    padded[: vector.shape[0]] = vector
    return padded


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


def _flatten_episode_samples(
    episode: dict,
    episode_id: int,
    dataset_cfg: dict,
    slices: RolloutSlices,
    action_dim: int,
) -> list[dict[str, torch.Tensor]]:
    steps = episode["steps"]
    if not steps:
        return []
    instruction_text = (
        episode["descriptions"][0]
        if episode.get("descriptions")
        else dataset_cfg["rlbench_export"].get("default_instruction", "rlbench rollout")
    )
    instruction_id = _stable_instruction_id(instruction_text, dataset_cfg["vocab_size"])
    samples = []
    for step_index, step in enumerate(steps):
        low_dim = step["low_dim"]
        obs = low_dim[slices.obs_start : slices.obs_end]
        proprio = low_dim[slices.proprio_start : slices.proprio_end]
        samples.append(
            {
                "obs": torch.tensor(obs, dtype=torch.float32),
                "proprio": torch.tensor(proprio, dtype=torch.float32),
                "instruction": torch.tensor(instruction_id, dtype=torch.long),
                "phase": torch.tensor(
                    _phase_bucket(step_index, len(steps), dataset_cfg["num_phases"]),
                    dtype=torch.long,
                ),
                "episode_id": torch.tensor(episode_id, dtype=torch.long),
                "step_index": torch.tensor(step_index, dtype=torch.long),
                "episode_length": torch.tensor(len(steps), dtype=torch.long),
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


def _collect_task_names(payload: dict) -> list[str]:
    names = []
    for episode in payload["episodes"]:
        task_name = episode.get("task_name", payload["task_name"])
        if task_name not in names:
            names.append(task_name)
    if names:
        return names
    task_name = payload["task_name"]
    if isinstance(task_name, list):
        return task_name
    return [task_name]


def build_data_bundle(
    dataset_cfg: dict,
    training_cfg: dict,
    benchmark_cfg: dict,
    seed: int,
) -> dict:
    rollout_cfg = dataset_cfg["rlbench_export"]
    rollout_path = Path(rollout_cfg["rollout_path"])
    payload = _load_rollout_payload(rollout_path)
    adapter_cfg = benchmark_cfg["adapter"]
    slices = RolloutSlices(
        obs_start=adapter_cfg["obs_slice"]["start"],
        obs_end=adapter_cfg["obs_slice"]["end"],
        proprio_start=adapter_cfg["proprio_slice"]["start"],
        proprio_end=adapter_cfg["proprio_slice"]["end"],
    )
    action_dim = adapter_cfg["action_dim"]

    episode_samples = [
        _flatten_episode_samples(
            episode=episode,
            episode_id=episode_index,
            dataset_cfg=dataset_cfg,
            slices=slices,
            action_dim=action_dim,
        )
        for episode_index, episode in enumerate(payload["episodes"])
    ]
    episode_indices = [idx for idx, samples in enumerate(episode_samples) if samples]
    if not episode_indices:
        raise ValueError(f"no usable rollout steps found in {rollout_path}")

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

    batch_size = training_cfg["batch_size"]
    loaders = {
        "train": DataLoader(
            RLBenchRolloutDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val_nominal": DataLoader(
            RLBenchRolloutDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
        ),
        "val_perturbed": DataLoader(
            RLBenchRolloutDataset(
                val_samples,
                obs_noise_scale=rollout_cfg["obs_perturb_scale"],
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    effective_dataset_cfg = {
        "obs_dim": slices.obs_end - slices.obs_start,
        "proprio_dim": slices.proprio_end - slices.proprio_start,
        "vocab_size": dataset_cfg["vocab_size"],
        "num_phases": dataset_cfg["num_phases"],
        "action_dim": action_dim,
        "chunk_len": dataset_cfg["chunk_len"],
    }
    summary = {
        "rollout_path": str(rollout_path),
        "task_name": payload["task_name"],
        "task_names": _collect_task_names(payload),
        "policy": payload["policy"],
        "episodes": len(payload["episodes"]),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "obs_dim": effective_dataset_cfg["obs_dim"],
        "proprio_dim": effective_dataset_cfg["proprio_dim"],
        "action_dim": action_dim,
        "chunk_len": effective_dataset_cfg["chunk_len"],
    }
    return {
        "source": "rlbench_export",
        "dataset_cfg": effective_dataset_cfg,
        "loaders": loaders,
        "summary": summary,
    }
