from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticSpec:
    obs_dim: int
    proprio_dim: int
    vocab_size: int
    num_phases: int
    action_dim: int
    chunk_len: int
    base_noise_scale: float
    perturb_noise_scale: float


class SyntheticManipulationDataset(Dataset):
    def __init__(
        self,
        size: int,
        spec: SyntheticSpec,
        seed: int,
        mapping_seed: int,
        regime: str,
        curriculum_bias: float,
        randomization_scale: float,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.obs = torch.randn(size, spec.obs_dim, generator=generator)
        self.proprio = torch.randn(size, spec.proprio_dim, generator=generator)
        self.instruction = torch.randint(
            0, spec.vocab_size, (size,), generator=generator
        )
        self.phase = torch.randint(0, spec.num_phases, (size,), generator=generator)

        phase_scale = self.phase.float().unsqueeze(1) / max(1, spec.num_phases - 1)
        instruction_scale = self.instruction.float().unsqueeze(1) / max(1, spec.vocab_size)

        weight_generator = torch.Generator().manual_seed(mapping_seed)
        obs_weight = torch.randn(
            spec.obs_dim, spec.action_dim * spec.chunk_len, generator=weight_generator
        ) * 0.35
        prop_weight = torch.randn(
            spec.proprio_dim, spec.action_dim * spec.chunk_len, generator=weight_generator
        ) * 0.25

        base_target = self.obs @ obs_weight + self.proprio @ prop_weight
        phase_target = phase_scale * (1.0 + curriculum_bias) * 0.25
        instr_target = instruction_scale * 0.15
        target = base_target + phase_target + instr_target
        target = target.view(size, spec.chunk_len, spec.action_dim)

        noise_scale = spec.base_noise_scale * randomization_scale
        if regime == "perturbed":
            noise_scale = spec.perturb_noise_scale * randomization_scale

        noise = torch.randn(target.shape, generator=generator) * noise_scale
        self.target = target + noise
        self.regime = regime

    def __len__(self) -> int:
        return self.target.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs[index],
            "proprio": self.proprio[index],
            "instruction": self.instruction[index],
            "phase": self.phase[index],
            "target_action": self.target[index],
        }


def build_loaders(
    dataset_cfg: dict,
    training_cfg: dict,
    seed: int,
) -> dict[str, DataLoader]:
    spec = SyntheticSpec(
        obs_dim=dataset_cfg["obs_dim"],
        proprio_dim=dataset_cfg["proprio_dim"],
        vocab_size=dataset_cfg["vocab_size"],
        num_phases=dataset_cfg["num_phases"],
        action_dim=dataset_cfg["action_dim"],
        chunk_len=dataset_cfg["chunk_len"],
        base_noise_scale=dataset_cfg["base_noise_scale"],
        perturb_noise_scale=dataset_cfg["perturb_noise_scale"],
    )
    train_ds = SyntheticManipulationDataset(
        size=dataset_cfg["train_size"],
        spec=spec,
        seed=seed,
        mapping_seed=seed + 97,
        regime="nominal",
        curriculum_bias=training_cfg["curriculum_bias"],
        randomization_scale=training_cfg["randomization_scale"],
    )
    val_nominal = SyntheticManipulationDataset(
        size=dataset_cfg["val_size"],
        spec=spec,
        seed=seed + 1,
        mapping_seed=seed + 97,
        regime="nominal",
        curriculum_bias=0.0,
        randomization_scale=1.0,
    )
    val_perturbed = SyntheticManipulationDataset(
        size=dataset_cfg["val_size"],
        spec=spec,
        seed=seed + 2,
        mapping_seed=seed + 97,
        regime="perturbed",
        curriculum_bias=0.0,
        randomization_scale=1.0,
    )
    batch_size = training_cfg["batch_size"]
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_nominal": DataLoader(val_nominal, batch_size=batch_size, shuffle=False),
        "val_perturbed": DataLoader(val_perturbed, batch_size=batch_size, shuffle=False),
    }
