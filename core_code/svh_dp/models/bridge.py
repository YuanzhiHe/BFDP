from __future__ import annotations

import torch
from torch import nn


class BridgeAdapter(nn.Module):
    def __init__(self, semantic_dim: int, cond_dim: int, num_phases: int) -> None:
        super().__init__()
        self.phase_embedding = nn.Embedding(num_phases, cond_dim)
        self.project = nn.Sequential(
            nn.Linear(semantic_dim + cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, semantic: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_embed = self.phase_embedding(phase)
        return self.project(torch.cat([semantic, phase_embed], dim=-1))

