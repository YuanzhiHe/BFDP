from __future__ import annotations

import torch
from torch import nn


class MockVLABackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        vocab_size: int,
        semantic_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim + proprio_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        lang = self.embedding(instruction)
        vis = self.obs_proj(torch.cat([obs, proprio], dim=-1))
        return self.encoder(torch.cat([lang, vis], dim=-1))

