from __future__ import annotations

import torch
from torch import nn


class DiffusionStyleExecutor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        cond_dim: int,
        hidden_dim: int,
        chunk_len: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + proprio_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_len * action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        output = self.net(torch.cat([obs, proprio, cond], dim=-1))
        return output.view(obs.shape[0], self.chunk_len, self.action_dim)


class DiffusionStyleGripperHead(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        cond_dim: int,
        hidden_dim: int,
        chunk_len: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim + proprio_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_len),
        )

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([obs, proprio, cond], dim=-1))


class TaskStructuredResidualDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        cond_dim: int,
        hidden_dim: int,
        chunk_len: int,
        action_dim: int,
        num_tasks: int,
        task_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.task_embedding = nn.Embedding(max(1, num_tasks), task_embedding_dim)
        input_dim = obs_dim + proprio_dim + cond_dim + task_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, chunk_len * action_dim)
        self.gripper_head = nn.Linear(hidden_dim, chunk_len)

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        cond: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_embedding = self.task_embedding(task_id)
        hidden = self.trunk(torch.cat([obs, proprio, cond, task_embedding], dim=-1))
        action_delta = self.action_head(hidden).view(obs.shape[0], self.chunk_len, self.action_dim)
        gripper_delta = self.gripper_head(hidden)
        return action_delta, gripper_delta


class PrefixConditionedJointDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        cond_dim: int,
        hidden_dim: int,
        chunk_len: int,
        action_dim: int,
        num_tasks: int,
        task_embedding_dim: int,
        prefix_steps: int,
        step_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.prefix_steps = max(1, prefix_steps)
        self.task_embedding = nn.Embedding(max(1, num_tasks), task_embedding_dim)
        self.step_embedding = nn.Embedding(self.prefix_steps, step_embedding_dim)
        input_dim = obs_dim + proprio_dim + cond_dim + task_embedding_dim + step_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, chunk_len * action_dim)
        self.gripper_head = nn.Linear(hidden_dim, chunk_len)

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        cond: torch.Tensor,
        task_id: torch.Tensor,
        step_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_embedding = self.task_embedding(task_id)
        clipped_step_index = torch.clamp(step_index, min=0, max=self.prefix_steps - 1)
        step_embedding = self.step_embedding(clipped_step_index)
        hidden = self.trunk(
            torch.cat([obs, proprio, cond, task_embedding, step_embedding], dim=-1)
        )
        action_delta = self.action_head(hidden).view(obs.shape[0], self.chunk_len, self.action_dim)
        gripper_delta = self.gripper_head(hidden)
        return action_delta, gripper_delta


class TurnOnLedTransitionGripperDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        cond_dim: int,
        hidden_dim: int,
        chunk_len: int,
        prefix_steps: int,
        step_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.prefix_steps = max(1, prefix_steps)
        self.step_embedding = nn.Embedding(self.prefix_steps, step_embedding_dim)
        input_dim = obs_dim + proprio_dim + cond_dim + step_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_len),
        )

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        cond: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        clipped_step_index = torch.clamp(step_index, min=0, max=self.prefix_steps - 1)
        step_embedding = self.step_embedding(clipped_step_index)
        return self.net(torch.cat([obs, proprio, cond, step_embedding], dim=-1))


class VLAOnlyActionHead(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        hidden_dim: int,
        chunk_len: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_len * action_dim),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        output = self.net(semantic)
        return output.view(semantic.shape[0], self.chunk_len, self.action_dim)


class VLAOnlyGripperHead(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        hidden_dim: int,
        chunk_len: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_len),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.net(semantic)
