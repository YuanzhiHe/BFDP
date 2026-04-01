from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from svh_dp.models.bridge import BridgeAdapter
from svh_dp.models.executor import (
    DiffusionStyleExecutor,
    DiffusionStyleGripperHead,
    PrefixConditionedJointDecoder,
    TaskStructuredResidualDecoder,
    TurnOnLedTransitionGripperDecoder,
    VLAOnlyActionHead,
    VLAOnlyGripperHead,
)
from svh_dp.models.vla import MockVLABackbone


@dataclass
class SystemOutputs:
    action: torch.Tensor
    gripper_logits: torch.Tensor | None
    semantic: torch.Tensor | None
    cond: torch.Tensor | None


class SVHDPModel(nn.Module):
    def __init__(self, dataset_cfg: dict, model_cfg: dict, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.vla = MockVLABackbone(
            obs_dim=dataset_cfg["obs_dim"],
            proprio_dim=dataset_cfg["proprio_dim"],
            vocab_size=dataset_cfg["vocab_size"],
            semantic_dim=model_cfg["semantic_dim"],
            hidden_dim=model_cfg["hidden_dim"],
        )
        self.bridge = BridgeAdapter(
            semantic_dim=model_cfg["semantic_dim"],
            cond_dim=model_cfg["cond_dim"],
            num_phases=dataset_cfg["num_phases"],
        )
        self.executor = DiffusionStyleExecutor(
            obs_dim=dataset_cfg["obs_dim"],
            proprio_dim=dataset_cfg["proprio_dim"],
            cond_dim=model_cfg["cond_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            chunk_len=dataset_cfg["chunk_len"],
            action_dim=dataset_cfg["action_dim"],
        )
        self.executor_gripper = DiffusionStyleGripperHead(
            obs_dim=dataset_cfg["obs_dim"],
            proprio_dim=dataset_cfg["proprio_dim"],
            cond_dim=model_cfg["cond_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            chunk_len=dataset_cfg["chunk_len"],
        )
        task_decoder_cfg = model_cfg.get("task_structured_decoder", {})
        self.task_structured_decoder_enabled = bool(task_decoder_cfg.get("enabled", False))
        self.task_structured_decoder_scale = float(task_decoder_cfg.get("residual_scale", 1.0))
        self.task_structured_decoder = None
        if self.task_structured_decoder_enabled:
            self.task_structured_decoder = TaskStructuredResidualDecoder(
                obs_dim=dataset_cfg["obs_dim"],
                proprio_dim=dataset_cfg["proprio_dim"],
                cond_dim=model_cfg["cond_dim"],
                hidden_dim=int(task_decoder_cfg.get("hidden_dim", model_cfg["hidden_dim"])),
                chunk_len=dataset_cfg["chunk_len"],
                action_dim=dataset_cfg["action_dim"],
                num_tasks=int(dataset_cfg.get("num_tasks", 1)),
                task_embedding_dim=int(task_decoder_cfg.get("task_embedding_dim", 8)),
            )
        prefix_decoder_cfg = model_cfg.get("prefix_decoder", {})
        self.prefix_decoder_enabled = bool(prefix_decoder_cfg.get("enabled", False))
        self.prefix_decoder_scale = float(prefix_decoder_cfg.get("residual_scale", 1.0))
        self.prefix_decoder_prefix_steps = int(prefix_decoder_cfg.get("prefix_steps", 5))
        self.prefix_decoder = None
        if self.prefix_decoder_enabled:
            self.prefix_decoder = PrefixConditionedJointDecoder(
                obs_dim=dataset_cfg["obs_dim"],
                proprio_dim=dataset_cfg["proprio_dim"],
                cond_dim=model_cfg["cond_dim"],
                hidden_dim=int(prefix_decoder_cfg.get("hidden_dim", model_cfg["hidden_dim"])),
                chunk_len=dataset_cfg["chunk_len"],
                action_dim=dataset_cfg["action_dim"],
                num_tasks=int(dataset_cfg.get("num_tasks", 1)),
                task_embedding_dim=int(prefix_decoder_cfg.get("task_embedding_dim", 8)),
                prefix_steps=self.prefix_decoder_prefix_steps,
                step_embedding_dim=int(prefix_decoder_cfg.get("step_embedding_dim", 8)),
            )
        led_transition_decoder_cfg = model_cfg.get("turn_on_led_transition_decoder", {})
        self.turn_on_led_transition_decoder_enabled = bool(
            led_transition_decoder_cfg.get("enabled", False)
        )
        self.turn_on_led_transition_decoder_scale = float(
            led_transition_decoder_cfg.get("residual_scale", 1.0)
        )
        self.turn_on_led_transition_decoder_prefix_steps = int(
            led_transition_decoder_cfg.get("prefix_steps", 8)
        )
        self.turn_on_led_transition_task_id = None
        if isinstance(dataset_cfg.get("task_name_to_id"), dict):
            self.turn_on_led_transition_task_id = dataset_cfg["task_name_to_id"].get("turn_on_led")
        self.turn_on_led_transition_decoder = None
        if self.turn_on_led_transition_decoder_enabled:
            self.turn_on_led_transition_decoder = TurnOnLedTransitionGripperDecoder(
                obs_dim=dataset_cfg["obs_dim"],
                proprio_dim=dataset_cfg["proprio_dim"],
                cond_dim=model_cfg["cond_dim"],
                hidden_dim=int(led_transition_decoder_cfg.get("hidden_dim", model_cfg["hidden_dim"])),
                chunk_len=dataset_cfg["chunk_len"],
                prefix_steps=self.turn_on_led_transition_decoder_prefix_steps,
                step_embedding_dim=int(led_transition_decoder_cfg.get("step_embedding_dim", 8)),
            )
        self.vla_only_head = VLAOnlyActionHead(
            semantic_dim=model_cfg["semantic_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            chunk_len=dataset_cfg["chunk_len"],
            action_dim=dataset_cfg["action_dim"],
        )
        self.vla_only_gripper = VLAOnlyGripperHead(
            semantic_dim=model_cfg["semantic_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            chunk_len=dataset_cfg["chunk_len"],
        )

        if model_cfg["freeze_vla"]:
            for parameter in self.vla.parameters():
                parameter.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> SystemOutputs:
        obs = batch["obs"]
        proprio = batch["proprio"]
        instruction = batch["instruction"]
        phase = batch["phase"]

        semantic = None
        cond = None
        task_id = batch.get("task_id")
        if task_id is None:
            task_id = torch.zeros(obs.shape[0], device=obs.device, dtype=torch.long)
        step_index = batch.get("step_index")
        if step_index is None:
            step_index = torch.zeros(obs.shape[0], device=obs.device, dtype=torch.long)

        if self.variant in {"vla_only", "modular", "full"}:
            semantic = self.vla(obs, proprio, instruction)

        if self.variant == "diffusion_only":
            cond = torch.zeros(
                obs.shape[0],
                self.bridge.project[-1].out_features,
                device=obs.device,
            )
            action = self.executor(obs, proprio, cond)
            gripper_logits = self.executor_gripper(obs, proprio, cond)
        elif self.variant == "vla_only":
            action = self.vla_only_head(semantic)
            gripper_logits = self.vla_only_gripper(semantic)
        else:
            cond = self.bridge(semantic, phase)
            action = self.executor(obs, proprio, cond)
            gripper_logits = self.executor_gripper(obs, proprio, cond)
            if self.task_structured_decoder is not None:
                action_delta, gripper_delta = self.task_structured_decoder(obs, proprio, cond, task_id)
                action = action + self.task_structured_decoder_scale * action_delta
                gripper_logits = gripper_logits + self.task_structured_decoder_scale * gripper_delta
            if self.prefix_decoder is not None:
                prefix_action_delta, prefix_gripper_delta = self.prefix_decoder(
                    obs,
                    proprio,
                    cond,
                    task_id,
                    step_index,
                )
                prefix_mask = (
                    step_index < self.prefix_decoder_prefix_steps
                ).to(dtype=action.dtype).view(-1, 1, 1)
                gripper_prefix_mask = prefix_mask.squeeze(-1)
                action = action + self.prefix_decoder_scale * prefix_action_delta * prefix_mask
                gripper_logits = (
                    gripper_logits
                    + self.prefix_decoder_scale * prefix_gripper_delta * gripper_prefix_mask
                )
            if (
                self.turn_on_led_transition_decoder is not None
                and self.turn_on_led_transition_task_id is not None
            ):
                led_gripper_delta = self.turn_on_led_transition_decoder(
                    obs,
                    proprio,
                    cond,
                    step_index,
                )
                led_mask = (
                    (task_id == int(self.turn_on_led_transition_task_id))
                    & (step_index < self.turn_on_led_transition_decoder_prefix_steps)
                ).to(dtype=action.dtype).view(-1, 1)
                gripper_logits = (
                    gripper_logits
                    + self.turn_on_led_transition_decoder_scale * led_gripper_delta * led_mask
                )

        return SystemOutputs(
            action=action,
            gripper_logits=gripper_logits,
            semantic=semantic,
            cond=cond,
        )
