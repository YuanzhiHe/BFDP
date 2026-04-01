from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from svh_dp.evaluation.engine import evaluate


@dataclass
class TrainResult:
    best_state: dict
    metrics: dict


def compute_loss(
    prediction: torch.Tensor,
    gripper_logits: torch.Tensor | None,
    target: torch.Tensor,
    phase: torch.Tensor,
    phase_loss_weight: float,
    step_index: torch.Tensor | None = None,
    episode_length: torch.Tensor | None = None,
    supervision_cfg: dict | None = None,
    gripper_change: torch.Tensor | None = None,
    turn_on_led_flag: torch.Tensor | None = None,
    turn_on_led_early_flag: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (prediction - target).pow(2)
    if supervision_cfg and float(supervision_cfg.get("gripper_dim_weight", 1.0)) != 1.0:
        action_dim_weights = torch.ones(
            squared_error.shape[2],
            device=squared_error.device,
            dtype=squared_error.dtype,
        )
        action_dim_weights[-1] = float(supervision_cfg["gripper_dim_weight"])
        squared_error = squared_error * action_dim_weights.view(1, 1, -1)
    offset_mse = squared_error.mean(dim=2)
    if supervision_cfg:
        chunk_len = offset_mse.shape[1]
        if chunk_len > 1:
            chunk_tail_weight = float(supervision_cfg["chunk_tail_weight"])
            chunk_weights = torch.linspace(
                1.0,
                chunk_tail_weight,
                steps=chunk_len,
                device=offset_mse.device,
                dtype=offset_mse.dtype,
            )
        else:
            chunk_weights = torch.ones(1, device=offset_mse.device, dtype=offset_mse.dtype)
        offset_mse = offset_mse * chunk_weights.unsqueeze(0)
        mse = offset_mse.mean(dim=1) / max(1.0, float(chunk_weights.mean().item()))
    else:
        mse = offset_mse.mean(dim=1)
    weights = 1.0 + phase.float() * phase_loss_weight
    if supervision_cfg and step_index is not None and episode_length is not None:
        progress = step_index.float() / torch.clamp(episode_length.float() - 1.0, min=1.0)
        tail_mask = progress >= float(supervision_cfg["tail_start_fraction"])
        weights = weights * torch.where(
            tail_mask,
            torch.full_like(weights, float(supervision_cfg["tail_sample_weight"])),
            torch.ones_like(weights),
        )
        terminal_mask = step_index >= (episode_length - 1)
        weights = weights * torch.where(
            terminal_mask,
            torch.full_like(weights, float(supervision_cfg["terminal_sample_weight"])),
            torch.ones_like(weights),
        )
        if gripper_change is not None:
            change_mask = gripper_change > 0.5
            weights = weights * torch.where(
                change_mask,
                torch.full_like(weights, float(supervision_cfg["gripper_change_sample_weight"])),
                torch.ones_like(weights),
            )
        if turn_on_led_flag is not None:
            led_mask = turn_on_led_flag > 0.5
            weights = weights * torch.where(
                led_mask,
                torch.full_like(weights, float(supervision_cfg["turn_on_led_sample_weight"])),
                torch.ones_like(weights),
            )
        if turn_on_led_early_flag is not None:
            led_early_mask = turn_on_led_early_flag > 0.5
            weights = weights * torch.where(
                led_early_mask,
                torch.full_like(weights, float(supervision_cfg["turn_on_led_early_sample_weight"])),
                torch.ones_like(weights),
            )
    loss = (mse * weights).mean()
    if supervision_cfg and gripper_logits is not None:
        target_gripper = target[:, :, -1]
        target_sign = torch.where(
            target_gripper >= 0.0,
            torch.ones_like(target_gripper),
            -torch.ones_like(target_gripper),
        )
        gripper_logistic = torch.log1p(torch.exp(-target_sign * gripper_logits)).mean()
        loss = loss + float(supervision_cfg["gripper_sign_head_weight"]) * gripper_logistic
    if supervision_cfg and turn_on_led_early_flag is not None:
        led_early_mask = turn_on_led_early_flag > 0.5
        if torch.any(led_early_mask):
            transition_offsets = int(supervision_cfg["turn_on_led_transition_offsets"])
            transition_offsets = min(transition_offsets, prediction.shape[1])
            pred_gripper = prediction[led_early_mask, :transition_offsets, -1]
            target_gripper = target[led_early_mask, :transition_offsets, -1]
            target_sign = torch.where(
                target_gripper >= 0.0,
                torch.ones_like(target_gripper),
                -torch.ones_like(target_gripper),
            )
            margin = float(supervision_cfg["turn_on_led_transition_margin"])
            hinge = torch.relu(margin - pred_gripper * target_sign).mean()
            loss = loss + float(supervision_cfg["turn_on_led_transition_weight"]) * hinge
            if gripper_logits is not None:
                pred_gripper_logits = gripper_logits[led_early_mask, :transition_offsets]
                logistic = torch.log1p(torch.exp(-target_sign * pred_gripper_logits)).mean()
                loss = loss + float(supervision_cfg["turn_on_led_transition_sign_weight"]) * logistic
    return loss


def fit_model(
    model: nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    training_cfg: dict,
    device: torch.device,
) -> TrainResult:
    best_selection_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_weighted_completion = float("-inf")
    best_weighted_success = float("-inf")
    best_weighted_mse = float("inf")
    best_state = None
    history: list[dict] = []
    strict_thresholds = [
        training_cfg["action_success_threshold"] * multiplier
        for multiplier in training_cfg.get("strict_success_multipliers", [])
    ]
    completion_cfg = {
        "tail_steps": training_cfg.get("completion_tail_steps", 2),
        "terminal_threshold": training_cfg["action_success_threshold"]
        * training_cfg.get("completion_terminal_multiplier", 0.05),
        "tail_mean_threshold": training_cfg["action_success_threshold"]
        * training_cfg.get("completion_tail_mean_multiplier", 0.0625),
        "tail_max_threshold": training_cfg["action_success_threshold"]
        * training_cfg.get("completion_tail_max_multiplier", 0.075),
    }
    supervision_cfg = {
        "tail_start_fraction": training_cfg.get("supervision_tail_start_fraction", 0.75),
        "tail_sample_weight": training_cfg.get("supervision_tail_sample_weight", 1.5),
        "terminal_sample_weight": training_cfg.get("supervision_terminal_sample_weight", 1.75),
        "chunk_tail_weight": training_cfg.get("supervision_chunk_tail_weight", 1.25),
        "gripper_dim_weight": training_cfg.get("supervision_gripper_dim_weight", 1.0),
        "gripper_change_sample_weight": training_cfg.get("supervision_gripper_change_sample_weight", 1.0),
        "gripper_sign_head_weight": training_cfg.get("supervision_gripper_sign_head_weight", 0.0),
        "turn_on_led_sample_weight": training_cfg.get("supervision_turn_on_led_sample_weight", 1.0),
        "turn_on_led_early_sample_weight": training_cfg.get("supervision_turn_on_led_early_sample_weight", 1.0),
        "turn_on_led_transition_weight": training_cfg.get("supervision_turn_on_led_transition_weight", 0.0),
        "turn_on_led_transition_sign_weight": training_cfg.get("supervision_turn_on_led_transition_sign_weight", 0.0),
        "turn_on_led_transition_margin": training_cfg.get("supervision_turn_on_led_transition_margin", 0.5),
        "turn_on_led_transition_offsets": training_cfg.get("supervision_turn_on_led_transition_offsets", 3),
    }

    for epoch in range(training_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = compute_loss(
                outputs.action,
                outputs.gripper_logits,
                batch["target_action"],
                batch["phase"],
                training_cfg["phase_loss_weight"],
                step_index=batch.get("step_index"),
                episode_length=batch.get("episode_length"),
                supervision_cfg=supervision_cfg,
                gripper_change=batch.get("gripper_change"),
                turn_on_led_flag=batch.get("turn_on_led_flag"),
                turn_on_led_early_flag=batch.get("turn_on_led_early_flag"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_cfg["grad_clip"]
            )
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        eval_nominal = evaluate(
            model,
            loaders["val_nominal"],
            device,
            training_cfg["action_success_threshold"],
            strict_thresholds=strict_thresholds,
            completion_cfg=completion_cfg,
        )
        eval_perturbed = evaluate(
            model,
            loaders["val_perturbed"],
            device,
            training_cfg["action_success_threshold"],
            strict_thresholds=strict_thresholds,
            completion_cfg=completion_cfg,
        )
        weighted_completion = (
            eval_nominal["task_completion_rate"] * 0.6
            + eval_perturbed["task_completion_rate"] * 0.4
        )
        weighted_success = eval_nominal["success_rate"] * 0.6 + eval_perturbed["success_rate"] * 0.4
        weighted_mse = eval_nominal["mse"] * 0.6 + eval_perturbed["mse"] * 0.4
        weighted_terminal_mse = (
            eval_nominal["terminal_step_mean_mse"] * 0.6
            + eval_perturbed["terminal_step_mean_mse"] * 0.4
        )
        selection_key = (
            weighted_completion,
            weighted_success,
            -weighted_terminal_mse,
            -weighted_mse,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, batch_count),
            "nominal_success_rate": eval_nominal["success_rate"],
            "perturbed_success_rate": eval_perturbed["success_rate"],
            "nominal_mse": eval_nominal["mse"],
            "perturbed_mse": eval_perturbed["mse"],
            "nominal_mae": eval_nominal["mae"],
            "perturbed_mae": eval_perturbed["mae"],
            "nominal_median_mse": eval_nominal["median_mse"],
            "perturbed_median_mse": eval_perturbed["median_mse"],
            "nominal_p90_mse": eval_nominal["p90_mse"],
            "perturbed_p90_mse": eval_perturbed["p90_mse"],
            "nominal_max_mse": eval_nominal["max_mse"],
            "perturbed_max_mse": eval_perturbed["max_mse"],
            "nominal_sequence_count": eval_nominal["sequence_count"],
            "perturbed_sequence_count": eval_perturbed["sequence_count"],
            "nominal_sequence_mean_mse": eval_nominal["sequence_mean_mse"],
            "perturbed_sequence_mean_mse": eval_perturbed["sequence_mean_mse"],
            "nominal_sequence_median_mean_mse": eval_nominal["sequence_median_mean_mse"],
            "perturbed_sequence_median_mean_mse": eval_perturbed["sequence_median_mean_mse"],
            "nominal_sequence_p90_mean_mse": eval_nominal["sequence_p90_mean_mse"],
            "perturbed_sequence_p90_mean_mse": eval_perturbed["sequence_p90_mean_mse"],
            "nominal_sequence_mean_max_step_mse": eval_nominal["sequence_mean_max_step_mse"],
            "perturbed_sequence_mean_max_step_mse": eval_perturbed["sequence_mean_max_step_mse"],
            "nominal_sequence_success_rate": eval_nominal["sequence_success_rate"],
            "perturbed_sequence_success_rate": eval_perturbed["sequence_success_rate"],
            "nominal_terminal_step_mean_mse": eval_nominal["terminal_step_mean_mse"],
            "perturbed_terminal_step_mean_mse": eval_perturbed["terminal_step_mean_mse"],
            "nominal_terminal_step_median_mse": eval_nominal["terminal_step_median_mse"],
            "perturbed_terminal_step_median_mse": eval_perturbed["terminal_step_median_mse"],
            "nominal_tail_mean_mse": eval_nominal["tail_mean_mse"],
            "perturbed_tail_mean_mse": eval_perturbed["tail_mean_mse"],
            "nominal_tail_max_mse": eval_nominal["tail_max_mse"],
            "perturbed_tail_max_mse": eval_perturbed["tail_max_mse"],
            "nominal_task_completion_rate": eval_nominal["task_completion_rate"],
            "perturbed_task_completion_rate": eval_perturbed["task_completion_rate"],
            "nominal_strict_success_rates": eval_nominal["strict_success_rates"],
            "perturbed_strict_success_rates": eval_perturbed["strict_success_rates"],
            "nominal_strict_sequence_success_rates": eval_nominal["strict_sequence_success_rates"],
            "perturbed_strict_sequence_success_rates": eval_perturbed["strict_sequence_success_rates"],
            "completion_cfg": completion_cfg,
            "supervision_cfg": supervision_cfg,
            "weighted_completion": weighted_completion,
            "weighted_success": weighted_success,
            "weighted_mse": weighted_mse,
        }
        history.append(epoch_metrics)
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_weighted_completion = weighted_completion
            best_weighted_success = weighted_success
            best_weighted_mse = weighted_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    final_metrics = {
        "history": history,
        "completion_cfg": completion_cfg,
        "supervision_cfg": supervision_cfg,
        "best_weighted_completion": best_weighted_completion,
        "best_weighted_success": best_weighted_success,
        "best_weighted_mse": best_weighted_mse,
        "final_nominal_success_rate": history[-1]["nominal_success_rate"],
        "final_perturbed_success_rate": history[-1]["perturbed_success_rate"],
        "final_nominal_mse": history[-1]["nominal_mse"],
        "final_perturbed_mse": history[-1]["perturbed_mse"],
        "final_nominal_mae": history[-1]["nominal_mae"],
        "final_perturbed_mae": history[-1]["perturbed_mae"],
        "final_nominal_median_mse": history[-1]["nominal_median_mse"],
        "final_perturbed_median_mse": history[-1]["perturbed_median_mse"],
        "final_nominal_p90_mse": history[-1]["nominal_p90_mse"],
        "final_perturbed_p90_mse": history[-1]["perturbed_p90_mse"],
        "final_nominal_max_mse": history[-1]["nominal_max_mse"],
        "final_perturbed_max_mse": history[-1]["perturbed_max_mse"],
        "final_nominal_sequence_count": history[-1]["nominal_sequence_count"],
        "final_perturbed_sequence_count": history[-1]["perturbed_sequence_count"],
        "final_nominal_sequence_mean_mse": history[-1]["nominal_sequence_mean_mse"],
        "final_perturbed_sequence_mean_mse": history[-1]["perturbed_sequence_mean_mse"],
        "final_nominal_sequence_median_mean_mse": history[-1]["nominal_sequence_median_mean_mse"],
        "final_perturbed_sequence_median_mean_mse": history[-1]["perturbed_sequence_median_mean_mse"],
        "final_nominal_sequence_p90_mean_mse": history[-1]["nominal_sequence_p90_mean_mse"],
        "final_perturbed_sequence_p90_mean_mse": history[-1]["perturbed_sequence_p90_mean_mse"],
        "final_nominal_sequence_mean_max_step_mse": history[-1]["nominal_sequence_mean_max_step_mse"],
        "final_perturbed_sequence_mean_max_step_mse": history[-1]["perturbed_sequence_mean_max_step_mse"],
        "final_nominal_sequence_success_rate": history[-1]["nominal_sequence_success_rate"],
        "final_perturbed_sequence_success_rate": history[-1]["perturbed_sequence_success_rate"],
        "final_nominal_terminal_step_mean_mse": history[-1]["nominal_terminal_step_mean_mse"],
        "final_perturbed_terminal_step_mean_mse": history[-1]["perturbed_terminal_step_mean_mse"],
        "final_nominal_terminal_step_median_mse": history[-1]["nominal_terminal_step_median_mse"],
        "final_perturbed_terminal_step_median_mse": history[-1]["perturbed_terminal_step_median_mse"],
        "final_nominal_tail_mean_mse": history[-1]["nominal_tail_mean_mse"],
        "final_perturbed_tail_mean_mse": history[-1]["perturbed_tail_mean_mse"],
        "final_nominal_tail_max_mse": history[-1]["nominal_tail_max_mse"],
        "final_perturbed_tail_max_mse": history[-1]["perturbed_tail_max_mse"],
        "final_nominal_task_completion_rate": history[-1]["nominal_task_completion_rate"],
        "final_perturbed_task_completion_rate": history[-1]["perturbed_task_completion_rate"],
        "final_nominal_strict_success_rates": history[-1]["nominal_strict_success_rates"],
        "final_perturbed_strict_success_rates": history[-1]["perturbed_strict_success_rates"],
        "final_nominal_strict_sequence_success_rates": history[-1]["nominal_strict_sequence_success_rates"],
        "final_perturbed_strict_sequence_success_rates": history[-1]["perturbed_strict_sequence_success_rates"],
    }
    return TrainResult(best_state=best_state, metrics=final_metrics)
