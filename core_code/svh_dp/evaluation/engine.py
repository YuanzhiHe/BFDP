from __future__ import annotations

from collections import defaultdict

import torch


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    success_threshold: float,
    strict_thresholds: list[float] | None = None,
    completion_cfg: dict | None = None,
) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total = 0
    success = 0
    mse_values: list[float] = []
    strict_hits = [0 for _ in strict_thresholds or []]
    episode_step_mse: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        mse_per_sample = (outputs.action - batch["target_action"]).pow(2).mean(dim=(1, 2))
        mae_per_sample = (outputs.action - batch["target_action"]).abs().mean(dim=(1, 2))
        mse_sum += float(mse_per_sample.sum().item())
        mae_sum += float(mae_per_sample.sum().item())
        success += int((mse_per_sample < success_threshold).sum().item())
        mse_values.extend(float(value) for value in mse_per_sample.detach().cpu().tolist())
        if strict_thresholds:
            for index, threshold in enumerate(strict_thresholds):
                strict_hits[index] += int((mse_per_sample < threshold).sum().item())
        if "episode_id" in batch:
            episode_ids = batch["episode_id"].detach().cpu().tolist()
            step_indices = batch["step_index"].detach().cpu().tolist()
            for episode_id, step_index, mse_value in zip(
                episode_ids,
                step_indices,
                mse_per_sample.detach().cpu().tolist(),
            ):
                episode_step_mse[int(episode_id)].append((int(step_index), float(mse_value)))
        total += int(mse_per_sample.shape[0])
    mse_tensor = torch.tensor(mse_values, dtype=torch.float32) if mse_values else torch.zeros(1)
    episode_mean_values = [
        sum(value for _, value in values) / len(values)
        for _, values in sorted(episode_step_mse.items())
        if values
    ]
    episode_max_values = [
        max(value for _, value in values)
        for _, values in sorted(episode_step_mse.items())
        if values
    ]
    episode_mean_tensor = (
        torch.tensor(episode_mean_values, dtype=torch.float32)
        if episode_mean_values
        else torch.zeros(1)
    )
    episode_max_tensor = (
        torch.tensor(episode_max_values, dtype=torch.float32)
        if episode_max_values
        else torch.zeros(1)
    )
    sequence_success_rate = 0.0
    strict_sequence_success_rates: list[dict[str, float]] = []
    terminal_step_mse_values: list[float] = []
    tail_mean_mse_values: list[float] = []
    tail_max_mse_values: list[float] = []
    task_completion_rate = 0.0
    if episode_step_mse:
        ordered_episode_values = {
            episode_id: [value for _, value in sorted(values, key=lambda item: item[0])]
            for episode_id, values in episode_step_mse.items()
        }
        sequence_success_rate = sum(
            1 for values in ordered_episode_values.values() if all(value < success_threshold for value in values)
        ) / len(ordered_episode_values)
        strict_sequence_success_rates = [
            {
                "threshold": threshold,
                "success_rate": (
                    sum(
                        1
                        for values in ordered_episode_values.values()
                        if all(value < threshold for value in values)
                    )
                    / len(ordered_episode_values)
                ),
            }
            for threshold in strict_thresholds or []
        ]
        if completion_cfg:
            tail_steps = max(1, int(completion_cfg["tail_steps"]))
            terminal_threshold = float(completion_cfg["terminal_threshold"])
            tail_mean_threshold = float(completion_cfg["tail_mean_threshold"])
            tail_max_threshold = float(completion_cfg["tail_max_threshold"])
            completed = 0
            for values in ordered_episode_values.values():
                tail_values = values[-tail_steps:]
                terminal_value = values[-1]
                tail_mean = sum(tail_values) / len(tail_values)
                tail_max = max(tail_values)
                terminal_step_mse_values.append(terminal_value)
                tail_mean_mse_values.append(tail_mean)
                tail_max_mse_values.append(tail_max)
                if (
                    terminal_value < terminal_threshold
                    and tail_mean < tail_mean_threshold
                    and tail_max < tail_max_threshold
                ):
                    completed += 1
            task_completion_rate = completed / len(ordered_episode_values)
    terminal_step_tensor = (
        torch.tensor(terminal_step_mse_values, dtype=torch.float32)
        if terminal_step_mse_values
        else torch.zeros(1)
    )
    tail_mean_tensor = (
        torch.tensor(tail_mean_mse_values, dtype=torch.float32)
        if tail_mean_mse_values
        else torch.zeros(1)
    )
    tail_max_tensor = (
        torch.tensor(tail_max_mse_values, dtype=torch.float32)
        if tail_max_mse_values
        else torch.zeros(1)
    )
    return {
        "mse": mse_sum / max(1, total),
        "mae": mae_sum / max(1, total),
        "success_rate": success / max(1, total),
        "median_mse": float(torch.quantile(mse_tensor, 0.5).item()),
        "p90_mse": float(torch.quantile(mse_tensor, 0.9).item()),
        "max_mse": float(mse_tensor.max().item()),
        "sequence_count": len(episode_step_mse),
        "sequence_mean_mse": float(episode_mean_tensor.mean().item()),
        "sequence_median_mean_mse": float(torch.quantile(episode_mean_tensor, 0.5).item()),
        "sequence_p90_mean_mse": float(torch.quantile(episode_mean_tensor, 0.9).item()),
        "sequence_mean_max_step_mse": float(episode_max_tensor.mean().item()),
        "sequence_success_rate": sequence_success_rate,
        "terminal_step_mean_mse": float(terminal_step_tensor.mean().item()),
        "terminal_step_median_mse": float(torch.quantile(terminal_step_tensor, 0.5).item()),
        "tail_mean_mse": float(tail_mean_tensor.mean().item()),
        "tail_max_mse": float(tail_max_tensor.mean().item()),
        "task_completion_rate": task_completion_rate,
        "strict_success_rates": [
            {
                "threshold": threshold,
                "success_rate": hits / max(1, total),
            }
            for threshold, hits in zip(strict_thresholds or [], strict_hits)
        ],
        "strict_sequence_success_rates": strict_sequence_success_rates,
    }
