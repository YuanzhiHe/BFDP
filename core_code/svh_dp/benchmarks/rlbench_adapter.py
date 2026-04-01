from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from svh_dp.benchmarks.common import build_coppeliasim_env


ROLLOUT_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "rlbench_collect_low_dim_rollouts.py"
)


@dataclass
class RLBenchAdapterSummary:
    episodes: int
    total_steps: int
    mean_episode_reward: float
    terminate_count: int
    obs_dim: int | None
    action_dim: int | None
    task_name: str | list[str]
    arm_action_mode: str | None
    policy: str
    horizon: int | None
    collection_seed: int | None
    output_path: str
    obs_dim_set: list[int] | None = None
    action_dim_set: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "mean_episode_reward": self.mean_episode_reward,
            "terminate_count": self.terminate_count,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "task_name": self.task_name,
            "arm_action_mode": self.arm_action_mode,
            "policy": self.policy,
            "horizon": self.horizon,
            "collection_seed": self.collection_seed,
            "output_path": self.output_path,
            "obs_dim_set": self.obs_dim_set,
            "action_dim_set": self.action_dim_set,
        }


def _run_rollout_export(command: list[str], env: dict[str, str]) -> RLBenchAdapterSummary:
    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"rlbench rollout export failed: {error}")

    payload = None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = json.loads(stripped)
            break
    if payload is None:
        raise RuntimeError(
            f"rlbench rollout export completed without a JSON summary. stdout={completed.stdout!r}"
        )
    return RLBenchAdapterSummary(**payload)


def collect_low_dim_rollouts(
    rlbench_cfg: dict,
    adapter_cfg: dict,
    output_path: str | Path,
    task: str | None = None,
    arm_action_mode: str | None = None,
    episodes: int | None = None,
    horizon: int | None = None,
    policy: str | None = None,
    seed: int | None = None,
) -> RLBenchAdapterSummary:
    output_path = Path(output_path)
    env = build_coppeliasim_env(
        coppeliasim_root=rlbench_cfg["coppeliasim_root"],
        qt_qpa_platform=rlbench_cfg["qt_qpa_platform"],
    )
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--task",
        rlbench_cfg["task"] if task is None else task,
        "--arm-action-mode",
        rlbench_cfg["arm_action_mode"] if arm_action_mode is None else arm_action_mode,
        "--episodes",
        str(adapter_cfg["sample_episodes"] if episodes is None else episodes),
        "--horizon",
        str(adapter_cfg["rollout_horizon"] if horizon is None else horizon),
        "--policy",
        adapter_cfg["policy"] if policy is None else policy,
        "--seed",
        str(7 if seed is None else seed),
        "--output",
        str(output_path),
    ]
    return _run_rollout_export(command, env=env)


def load_rollout_export(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_rollout_export(payload: dict[str, Any], output_path: str | Path) -> RLBenchAdapterSummary:
    episodes = payload["episodes"]
    total_steps = sum(len(episode["steps"]) for episode in episodes)
    terminate_count = sum(
        1 for episode in episodes for step in episode["steps"] if step["terminate"]
    )
    rewards = [episode["total_reward"] for episode in episodes]
    mean_episode_reward = sum(rewards) / max(1, len(rewards))
    obs_dim_set = []
    action_dim_set = []
    for episode in episodes:
        if not episode["steps"]:
            continue
        obs_dim = len(episode["steps"][0]["low_dim"])
        action_dim = len(episode["steps"][0]["action"])
        if obs_dim not in obs_dim_set:
            obs_dim_set.append(obs_dim)
        if action_dim not in action_dim_set:
            action_dim_set.append(action_dim)
    first_obs_dim = obs_dim_set[0] if len(obs_dim_set) == 1 else None
    first_action_dim = action_dim_set[0] if len(action_dim_set) == 1 else None
    task_name: str | list[str] = payload["task_name"]
    if episodes:
        episode_task_names = []
        for episode in episodes:
            name = episode.get("task_name", payload["task_name"])
            if name not in episode_task_names:
                episode_task_names.append(name)
        if len(episode_task_names) > 1:
            task_name = episode_task_names
    return RLBenchAdapterSummary(
        episodes=len(episodes),
        total_steps=total_steps,
        mean_episode_reward=mean_episode_reward,
        terminate_count=terminate_count,
        obs_dim=first_obs_dim,
        action_dim=first_action_dim,
        task_name=payload["task_name"],
        arm_action_mode=payload.get("arm_action_mode"),
        policy=payload["policy"],
        horizon=payload.get("horizon"),
        collection_seed=payload.get("collection_seed"),
        output_path=str(output_path),
        obs_dim_set=obs_dim_set or None,
        action_dim_set=action_dim_set or None,
    )


def merge_rollout_exports(
    payloads: list[dict[str, Any]],
    output_path: str | Path,
) -> tuple[dict[str, Any], RLBenchAdapterSummary]:
    if not payloads:
        raise ValueError("payloads must be non-empty")

    merged_episodes = []
    task_names = []
    arm_action_mode = payloads[0].get("arm_action_mode")
    policy = payloads[0]["policy"]
    horizon = payloads[0].get("horizon")
    task_episode_counts: dict[str, int] = {}
    for shard_index, payload in enumerate(payloads):
        if payload["policy"] != policy:
            raise ValueError("all rollout payloads must share the same policy")
        if payload.get("arm_action_mode") != arm_action_mode:
            raise ValueError("all rollout payloads must share the same arm_action_mode")
        task_name = payload["task_name"]
        if task_name not in task_names:
            task_names.append(task_name)
        for episode in payload["episodes"]:
            merged_episode = dict(episode)
            merged_episode["episode_index"] = len(merged_episodes)
            merged_episode["shard_index"] = shard_index
            merged_episode["task_name"] = task_name
            merged_episodes.append(merged_episode)
            task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1

    merged_payload = {
        "task_name": task_names[0] if len(task_names) == 1 else task_names,
        "arm_action_mode": arm_action_mode,
        "policy": policy,
        "horizon": horizon,
        "source_shards": len(payloads),
        "task_names": task_names,
        "task_episode_counts": task_episode_counts,
        "episodes": merged_episodes,
    }
    summary = summarize_rollout_export(merged_payload, output_path=output_path)
    return merged_payload, summary
