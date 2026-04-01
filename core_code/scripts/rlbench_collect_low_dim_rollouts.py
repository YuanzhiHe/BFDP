from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ReachTarget")
    parser.add_argument("--arm-action-mode", default="JointVelocity")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    arm_module = __import__(
        "rlbench.action_modes.arm_action_modes", fromlist=[args.arm_action_mode]
    )
    tasks_module = __import__("rlbench.tasks", fromlist=[args.task])
    ArmActionMode = getattr(arm_module, args.arm_action_mode)
    TaskClass = getattr(tasks_module, args.task)

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.set_all_low_dim(True)

    rng = np.random.default_rng(args.seed)
    env = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": args.task,
        "arm_action_mode": args.arm_action_mode,
        "policy": args.policy,
        "horizon": args.horizon,
        "collection_seed": args.seed,
        "episodes": [],
    }

    try:
        env = Environment(
            MoveArmThenGripper(ArmActionMode(), Discrete()),
            obs_config=obs_config,
            headless=True,
        )
        env.launch()
        task = env.get_task(TaskClass)
        action_shape = int(env.action_shape[0])

        for episode_idx in range(args.episodes):
            descriptions, obs = task.reset()
            steps = []
            total_reward = 0.0
            for step_idx in range(args.horizon):
                if args.policy == "zero":
                    action = np.zeros(action_shape, dtype=np.float32)
                else:
                    action = rng.normal(0.0, 0.05, size=action_shape).astype(np.float32)
                next_obs, reward, terminate = task.step(action)
                total_reward += float(reward)
                steps.append(
                    {
                        "step_index": step_idx,
                        "action": [float(x) for x in action.tolist()],
                        "reward": float(reward),
                        "terminate": bool(terminate),
                        "low_dim": [float(x) for x in next_obs.get_low_dim_data().tolist()],
                    }
                )
                obs = next_obs
                if terminate:
                    break
            payload["episodes"].append(
                {
                    "episode_index": episode_idx,
                    "descriptions": list(descriptions),
                    "total_reward": total_reward,
                    "steps": steps,
                }
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        total_steps = sum(len(episode["steps"]) for episode in payload["episodes"])
        terminate_count = sum(
            1 for episode in payload["episodes"] for step in episode["steps"] if step["terminate"]
        )
        summary = {
            "episodes": int(args.episodes),
            "total_steps": int(total_steps),
            "mean_episode_reward": float(
                sum(ep["total_reward"] for ep in payload["episodes"]) / max(1, len(payload["episodes"]))
            ),
            "terminate_count": int(terminate_count),
            "obs_dim": int(len(payload["episodes"][0]["steps"][0]["low_dim"])) if payload["episodes"] and payload["episodes"][0]["steps"] else None,
            "action_dim": int(action_shape),
            "task_name": args.task,
            "arm_action_mode": args.arm_action_mode,
            "policy": args.policy,
            "horizon": args.horizon,
            "collection_seed": args.seed,
            "output_path": str(output_path),
        }
        print(json.dumps(summary))
    finally:
        if env is not None:
            try:
                env.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
