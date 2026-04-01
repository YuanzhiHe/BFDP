from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ReachTarget")
    parser.add_argument("--arm-action-mode", default="JointVelocity")
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

    env = None
    try:
        env = Environment(
            MoveArmThenGripper(ArmActionMode(), Discrete()),
            obs_config=obs_config,
            headless=True,
        )
        env.launch()
        task = env.get_task(TaskClass)
        descriptions, obs = task.reset()
        action_shape = env.action_shape[0]
        action = np.zeros(action_shape, dtype=np.float32)
        obs, reward, terminate = task.step(action)
        payload = {
            "backend_available": True,
            "launch_ok": True,
            "reset_ok": True,
            "step_ok": True,
            "task_name": args.task,
            "action_shape": int(action_shape),
            "reward": float(reward),
            "terminate": bool(terminate),
            "low_dim_size": [int(x) for x in obs.get_low_dim_data().shape],
            "descriptions": list(descriptions),
            "error": None,
        }
        print(json.dumps(payload))
    finally:
        if env is not None:
            try:
                env.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
