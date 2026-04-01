from __future__ import annotations

import argparse
import json


CAMERA_ATTR_PREFIX = {
    "front": ("front_camera", "front_rgb"),
    "left_shoulder": ("left_shoulder_camera", "left_shoulder_rgb"),
    "right_shoulder": ("right_shoulder_camera", "right_shoulder_rgb"),
    "overhead": ("overhead_camera", "overhead_rgb"),
    "wrist": ("wrist_camera", "wrist_rgb"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ReachTarget")
    parser.add_argument("--arm-action-mode", default="JointVelocity")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_ATTR_PREFIX.keys()),
        default="front",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--render-mode", default="OPENGL3")
    args = parser.parse_args()

    import numpy as np
    from pyrep.const import RenderMode
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
    camera_cfg_name, obs_rgb_attr = CAMERA_ATTR_PREFIX[args.camera]
    camera_cfg = getattr(obs_config, camera_cfg_name)
    camera_cfg.set_all(False)
    camera_cfg.rgb = True
    camera_cfg.image_size = (args.image_size, args.image_size)
    camera_cfg.render_mode = getattr(RenderMode, args.render_mode)

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
        image = getattr(obs, obs_rgb_attr)
        payload = {
            "backend_available": True,
            "launch_ok": True,
            "reset_ok": True,
            "step_ok": True,
            "task_name": args.task,
            "camera": args.camera,
            "render_mode": args.render_mode,
            "image_shape": [int(x) for x in image.shape],
            "action_shape": int(action_shape),
            "reward": float(reward),
            "terminate": bool(terminate),
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
