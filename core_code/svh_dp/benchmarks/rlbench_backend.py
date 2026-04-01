from __future__ import annotations

import importlib
from dataclasses import dataclass
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from svh_dp.benchmarks.common import build_coppeliasim_env


@dataclass
class RLBenchSmokeResult:
    backend_available: bool
    launch_ok: bool
    reset_ok: bool
    step_ok: bool
    task_name: str
    action_shape: int | None
    reward: float | None
    terminate: bool | None
    low_dim_size: list[int] | None
    descriptions: list[str] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_available": self.backend_available,
            "launch_ok": self.launch_ok,
            "reset_ok": self.reset_ok,
            "step_ok": self.step_ok,
            "task_name": self.task_name,
            "action_shape": self.action_shape,
            "reward": self.reward,
            "terminate": self.terminate,
            "low_dim_size": self.low_dim_size,
            "descriptions": self.descriptions,
            "error": self.error,
        }


SMOKE_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "rlbench_low_dim_smoke.py"
)
VISUAL_SMOKE_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "rlbench_visual_smoke.py"
)
WINDOWS_UNSUPPORTED_MESSAGE = (
    "RLBench/PyRep are not supported on Windows in the upstream setup. "
    "Use a Linux machine or WSL-compatible Linux environment for RLBench probes."
)


@dataclass
class RLBenchVisualSmokeResult:
    success: bool
    task_name: str
    qt_qpa_platform: str
    camera: str
    render_mode: str
    image_size: int
    return_code: int | None
    timed_out: bool
    image_shape: list[int] | None
    action_shape: int | None
    reward: float | None
    terminate: bool | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "task_name": self.task_name,
            "qt_qpa_platform": self.qt_qpa_platform,
            "camera": self.camera,
            "render_mode": self.render_mode,
            "image_size": self.image_size,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
            "image_shape": self.image_shape,
            "action_shape": self.action_shape,
            "reward": self.reward,
            "terminate": self.terminate,
            "error": self.error,
        }


def rlbench_available(config: dict) -> tuple[bool, str]:
    if os.name == "nt":
        return False, WINDOWS_UNSUPPORTED_MESSAGE
    coppeliasim_root = Path(
        os.environ.get("COPPELIASIM_ROOT", config["coppeliasim_root"])
    ).expanduser()
    if not coppeliasim_root.exists():
        return (
            False,
            "CoppeliaSim root not found. Set benchmarks.rlbench.coppeliasim_root or "
            "the COPPELIASIM_ROOT environment variable to your local install path.",
        )
    try:
        importlib.import_module("rlbench")
    except Exception:
        # The process-local import path is expected to fail on some systems until
        # the CoppeliaSim library path is applied in a fresh subprocess.
        pass
    env = build_coppeliasim_env(
        coppeliasim_root=config["coppeliasim_root"],
        qt_qpa_platform=config["qt_qpa_platform"],
    )
    completed = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--task", config["task"], "--arm-action-mode", config["arm_action_mode"]],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        return False, error
    return True, "rlbench low-dim subprocess smoke succeeded"


def run_low_dim_smoke(config: dict) -> RLBenchSmokeResult:
    if os.name == "nt":
        return RLBenchSmokeResult(
            backend_available=False,
            launch_ok=False,
            reset_ok=False,
            step_ok=False,
            task_name=config["task"],
            action_shape=None,
            reward=None,
            terminate=None,
            low_dim_size=None,
            descriptions=None,
            error=WINDOWS_UNSUPPORTED_MESSAGE,
        )
    coppeliasim_root = Path(
        os.environ.get("COPPELIASIM_ROOT", config["coppeliasim_root"])
    ).expanduser()
    if not coppeliasim_root.exists():
        return RLBenchSmokeResult(
            backend_available=False,
            launch_ok=False,
            reset_ok=False,
            step_ok=False,
            task_name=config["task"],
            action_shape=None,
            reward=None,
            terminate=None,
            low_dim_size=None,
            descriptions=None,
            error=(
                "CoppeliaSim root not found. Set benchmarks.rlbench.coppeliasim_root or "
                "the COPPELIASIM_ROOT environment variable to your local install path."
            ),
        )
    env = build_coppeliasim_env(
        coppeliasim_root=config["coppeliasim_root"],
        qt_qpa_platform=config["qt_qpa_platform"],
    )
    completed = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--task", config["task"], "--arm-action-mode", config["arm_action_mode"]],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:  # pragma: no cover - environment-dependent
        error = completed.stderr.strip() or completed.stdout.strip()
        return RLBenchSmokeResult(
            backend_available=False,
            launch_ok=False,
            reset_ok=False,
            step_ok=False,
            task_name=config["task"],
            action_shape=None,
            reward=None,
            terminate=None,
            low_dim_size=None,
            descriptions=None,
            error=error,
        )
    payload = None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = json.loads(stripped)
            break
    if payload is None:
        raise RuntimeError(
            f"rlbench smoke completed without a JSON payload. stdout={completed.stdout!r}"
        )
    return RLBenchSmokeResult(**payload)


def run_visual_smoke_suite(config: dict) -> list[RLBenchVisualSmokeResult]:
    visual_cfg = config["visual_probe"]
    if os.name == "nt":
        return [
            RLBenchVisualSmokeResult(
                success=False,
                task_name=visual_cfg.get("task", config["task"]),
                qt_qpa_platform=(visual_cfg.get("qt_qpa_platforms") or [config["qt_qpa_platform"]])[0],
                camera=(visual_cfg.get("cameras") or [visual_cfg.get("camera", "front")])[0],
                render_mode=(visual_cfg.get("render_modes") or ["OPENGL3"])[0],
                image_size=(visual_cfg.get("image_sizes") or [visual_cfg.get("image_size", 64)])[0],
                return_code=None,
                timed_out=False,
                image_shape=None,
                action_shape=None,
                reward=None,
                terminate=None,
                error=WINDOWS_UNSUPPORTED_MESSAGE,
            )
        ]
    results = []
    coppeliasim_root = Path(
        os.environ.get("COPPELIASIM_ROOT", config["coppeliasim_root"])
    ).expanduser()
    if not coppeliasim_root.exists():
        return [
            RLBenchVisualSmokeResult(
                success=False,
                task_name=visual_cfg.get("task", config["task"]),
                qt_qpa_platform=config["qt_qpa_platform"],
                camera=(visual_cfg.get("cameras") or [visual_cfg.get("camera", "front")])[0],
                render_mode=(visual_cfg.get("render_modes") or ["OPENGL3"])[0],
                image_size=(visual_cfg.get("image_sizes") or [visual_cfg.get("image_size", 64)])[0],
                return_code=None,
                timed_out=False,
                image_shape=None,
                action_shape=None,
                reward=None,
                terminate=None,
                error=(
                    "CoppeliaSim root not found. Set benchmarks.rlbench.coppeliasim_root or "
                    "the COPPELIASIM_ROOT environment variable to your local install path."
                ),
            )
        ]
    task_name = visual_cfg.get("task", config["task"])
    qt_qpa_platforms = visual_cfg.get("qt_qpa_platforms")
    if qt_qpa_platforms is None:
        qt_qpa_platforms = [config["qt_qpa_platform"]]
    cameras = visual_cfg.get("cameras")
    if cameras is None:
        cameras = [visual_cfg["camera"]]
    image_sizes = visual_cfg.get("image_sizes")
    if image_sizes is None:
        image_sizes = [visual_cfg["image_size"]]
    timeout_sec = visual_cfg.get("timeout_sec")

    for qt_qpa_platform in qt_qpa_platforms:
        env = build_coppeliasim_env(
            coppeliasim_root=config["coppeliasim_root"],
            qt_qpa_platform=qt_qpa_platform,
        )
        for camera in cameras:
            for image_size in image_sizes:
                for render_mode in visual_cfg["render_modes"]:
                    try:
                        completed = subprocess.run(
                            [
                                sys.executable,
                                str(VISUAL_SMOKE_SCRIPT),
                                "--task",
                                task_name,
                                "--arm-action-mode",
                                config["arm_action_mode"],
                                "--camera",
                                camera,
                                "--image-size",
                                str(image_size),
                                "--render-mode",
                                render_mode,
                            ],
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.TimeoutExpired as exc:
                        results.append(
                            RLBenchVisualSmokeResult(
                                success=False,
                                task_name=task_name,
                                qt_qpa_platform=qt_qpa_platform,
                                camera=camera,
                                render_mode=render_mode,
                                image_size=image_size,
                                return_code=None,
                                timed_out=True,
                                image_shape=None,
                                action_shape=None,
                                reward=None,
                                terminate=None,
                                error=f"visual smoke timed out after {timeout_sec}s: {exc}",
                            )
                        )
                        continue
                    if completed.returncode != 0:
                        error = completed.stderr.strip() or completed.stdout.strip()
                        results.append(
                            RLBenchVisualSmokeResult(
                                success=False,
                                task_name=task_name,
                                qt_qpa_platform=qt_qpa_platform,
                                camera=camera,
                                render_mode=render_mode,
                                image_size=image_size,
                                return_code=completed.returncode,
                                timed_out=False,
                                image_shape=None,
                                action_shape=None,
                                reward=None,
                                terminate=None,
                                error=error,
                            )
                        )
                        continue
                    payload = None
                    for line in completed.stdout.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("{") and stripped.endswith("}"):
                            payload = json.loads(stripped)
                            break
                    if payload is None:
                        results.append(
                            RLBenchVisualSmokeResult(
                                success=False,
                                task_name=task_name,
                                qt_qpa_platform=qt_qpa_platform,
                                camera=camera,
                                render_mode=render_mode,
                                image_size=image_size,
                                return_code=completed.returncode,
                                timed_out=False,
                                image_shape=None,
                                action_shape=None,
                                reward=None,
                                terminate=None,
                                error=f"visual smoke completed without JSON payload. stdout={completed.stdout!r}",
                            )
                        )
                        continue
                    results.append(
                        RLBenchVisualSmokeResult(
                            success=True,
                            task_name=payload["task_name"],
                            qt_qpa_platform=qt_qpa_platform,
                            camera=payload["camera"],
                            render_mode=payload["render_mode"],
                            image_size=image_size,
                            return_code=completed.returncode,
                            timed_out=False,
                            image_shape=payload["image_shape"],
                            action_shape=payload["action_shape"],
                            reward=payload["reward"],
                            terminate=payload["terminate"],
                            error=None,
                        )
                    )
    return results
