from __future__ import annotations

import os
from pathlib import Path


def build_coppeliasim_env(
    coppeliasim_root: str, qt_qpa_platform: str, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    configured_root = env.get("COPPELIASIM_ROOT", coppeliasim_root)
    root = Path(configured_root).expanduser()
    if root.exists():
        resolved_root = root.resolve()
    else:
        resolved_root = root
    if os.name == "nt":
        current = env.get("PATH", "")
        paths = [str(resolved_root)]
        if current:
            paths.append(current)
        env["PATH"] = os.pathsep.join(paths)
    else:
        current = env.get("LD_LIBRARY_PATH", "")
        paths = [str(resolved_root)]
        if current:
            paths.append(current)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    env["QT_QPA_PLATFORM"] = qt_qpa_platform
    env["COPPELIASIM_ROOT"] = str(resolved_root)
    return env
