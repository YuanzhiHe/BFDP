from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.common import build_coppeliasim_env
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


DISCOVERY_SNIPPET = r"""
import importlib
import json
import pkgutil

tasks_pkg = importlib.import_module("rlbench.tasks")
task_names = []
for module_info in pkgutil.iter_modules(tasks_pkg.__path__):
    module = importlib.import_module(f"rlbench.tasks.{module_info.name}")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr.__module__ == module.__name__
            and hasattr(tasks_pkg, attr.__name__)
        ):
            task_names.append(attr.__name__)
task_names = sorted(set(task_names))
print(json.dumps({"task_names": task_names, "count": len(task_names)}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--output", help="Optional output path for the catalog report.")
    args = parser.parse_args()

    config = load_config(args.config).data
    rlbench_cfg = config["benchmarks"]["rlbench"]
    env = build_coppeliasim_env(
        coppeliasim_root=rlbench_cfg["coppeliasim_root"],
        qt_qpa_platform=rlbench_cfg["qt_qpa_platform"],
    )
    completed = subprocess.run(
        [sys.executable, "-c", DISCOVERY_SNIPPET],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"rlbench task catalog discovery failed: {error}")

    payload = None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = json.loads(stripped)
            break
    if payload is None:
        raise RuntimeError(
            f"rlbench task catalog discovery completed without JSON output. stdout={completed.stdout!r}"
        )

    log_dir = ensure_dir(config["paths"]["log_dir"])
    output_path = Path(args.output) if args.output else log_dir / "rlbench_task_catalog.json"
    write_json(output_path, payload)
    print(f"saved_catalog={output_path}")
    print(payload)


if __name__ == "__main__":
    main()
