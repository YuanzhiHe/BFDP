from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.config import load_config
from svh_dp.benchmarks.calvin_backend import build_calvin_probe
from svh_dp.utils.common import ensure_dir, write_json


def main() -> None:
    config = load_config(ROOT / "config" / "default.yaml").data
    calvin_cfg = dict(config["benchmarks"]["calvin"])
    calvin_cfg["probe"] = dict(calvin_cfg["probe"])
    calvin_cfg["probe"]["project_root"] = str(ROOT.parents[1])
    probe = build_calvin_probe(calvin_cfg)
    activation_script_path = Path(calvin_cfg["workspace"]["activation_script_path"])
    if not activation_script_path.is_absolute():
        activation_script_path = (ROOT.parents[1] / activation_script_path).resolve()
    activation_powershell_script_path = activation_script_path.with_suffix(".ps1")
    suggested_activation_command = (
        f". {activation_powershell_script_path}"
        if os.name == "nt"
        else f"source {activation_script_path}"
    )

    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "available": probe["available"],
        "bootstrap_available": probe["bootstrap_available"],
        "source_layout_available": probe["source_layout_available"],
        "message": probe["message"],
        "module_status": probe["module_status"],
        "module_probe": probe["module_probe"],
        "available_modules": [
            item["module_name"] for item in probe["module_status"] if item["available"]
        ],
        "missing_modules": [
            item["module_name"]
            for item in probe["module_status"]
            if not item["available"]
        ],
        "search_roots": probe["search_roots"],
        "workspace_candidates": probe["workspace_candidates"],
        "bootstrap_root": probe["bootstrap_root"],
        "pythonpath_entries": probe["pythonpath_entries"],
        "dependency_blockers": probe["dependency_blockers"],
        "smoke_dependency_blockers": probe["smoke_dependency_blockers"],
        "runtime_ready": probe["runtime_ready"],
        "smoke_imports": probe["smoke_imports"],
        "smoke_status": probe["smoke_status"],
        "activation_script_path": str(activation_script_path),
        "activation_powershell_script_path": str(activation_powershell_script_path),
        "suggested_source_command": suggested_activation_command,
        "diagnosis": probe["diagnosis"],
    }
    log_dir = ensure_dir(config["paths"]["log_dir"])
    output_path = log_dir / "calvin_env_diagnostic.json"
    write_json(output_path, payload)
    print(f"saved_diagnostic={output_path}")
    print(payload)


if __name__ == "__main__":
    main()
