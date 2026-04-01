from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_backend import build_calvin_probe
from svh_dp.benchmarks.rlbench_backend import (
    rlbench_available,
    run_low_dim_smoke,
    run_visual_smoke_suite,
)
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def main() -> None:
    config = load_config(ROOT / "config" / "default.yaml").data
    rlbench_cfg = config["benchmarks"]["rlbench"]
    calvin_cfg = dict(config["benchmarks"]["calvin"])
    calvin_cfg["probe"] = dict(calvin_cfg["probe"])
    calvin_cfg["probe"]["project_root"] = str(ROOT.parents[1])
    activation_script_path = Path(calvin_cfg["workspace"]["activation_script_path"])
    if not activation_script_path.is_absolute():
        activation_script_path = (ROOT.parents[1] / activation_script_path).resolve()
    activation_powershell_script_path = activation_script_path.with_suffix(".ps1")
    suggested_activation_command = (
        f". {activation_powershell_script_path}"
        if os.name == "nt"
        else f"source {activation_script_path}"
    )

    rlbench_ok, rlbench_message = rlbench_available(rlbench_cfg)
    calvin_probe = build_calvin_probe(calvin_cfg)
    smoke = run_low_dim_smoke(rlbench_cfg)
    visual_smokes = run_visual_smoke_suite(rlbench_cfg)

    payload = {
        "rlbench": {
            "available": rlbench_ok,
            "message": rlbench_message,
            "smoke": smoke.to_dict(),
            "visual_probe": [item.to_dict() for item in visual_smokes],
        },
        "calvin": {
            "available": calvin_probe["available"],
            "bootstrap_available": calvin_probe["bootstrap_available"],
            "source_layout_available": calvin_probe["source_layout_available"],
            "message": calvin_probe["message"],
            "module_probe": calvin_probe["module_probe"],
            "module_status": calvin_probe["module_status"],
            "search_roots": calvin_probe["search_roots"],
            "workspace_candidates": calvin_probe["workspace_candidates"],
            "bootstrap_root": calvin_probe["bootstrap_root"],
            "pythonpath_entries": calvin_probe["pythonpath_entries"],
            "dependency_blockers": calvin_probe["dependency_blockers"],
            "smoke_dependency_blockers": calvin_probe["smoke_dependency_blockers"],
            "runtime_ready": calvin_probe["runtime_ready"],
            "smoke_imports": calvin_probe["smoke_imports"],
            "smoke_status": calvin_probe["smoke_status"],
            "activation_script_path": str(activation_script_path),
            "activation_powershell_script_path": str(activation_powershell_script_path),
            "suggested_source_command": suggested_activation_command,
            "diagnosis": calvin_probe["diagnosis"],
        },
    }
    log_dir = ensure_dir(config["paths"]["log_dir"])
    out_path = log_dir / "benchmark_backends_status.json"
    write_json(out_path, payload)
    print(f"saved_status={out_path}")
    print(payload)


if __name__ == "__main__":
    main()
