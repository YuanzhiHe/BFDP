from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.rlbench_adapter import collect_low_dim_rollouts
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def main() -> None:
    config = load_config(ROOT / "config" / "default.yaml").data
    rlbench_cfg = config["benchmarks"]["rlbench"]
    adapter_cfg = rlbench_cfg["adapter"]
    log_dir = ensure_dir(config["paths"]["log_dir"])
    export_path = log_dir / "rlbench_low_dim_rollouts_sample.json"
    summary_path = log_dir / "rlbench_low_dim_rollouts_summary.json"
    summary = collect_low_dim_rollouts(rlbench_cfg, adapter_cfg, export_path)
    write_json(summary_path, summary.to_dict())
    print(f"saved_rollouts={export_path}")
    print(f"saved_summary={summary_path}")
    print(summary.to_dict())


if __name__ == "__main__":
    main()
