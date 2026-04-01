from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_adapter import (
    export_calvin_language_rollouts,
    resolve_calvin_dataset_path,
)
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Optional CALVIN dataset root override.",
    )
    parser.add_argument(
        "--output",
        help="Optional export JSON output path override.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional summary JSON output path override.",
    )
    parser.add_argument(
        "--require-non-mock",
        action="store_true",
        help="Require a non-mock CALVIN dataset candidate.",
    )
    parser.add_argument(
        "--sample-sequences",
        type=int,
        help="Optional override for the number of CALVIN sequences to export.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        help="Optional override for the exported rollout horizon.",
    )
    parser.add_argument(
        "--export-mode",
        choices=["prefix", "tail", "full"],
        help="Optional export window mode: keep the first horizon steps, the last horizon steps, or the full episode.",
    )
    parser.add_argument(
        "--split",
        help="Optional override for the CALVIN split to export.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task-name filter for the exported CALVIN sequences.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    calvin_cfg = config["benchmarks"]["calvin"]
    adapter_cfg = calvin_cfg["adapter"]
    dataset_cfg = calvin_cfg.get("dataset", {})
    search_roots = []
    for raw_path in dataset_cfg.get("search_roots", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        search_roots.append(path.resolve())
    dataset_path, discovered_candidates = resolve_calvin_dataset_path(
        dataset_path=args.dataset_path or adapter_cfg["dataset_path"],
        search_roots=search_roots,
        max_depth=dataset_cfg.get("search_max_depth", 4),
        require_non_mock=(
            args.require_non_mock
            or dataset_cfg.get("require_non_mock_for_default_export", False)
        ),
    )

    log_dir = ensure_dir(config["paths"]["log_dir"])
    export_path = Path(args.output) if args.output else log_dir / "calvin_adapter_sample.json"
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else log_dir / "calvin_adapter_summary.json"
    )
    summary = export_calvin_language_rollouts(
        dataset_path=dataset_path,
        output_path=export_path,
        split=args.split or adapter_cfg["split"],
        lang_folder=adapter_cfg["lang_folder"],
        max_sequences=args.sample_sequences or adapter_cfg["sample_sequences"],
        horizon=args.rollout_horizon or adapter_cfg["rollout_horizon"],
        export_mode=args.export_mode or adapter_cfg.get("export_mode", "prefix"),
        action_key=adapter_cfg["action_key"],
        include_tasks=args.tasks,
    )
    summary_payload = summary.to_dict()
    summary_payload["resolved_dataset_path"] = str(dataset_path)
    summary_payload["dataset_candidates"] = [
        candidate.to_dict() for candidate in discovered_candidates
    ]
    write_json(summary_path, summary_payload)
    print(f"saved_rollouts={export_path}")
    print(f"saved_summary={summary_path}")
    print(summary_payload)


if __name__ == "__main__":
    main()
