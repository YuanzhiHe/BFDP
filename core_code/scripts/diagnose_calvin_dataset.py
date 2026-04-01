from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_adapter import discover_calvin_dataset_candidates
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
        "--output",
        help="Optional output path for dataset diagnosis JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    calvin_cfg = config["benchmarks"]["calvin"]
    dataset_cfg = calvin_cfg.get("dataset", {})
    search_roots = []
    for raw_path in dataset_cfg.get("search_roots", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        search_roots.append(path.resolve())

    candidates = discover_calvin_dataset_candidates(
        search_roots=search_roots,
        max_depth=dataset_cfg.get("search_max_depth", 4),
    )
    real_candidates = [candidate for candidate in candidates if not candidate.mock_like]
    preferred_candidate = None if not candidates else candidates[0]
    preferred_real_candidate = None if not real_candidates else real_candidates[0]

    payload = {
        "search_roots": [str(path) for path in search_roots],
        "candidate_count": len(candidates),
        "real_candidate_count": len(real_candidates),
        "mock_only": bool(candidates) and not real_candidates,
        "real_dataset_ready": bool(real_candidates),
        "preferred_candidate": (
            None if preferred_candidate is None else preferred_candidate.to_dict()
        ),
        "preferred_real_candidate": (
            None
            if preferred_real_candidate is None
            else preferred_real_candidate.to_dict()
        ),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "diagnosis": (
            "A non-mock CALVIN dataset is available for benchmark-facing export."
            if real_candidates
            else (
                "Only mock CALVIN-style datasets are currently available under the project dataset roots."
                if candidates
                else "No CALVIN-style dataset candidates were found under the configured project dataset roots."
            )
        ),
    }

    log_dir = ensure_dir(config["paths"]["log_dir"])
    output_path = Path(args.output) if args.output else log_dir / "calvin_dataset_diagnostic.json"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    write_json(output_path, payload)
    print(f"saved_diagnostic={output_path}")
    print(payload)


if __name__ == "__main__":
    main()
