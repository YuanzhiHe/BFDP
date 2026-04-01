from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_adapter import load_calvin_export, summarize_calvin_export
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_official_matches(native_payload: dict) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for sequence in native_payload.get("sequence_results", []):
        native_sequence_index = sequence.get("sequence_index")
        initial_state = sequence.get("initial_state")
        eval_sequence = sequence.get("eval_sequence", [])
        for subtask in sequence.get("subtask_results", []):
            matched_episode = subtask.get("teacher_gripper_match", {}).get("episode")
            if not matched_episode:
                continue
            matches.append(
                {
                    "native_sequence_index": native_sequence_index,
                    "native_initial_state": initial_state,
                    "native_eval_sequence": eval_sequence,
                    "subtask": subtask.get("subtask"),
                    "language_annotation": subtask.get("language_annotation"),
                    "matched_sequence_index": matched_episode.get("sequence_index"),
                    "matched_source_sequence_index": matched_episode.get("source_sequence_index"),
                    "matched_instruction": matched_episode.get("instruction"),
                    "matched_task_name": matched_episode.get("task_name"),
                }
            )
    return matches


def _build_episode_lookup(rollout_payload: dict) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for episode in rollout_payload.get("episodes", []):
        source_sequence_index = episode.get("source_sequence_index")
        if isinstance(source_sequence_index, int):
            lookup[source_sequence_index] = episode
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--rollout-path",
        required=True,
        help="Existing CALVIN export JSON used as the episode source pool.",
    )
    parser.add_argument(
        "--native-log",
        required=True,
        help="Native harness log whose teacher matches define the official-sequence supervision set.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the official-sequence-matched export.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Output JSON path for the export summary.",
    )
    parser.add_argument(
        "--dedupe-by-source-sequence",
        action="store_true",
        help="Keep only one episode per source_sequence_index while preserving first-seen order.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    rollout_payload = load_calvin_export(args.rollout_path)
    native_payload = _load_json(args.native_log)
    episode_lookup = _build_episode_lookup(rollout_payload)
    matches = _collect_official_matches(native_payload)

    selected_episodes: list[dict] = []
    selected_metadata: list[dict[str, object]] = []
    seen_source_indices: set[int] = set()
    for export_index, match in enumerate(matches):
        source_sequence_index = match["matched_source_sequence_index"]
        if not isinstance(source_sequence_index, int):
            continue
        if args.dedupe_by_source_sequence and source_sequence_index in seen_source_indices:
            continue
        source_episode = episode_lookup.get(source_sequence_index)
        if source_episode is None:
            raise KeyError(
                f"missing source_sequence_index={source_sequence_index} in rollout_path={args.rollout_path}"
            )
        episode = dict(source_episode)
        episode["sequence_index"] = len(selected_episodes)
        episode["matched_native_sequence_index"] = match["native_sequence_index"]
        episode["matched_native_initial_state"] = match["native_initial_state"]
        episode["matched_native_eval_sequence"] = match["native_eval_sequence"]
        episode["matched_subtask"] = match["subtask"]
        episode["matched_language_annotation"] = match["language_annotation"]
        selected_episodes.append(episode)
        selected_metadata.append(
            {
                "export_sequence_index": episode["sequence_index"],
                "source_sequence_index": source_sequence_index,
                "task_name": episode.get("task_name"),
                "instruction": episode.get("instruction"),
                "matched_native_sequence_index": match["native_sequence_index"],
                "matched_subtask": match["subtask"],
                "matched_language_annotation": match["language_annotation"],
            }
        )
        seen_source_indices.add(source_sequence_index)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    summary_path = Path(args.summary_output)
    ensure_dir(summary_path.parent)

    export_payload = {
        "dataset_path": rollout_payload["dataset_path"],
        "split": rollout_payload["split"],
        "lang_folder": rollout_payload["lang_folder"],
        "action_key": rollout_payload["action_key"],
        "horizon": rollout_payload["horizon"],
        "export_mode": rollout_payload.get("export_mode", "full"),
        "task_filter": sorted(
            {
                episode.get("task_name")
                for episode in selected_episodes
                if episode.get("task_name")
            }
        ),
        "selection_mode": "official_sequence_match",
        "selection_source_rollout_path": str(Path(args.rollout_path).resolve()),
        "selection_source_native_log": str(Path(args.native_log).resolve()),
        "selection_matches": selected_metadata,
        "episodes": selected_episodes,
    }
    write_json(output_path, export_payload)

    summary_payload = summarize_calvin_export(export_payload, output_path=output_path).to_dict()
    summary_payload["selection_mode"] = export_payload["selection_mode"]
    summary_payload["selection_match_count"] = len(selected_metadata)
    summary_payload["selection_source_rollout_path"] = export_payload["selection_source_rollout_path"]
    summary_payload["selection_source_native_log"] = export_payload["selection_source_native_log"]
    summary_payload["selection_matches"] = selected_metadata
    summary_payload["seed"] = config.get("seed")
    write_json(summary_path, summary_payload)

    print(f"saved_rollouts={output_path}")
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
