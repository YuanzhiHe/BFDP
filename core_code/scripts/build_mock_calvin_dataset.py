from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.utils.common import ensure_dir, write_json


def _robot_obs(frame_index: int) -> np.ndarray:
    base = np.linspace(0.0, 1.4, 15, dtype=np.float32)
    return base + frame_index * 0.05


def _scene_obs(frame_index: int) -> np.ndarray:
    base = np.linspace(-0.5, 0.65, 24, dtype=np.float32)
    return base + frame_index * 0.02


def _rel_action(frame_index: int) -> np.ndarray:
    action = np.array(
        [
            0.1 + frame_index * 0.01,
            -0.05 + frame_index * 0.01,
            0.02,
            0.0,
            0.01,
            -0.02,
            1.0 if frame_index % 2 == 0 else -1.0,
        ],
        dtype=np.float32,
    )
    return action


def _write_episode(path: Path, frame_index: int) -> None:
    np.savez(
        path,
        robot_obs=_robot_obs(frame_index),
        scene_obs=_scene_obs(frame_index),
        rel_actions=_rel_action(frame_index),
        actions=_rel_action(frame_index),
    )


def _write_lang_annotations(split_dir: Path, frame_ranges: list[tuple[int, int]], instructions: list[str], tasks: list[str]) -> None:
    lang_dir = ensure_dir(split_dir / "lang_annotations")
    embeddings = np.stack(
        [
            np.linspace(0.0, 1.0, 8, dtype=np.float32) + idx * 0.1
            for idx in range(len(instructions))
        ],
        axis=0,
    )
    payload = {
        "language": {
            "ann": np.array(instructions, dtype=object),
            "task": np.array(tasks, dtype=object),
            "emb": embeddings,
        },
        "info": {
            "indx": np.array(frame_ranges, dtype=np.int64),
        },
    }
    np.save(lang_dir / "auto_lang_ann.npy", payload, allow_pickle=True)


def _write_split(root: Path, split: str) -> dict[str, str]:
    split_dir = ensure_dir(root / split)
    frame_ranges = [(0, 3), (4, 7)]
    instructions = [
        "open the drawer",
        "push the blue block to the slider",
    ]
    tasks = [
        "open_drawer",
        "push_blue_block_slider",
    ]
    for frame_index in range(8):
        _write_episode(split_dir / f"episode_{frame_index:07d}.npz", frame_index)
    _write_lang_annotations(split_dir, frame_ranges, instructions, tasks)
    np.save(split_dir / "ep_start_end_ids.npy", np.array(frame_ranges, dtype=np.int64))
    np.save(split_dir / "ep_lens.npy", np.array([4, 4], dtype=np.int64))
    return {
        "split_dir": str(split_dir),
        "instructions": instructions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(ROOT.parents[1] / "Experiment" / "datasets" / "calvin_mock_debug" / "task_D_D"),
        help="Output root for the mock CALVIN dataset.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    training = _write_split(output_root, "training")
    validation = _write_split(output_root, "validation")
    summary = {
        "dataset_root": str(output_root),
        "training_split": training["split_dir"],
        "validation_split": validation["split_dir"],
        "num_sequences_per_split": 2,
        "frames_per_split": 8,
        "note": "Project-local mock CALVIN-style dataset for adapter smoke validation only.",
    }
    summary_path = output_root / "mock_dataset_summary.json"
    write_json(summary_path, summary)
    print(f"saved_summary={summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
