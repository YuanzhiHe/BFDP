from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from collections import deque
from typing import Any

import numpy as np


@dataclass
class CALVINAdapterSummary:
    sequences: int
    total_steps: int
    mean_instruction_length: float
    obs_dim: int | None
    action_dim: int | None
    dataset_path: str
    split: str
    lang_folder: str
    action_key: str
    horizon: int
    export_mode: str
    output_path: str
    task_names: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "total_steps": self.total_steps,
            "mean_instruction_length": self.mean_instruction_length,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "dataset_path": self.dataset_path,
            "split": self.split,
            "lang_folder": self.lang_folder,
            "action_key": self.action_key,
            "horizon": self.horizon,
            "export_mode": self.export_mode,
            "output_path": self.output_path,
            "task_names": self.task_names,
        }


@dataclass
class CALVINDatasetCandidate:
    root: str
    training_ready: bool
    validation_ready: bool
    lang_annotation_splits: list[str]
    mock_like: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "training_ready": self.training_ready,
            "validation_ready": self.validation_ready,
            "lang_annotation_splits": self.lang_annotation_splits,
            "mock_like": self.mock_like,
        }


def _split_ready(dataset_root: Path, split: str) -> tuple[bool, bool]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        return False, False
    has_episode_index = (split_dir / "ep_start_end_ids.npy").exists()
    has_lang = any(
        candidate.exists()
        for candidate in [
            split_dir / "lang_annotations" / "auto_lang_ann.npy",
            split_dir / "auto_lang_ann.npy",
        ]
    )
    return has_episode_index, has_lang


def _looks_like_calvin_root(dataset_root: Path) -> bool:
    return (dataset_root / "training").is_dir() or (dataset_root / "validation").is_dir()


def _looks_mock_dataset(dataset_root: Path) -> bool:
    lowered = str(dataset_root).lower()
    return "mock" in lowered or "debug" in lowered


def discover_calvin_dataset_candidates(
    search_roots: list[str | Path],
    max_depth: int = 4,
) -> list[CALVINDatasetCandidate]:
    candidates: list[CALVINDatasetCandidate] = []
    seen: set[str] = set()
    for raw_root in search_roots:
        search_root = Path(raw_root).resolve()
        if not search_root.exists():
            continue
        paths_to_check: list[Path] = []
        queue: deque[tuple[Path, int]] = deque([(search_root, 0)])
        while queue:
            current_root, depth = queue.popleft()
            paths_to_check.append(current_root)
            if _looks_like_calvin_root(current_root):
                continue
            if depth >= max_depth:
                continue
            try:
                children = sorted(
                    path for path in current_root.iterdir() if path.is_dir()
                )
            except (PermissionError, FileNotFoundError):
                continue
            for child in children:
                queue.append((child, depth + 1))
        for candidate_root in paths_to_check:
            candidate_key = str(candidate_root.resolve())
            if candidate_key in seen:
                continue
            training_ready, training_has_lang = _split_ready(candidate_root, "training")
            validation_ready, validation_has_lang = _split_ready(candidate_root, "validation")
            if not (
                (training_ready and training_has_lang)
                or (validation_ready and validation_has_lang)
            ):
                continue
            lang_annotation_splits = []
            if training_has_lang:
                lang_annotation_splits.append("training")
            if validation_has_lang:
                lang_annotation_splits.append("validation")
            seen.add(candidate_key)
            candidates.append(
                CALVINDatasetCandidate(
                    root=candidate_key,
                    training_ready=training_ready,
                    validation_ready=validation_ready,
                    lang_annotation_splits=lang_annotation_splits,
                    mock_like=_looks_mock_dataset(candidate_root),
                )
            )
    return sorted(
        candidates,
        key=lambda item: (
            item.mock_like,
            not item.validation_ready,
            not item.training_ready,
            item.root,
        ),
    )


def resolve_calvin_dataset_path(
    dataset_path: str | Path | None,
    search_roots: list[str | Path] | None = None,
    max_depth: int = 4,
    require_non_mock: bool = False,
) -> tuple[Path, list[CALVINDatasetCandidate]]:
    if dataset_path:
        resolved = Path(dataset_path).resolve()
        if require_non_mock and _looks_mock_dataset(resolved):
            raise FileNotFoundError(
                f"resolved CALVIN dataset path is mock-like but a non-mock dataset is required: {resolved}"
            )
        return resolved, []
    candidates = discover_calvin_dataset_candidates(
        search_roots=search_roots or [],
        max_depth=max_depth,
    )
    if require_non_mock:
        candidates = [candidate for candidate in candidates if not candidate.mock_like]
    if not candidates:
        raise FileNotFoundError(
            "could not find a CALVIN dataset root under the configured project-local search roots"
            if not require_non_mock
            else "could not find a non-mock CALVIN dataset root under the configured project-local search roots"
        )
    return Path(candidates[0].root), candidates


def _load_lang_payload(split_dir: Path, lang_folder: str) -> dict[str, Any]:
    candidate_paths = [
        split_dir / lang_folder / "auto_lang_ann.npy",
        split_dir / "auto_lang_ann.npy",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return np.load(candidate_path, allow_pickle=True).item()
    raise FileNotFoundError(
        f"could not find CALVIN language annotations under {split_dir}"
    )


def _lookup_naming_pattern(dataset_dir: Path, suffix: str = ".npz") -> tuple[str, int]:
    for path in sorted(dataset_dir.glob(f"*{suffix}")):
        match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
        if match:
            prefix = path.name[: match.start()]
            return prefix, len(match.group(1))
    raise FileNotFoundError(f"could not find any {suffix} episode files in {dataset_dir}")


def _episode_path(dataset_dir: Path, prefix: str, n_digits: int, frame_index: int) -> Path:
    return dataset_dir / f"{prefix}{frame_index:0{n_digits}d}.npz"


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _coerce_text(value.item())
        if value.size == 1:
            return _coerce_text(value.reshape(-1)[0])
        return " ".join(_coerce_text(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _coerce_text(value[0])
        return " ".join(_coerce_text(item) for item in value)
    return str(value)


def _load_step(path: Path, action_key: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        step = {
            "frame_index": int(re.search(r"(\d+)(?=\.[^.]+$)", path.name).group(1)),
            "robot_obs": np.asarray(payload["robot_obs"], dtype=np.float32).tolist(),
            "scene_obs": np.asarray(payload["scene_obs"], dtype=np.float32).tolist(),
            "action": np.asarray(payload[action_key], dtype=np.float32).tolist(),
        }
    return step


def summarize_calvin_export(payload: dict[str, Any], output_path: str | Path) -> CALVINAdapterSummary:
    sequences = payload["episodes"]
    total_steps = sum(len(sequence["steps"]) for sequence in sequences)
    instruction_lengths = [len(sequence["instruction"].split()) for sequence in sequences]
    obs_dim = None
    action_dim = None
    if sequences and sequences[0]["steps"]:
        obs_dim = len(sequences[0]["steps"][0]["robot_obs"])
        action_dim = len(sequences[0]["steps"][0]["action"])
    task_names = []
    for sequence in sequences:
        task_name = sequence.get("task_name")
        if task_name and task_name not in task_names:
            task_names.append(task_name)
    return CALVINAdapterSummary(
        sequences=len(sequences),
        total_steps=total_steps,
        mean_instruction_length=(
            sum(instruction_lengths) / max(1, len(instruction_lengths))
        ),
        obs_dim=obs_dim,
        action_dim=action_dim,
        dataset_path=payload["dataset_path"],
        split=payload["split"],
        lang_folder=payload["lang_folder"],
        action_key=payload["action_key"],
        horizon=payload["horizon"],
        export_mode=payload.get("export_mode", "prefix"),
        output_path=str(output_path),
        task_names=task_names or None,
    )


def _resolve_export_window(
    start_idx: int,
    end_idx: int,
    horizon: int,
    export_mode: str,
) -> tuple[int, int]:
    if end_idx < start_idx:
        raise ValueError(
            f"invalid CALVIN sequence bounds: start_idx={start_idx}, end_idx={end_idx}"
        )
    sequence_len = end_idx - start_idx + 1
    if export_mode == "full":
        return start_idx, end_idx
    effective_horizon = max(1, min(horizon, sequence_len))
    if export_mode == "prefix":
        return start_idx, start_idx + effective_horizon - 1
    if export_mode == "tail":
        return end_idx - effective_horizon + 1, end_idx
    raise ValueError(
        f"unsupported CALVIN export_mode={export_mode!r}; expected one of prefix, tail, full"
    )


def export_calvin_language_rollouts(
    dataset_path: str | Path,
    output_path: str | Path,
    split: str = "validation",
    lang_folder: str = "lang_annotations",
    max_sequences: int = 2,
    horizon: int = 4,
    export_mode: str = "prefix",
    action_key: str = "rel_actions",
    include_tasks: list[str] | None = None,
) -> CALVINAdapterSummary:
    dataset_path = Path(dataset_path).resolve()
    split_dir = dataset_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"CALVIN split directory does not exist: {split_dir}")

    lang_payload = _load_lang_payload(split_dir, lang_folder=lang_folder)
    prefix, n_digits = _lookup_naming_pattern(split_dir)
    annotations = lang_payload["language"]["ann"]
    tasks = lang_payload["language"].get("task")
    indices = lang_payload["info"]["indx"]
    normalized_task_filter = None
    if include_tasks:
        normalized_task_filter = {task.strip() for task in include_tasks if task.strip()}
    normalized_export_mode = export_mode.strip().lower()

    episodes = []
    selected_sequence_indices: list[int] = []
    for sequence_index, _ in enumerate(indices):
        task_name = None if tasks is None else _coerce_text(tasks[sequence_index])
        if normalized_task_filter and task_name not in normalized_task_filter:
            continue
        selected_sequence_indices.append(sequence_index)
        if len(selected_sequence_indices) >= max_sequences:
            break

    for export_index, sequence_index in enumerate(selected_sequence_indices):
        start_idx, end_idx = indices[sequence_index]
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        effective_start_idx, effective_end_idx = _resolve_export_window(
            start_idx=start_idx,
            end_idx=end_idx,
            horizon=horizon,
            export_mode=normalized_export_mode,
        )
        steps = [
            _load_step(
                _episode_path(split_dir, prefix, n_digits, frame_index),
                action_key=action_key,
            )
            for frame_index in range(effective_start_idx, effective_end_idx + 1)
        ]
        episodes.append(
            {
                "sequence_index": export_index,
                "source_sequence_index": sequence_index,
                "instruction": _coerce_text(annotations[sequence_index]),
                "task_name": None if tasks is None else _coerce_text(tasks[sequence_index]),
                "start_frame_index": start_idx,
                "end_frame_index": end_idx,
                "effective_start_frame_index": effective_start_idx,
                "effective_end_frame_index": effective_end_idx,
                "export_mode": normalized_export_mode,
                "steps": steps,
            }
        )

    payload = {
        "dataset_path": str(dataset_path),
        "split": split,
        "lang_folder": lang_folder,
        "action_key": action_key,
        "horizon": horizon,
        "export_mode": normalized_export_mode,
        "task_filter": sorted(normalized_task_filter) if normalized_task_filter else None,
        "episodes": episodes,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summarize_calvin_export(payload, output_path=output_path)


def load_calvin_export(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
