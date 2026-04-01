from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svh_dp.benchmarks.calvin_adapter import discover_calvin_dataset_candidates
from svh_dp.config import load_config
from svh_dp.utils.common import ensure_dir, write_json


def _resolve_path(raw_path: str | None, default_path: Path) -> Path:
    if raw_path is None:
        path = default_path
    else:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return path.resolve()


def _validate_inside_project(path: Path) -> None:
    if PROJECT_ROOT != path and PROJECT_ROOT not in path.parents:
        raise ValueError(f"path must remain inside the project: {path}")


def _count_members(archive: zipfile.ZipFile) -> tuple[int, int]:
    file_count = 0
    directory_count = 0
    for info in archive.infolist():
        if info.is_dir():
            directory_count += 1
        else:
            file_count += 1
    return file_count, directory_count


def _top_level_entries(archive: zipfile.ZipFile) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/").strip("/")
        if not normalized:
            continue
        top_level = normalized.split("/", 1)[0]
        if top_level in seen:
            continue
        seen.add(top_level)
        entries.append(top_level)
    return sorted(entries)


def _extract_archive(archive_path: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite and any(destination.iterdir()):
            raise RuntimeError(
                "destination already contains files; rerun with --overwrite to replace it"
            )
        if overwrite:
            shutil.rmtree(destination)
    ensure_dir(destination)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)


def _search_roots(dataset_cfg: dict, extracted_root: Path) -> list[Path]:
    roots = [extracted_root]
    for raw_path in dataset_cfg.get("search_roots", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--archive",
        default=str(PROJECT_ROOT / "Experiment" / "datasets" / "task_D_D.zip"),
        help="Path to the CALVIN dataset zip archive.",
    )
    parser.add_argument(
        "--extract-to",
        default=str(PROJECT_ROOT / "Experiment" / "datasets" / "calvin_real"),
        help="Destination directory for archive extraction.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the extraction directory before extracting.",
    )
    args = parser.parse_args()

    config = load_config(args.config).data
    dataset_cfg = config["benchmarks"]["calvin"].get("dataset", {})
    archive_path = _resolve_path(
        args.archive,
        PROJECT_ROOT / "Experiment" / "datasets" / "task_D_D.zip",
    )
    extract_to = _resolve_path(
        args.extract_to,
        PROJECT_ROOT / "Experiment" / "datasets" / "calvin_real",
    )
    output_path = _resolve_path(
        args.output,
        ROOT / "logs" / "calvin_dataset_prepare.json",
    )

    _validate_inside_project(archive_path)
    _validate_inside_project(extract_to)
    _validate_inside_project(output_path)

    payload = {
        "archive_path": str(archive_path),
        "archive_exists": archive_path.exists(),
        "archive_size_bytes": None if not archive_path.exists() else archive_path.stat().st_size,
        "archive_integrity_ok": False,
        "archive_member_count": 0,
        "archive_directory_count": 0,
        "archive_top_level_entries": [],
        "archive_bad_member": None,
        "extract_to": str(extract_to),
        "search_roots": [],
        "candidate_count": 0,
        "real_candidate_count": 0,
        "preferred_candidate": None,
        "preferred_real_candidate": None,
        "candidates": [],
        "ready_for_strict_export": False,
        "extracted": False,
        "diagnosis": "",
    }

    if not archive_path.exists():
        payload["diagnosis"] = (
            "CALVIN archive does not exist yet. Place a valid project-local archive under "
            "Experiment/datasets and rerun."
        )
    else:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                bad_member = archive.testzip()
                file_count, directory_count = _count_members(archive)
                top_level_entries = _top_level_entries(archive)
                payload["archive_member_count"] = file_count
                payload["archive_directory_count"] = directory_count
                payload["archive_top_level_entries"] = top_level_entries
                payload["archive_bad_member"] = bad_member
                if bad_member is not None:
                    raise RuntimeError(
                        f"CALVIN archive integrity check failed at member: {bad_member}"
                    )
                payload["archive_integrity_ok"] = True

            _extract_archive(
                archive_path=archive_path,
                destination=extract_to,
                overwrite=args.overwrite,
            )
            payload["extracted"] = True

            search_roots = _search_roots(dataset_cfg=dataset_cfg, extracted_root=extract_to)
            candidates = discover_calvin_dataset_candidates(
                search_roots=search_roots,
                max_depth=dataset_cfg.get("search_max_depth", 4),
            )
            real_candidates = [candidate for candidate in candidates if not candidate.mock_like]
            payload["search_roots"] = [str(path) for path in search_roots]
            payload["candidate_count"] = len(candidates)
            payload["real_candidate_count"] = len(real_candidates)
            payload["preferred_candidate"] = (
                None if not candidates else candidates[0].to_dict()
            )
            payload["preferred_real_candidate"] = (
                None if not real_candidates else real_candidates[0].to_dict()
            )
            payload["candidates"] = [candidate.to_dict() for candidate in candidates]
            payload["ready_for_strict_export"] = bool(real_candidates)
            payload["diagnosis"] = (
                "A non-mock CALVIN dataset candidate is available after archive extraction."
                if real_candidates
                else "Archive extraction completed, but no non-mock CALVIN dataset candidate was discovered yet."
            )
        except zipfile.BadZipFile as exc:
            payload["diagnosis"] = f"CALVIN archive is not a readable zip file: {exc}"
        except RuntimeError as exc:
            payload["diagnosis"] = str(exc)

    write_json(output_path, payload)
    print(f"saved_prepare_summary={output_path}")
    print(payload)


if __name__ == "__main__":
    main()
