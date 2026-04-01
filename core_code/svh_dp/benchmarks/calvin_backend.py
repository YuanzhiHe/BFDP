from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROBE_SNIPPET = r"""
import importlib
import importlib.util
import json
import sys

module_names = sys.argv[1:]
status = []
for module_name in module_names:
    spec = None
    spec_error = None
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        spec_error = f"{type(exc).__name__}: {exc}"
    import_error = None
    available = False
    origin = None if spec is None else spec.origin
    try:
        module = importlib.import_module(module_name)
        available = True
        origin = getattr(module, "__file__", origin)
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    status.append(
        {
            "module_name": module_name,
            "available": available,
            "origin": origin,
            "spec_available": spec is not None,
            "spec_error": spec_error,
            "import_error": import_error,
        }
    )
print(json.dumps({"module_status": status}))
"""


def _prefix_pythonpath(
    pythonpath_entries: list[str],
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if not pythonpath_entries:
        return env
    current = env.get("PYTHONPATH", "")
    joined = os.pathsep.join(pythonpath_entries)
    env["PYTHONPATH"] = joined if not current else f"{joined}{os.pathsep}{current}"
    return env


def _probe_module_status(
    module_names: list[str],
    pythonpath_entries: list[str] | None = None,
) -> list[dict]:
    env = _prefix_pythonpath(pythonpath_entries, base_env=os.environ)
    completed = subprocess.run(
        [sys.executable, "-c", PROBE_SNIPPET, *module_names],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"calvin probe failed: {error}")
    payload = None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = json.loads(stripped)
            break
    if payload is None:
        raise RuntimeError(
            f"calvin probe completed without JSON payload. stdout={completed.stdout!r}"
        )
    return payload["module_status"]


def _top_level_packages(module_names: list[str]) -> list[str]:
    packages = []
    seen = set()
    for module_name in module_names:
        package = module_name.split(".", 1)[0]
        if package not in seen:
            seen.add(package)
            packages.append(package)
    return packages


def _resolve_search_roots(search_roots: list[str], project_root: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    resolved = []
    seen = set()
    for candidate in search_roots:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = root / candidate_path
        candidate_path = candidate_path.resolve()
        candidate_key = str(candidate_path)
        if (root == candidate_path or root in candidate_path.parents) and candidate_key not in seen:
            seen.add(candidate_key)
            resolved.append(candidate_path)
    return resolved


def _discover_workspace_candidates(
    search_roots: list[Path],
    module_names: list[str],
    max_depth: int,
) -> list[Path]:
    top_level_packages = _top_level_packages(module_names)
    candidates = []
    seen = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        to_check = [search_root]
        for path in search_root.rglob("*"):
            if not path.is_dir():
                continue
            try:
                depth = len(path.relative_to(search_root).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            to_check.append(path)
        for candidate in to_check:
            if ".git" in candidate.parts:
                continue
            if any((candidate / module_name).exists() for module_name in top_level_packages):
                candidate_key = str(candidate)
                if candidate_key not in seen:
                    seen.add(candidate_key)
                    candidates.append(candidate)
    return candidates


def _build_runtime_pythonpath_entries(
    candidate_root: str | Path | None,
    module_names: list[str],
) -> list[str]:
    if candidate_root is None:
        return []
    root = Path(candidate_root).resolve()
    entries = []
    seen = set()

    def add_entry(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            entries.append(key)

    add_entry(root)
    top_level_packages = _top_level_packages(module_names)
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if any((child / module_name).exists() for module_name in top_level_packages):
                add_entry(child.resolve())
    return entries


_MISSING_MODULE_PATTERN = re.compile(r"No module named '([^']+)'")


def _extract_dependency_blockers(module_status: list[dict]) -> list[str]:
    blockers = []
    seen = set()
    for item in module_status:
        if item.get("available"):
            continue
        if not item.get("spec_available"):
            continue
        import_error = item.get("import_error")
        if not import_error:
            continue
        match = _MISSING_MODULE_PATTERN.search(import_error)
        if not match:
            continue
        blocker = match.group(1)
        if blocker not in seen:
            seen.add(blocker)
            blockers.append(blocker)
    return blockers


def build_calvin_probe(config: dict | None = None) -> dict:
    probe_cfg = {} if config is None else config.get("probe", {})
    module_names = probe_cfg.get(
        "module_names",
        ["calvin_env.envs.play_table_env", "calvin_agent"],
    )
    smoke_imports = probe_cfg.get("smoke_imports", ["calvin_agent.evaluation.utils"])
    project_root = probe_cfg.get("project_root", Path.cwd())
    search_roots = _resolve_search_roots(
        probe_cfg.get("search_roots", []),
        project_root=project_root,
    )
    current_status = _probe_module_status(module_names)
    available = all(item["available"] for item in current_status)
    current_smoke_status = (
        _probe_module_status(smoke_imports) if available and smoke_imports else []
    )
    current_runtime_ready = available and all(
        item["available"] for item in current_smoke_status
    )

    workspace_candidates = []
    bootstrap_candidate = None
    if not available:
        candidates = _discover_workspace_candidates(
            search_roots=search_roots,
            module_names=module_names,
            max_depth=probe_cfg.get("search_max_depth", 4),
        )
        for candidate in candidates:
            candidate_pythonpath_entries = _build_runtime_pythonpath_entries(
                candidate,
                module_names,
            )
            candidate_status = _probe_module_status(
                module_names,
                pythonpath_entries=candidate_pythonpath_entries,
            )
            candidate_source_layout_available = all(
                item["spec_available"] for item in candidate_status
            )
            candidate_smoke_status = (
                _probe_module_status(
                    smoke_imports,
                    pythonpath_entries=candidate_pythonpath_entries,
                )
                if smoke_imports and candidate_source_layout_available
                else []
            )
            candidate_payload = {
                "root": str(candidate),
                "pythonpath_entries": candidate_pythonpath_entries,
                "module_status": candidate_status,
                "source_layout_available": candidate_source_layout_available,
                "all_modules_available": all(
                    item["available"] for item in candidate_status
                ),
                "dependency_blockers": _extract_dependency_blockers(candidate_status),
                "smoke_status": candidate_smoke_status,
                "smoke_dependency_blockers": _extract_dependency_blockers(
                    candidate_smoke_status
                ),
                "runtime_ready": all(
                    item["available"] for item in candidate_status
                )
                and all(item["available"] for item in candidate_smoke_status),
            }
            workspace_candidates.append(candidate_payload)
            if bootstrap_candidate is None and candidate_payload["source_layout_available"]:
                bootstrap_candidate = candidate_payload

    active_pythonpath_entries = []
    smoke_status = current_smoke_status
    runtime_ready = current_runtime_ready
    dependency_blockers = _extract_dependency_blockers(current_status)
    smoke_dependency_blockers = _extract_dependency_blockers(current_smoke_status)
    source_layout_available = available
    if bootstrap_candidate is not None:
        active_pythonpath_entries = bootstrap_candidate["pythonpath_entries"]
        smoke_status = bootstrap_candidate["smoke_status"]
        runtime_ready = bootstrap_candidate["runtime_ready"]
        dependency_blockers = bootstrap_candidate["dependency_blockers"]
        smoke_dependency_blockers = bootstrap_candidate["smoke_dependency_blockers"]
        source_layout_available = bootstrap_candidate["source_layout_available"]

    if available and runtime_ready:
        message = "CALVIN modules are importable on the current Python path"
        diagnosis = "CALVIN backend import checks pass in the current environment."
    elif available and not runtime_ready:
        message = (
            "CALVIN base modules are importable on the current Python path, but runtime "
            "smoke imports still fail."
        )
        diagnosis = (
            "CALVIN code is visible to the current environment, but at least one runtime "
            "dependency needed by the benchmark stack is still missing."
        )
    elif bootstrap_candidate is not None:
        if runtime_ready:
            message = (
                "CALVIN modules are not importable on the current Python path but become "
                "available when PYTHONPATH is prefixed with the discovered workspace entries."
            )
            diagnosis = (
                "CALVIN is locally bootstrappable from a project-resident candidate root, "
                "but the current runtime has not been configured to use it."
            )
        else:
            blocker_text = (
                " Missing dependencies detected during import: "
                + ", ".join(
                    dependency_blockers
                    + [x for x in smoke_dependency_blockers if x not in dependency_blockers]
                )
                + "."
                if dependency_blockers or smoke_dependency_blockers
                else ""
            )
            message = (
                "CALVIN source layout is locally bootstrappable, but runtime imports "
                f"still fail after PYTHONPATH is configured.{blocker_text}"
            )
            diagnosis = (
                "The project-local CALVIN code drop is structurally usable, but the current "
                "Python environment is missing at least one runtime dependency required by "
                "CALVIN imports."
            )
    elif workspace_candidates:
        message = (
            "CALVIN-related directories were found inside the project, but the expected "
            "modules still do not become importable when probed through PYTHONPATH."
        )
        diagnosis = (
            "Project-local CALVIN candidate roots exist, but their package layout is not "
            "sufficient for the expected runtime imports. This usually indicates an "
            "incomplete code drop or missing submodule content."
        )
    else:
        message = "calvin_env is not installed in the current environment"
        diagnosis = (
            "CALVIN backend remains blocked because the expected modules are not "
            "importable and no project-local candidate CALVIN roots were found."
        )

    return {
        "available": available,
        "bootstrap_available": bootstrap_candidate is not None,
        "source_layout_available": source_layout_available,
        "message": message,
        "diagnosis": diagnosis,
        "module_status": current_status,
        "module_probe": {
            item["module_name"]: item["available"] for item in current_status
        },
        "search_roots": [str(path) for path in search_roots],
        "workspace_candidates": workspace_candidates,
        "bootstrap_root": None if bootstrap_candidate is None else bootstrap_candidate["root"],
        "pythonpath_entries": active_pythonpath_entries,
        "dependency_blockers": dependency_blockers,
        "smoke_dependency_blockers": smoke_dependency_blockers,
        "smoke_imports": smoke_imports,
        "smoke_status": smoke_status,
        "runtime_ready": runtime_ready,
    }


def calvin_available(config: dict | None = None) -> tuple[bool, str]:
    probe = build_calvin_probe(config)
    return probe["available"], probe["message"]


def build_calvin_runtime_env(
    config: dict | None = None,
    base_env: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict]:
    probe = build_calvin_probe(config)
    pythonpath_entries = probe.get("pythonpath_entries", [])
    env = _prefix_pythonpath(pythonpath_entries, base_env=base_env)
    return env, {
        "available": probe["available"],
        "bootstrap_available": probe["bootstrap_available"],
        "bootstrap_root": probe["bootstrap_root"],
        "pythonpath_entries": pythonpath_entries,
        "source_layout_available": probe["source_layout_available"],
        "dependency_blockers": probe["dependency_blockers"],
        "smoke_dependency_blockers": probe["smoke_dependency_blockers"],
        "message": probe["message"],
        "diagnosis": probe["diagnosis"],
        "runtime_ready": probe["runtime_ready"],
        "smoke_status": probe["smoke_status"],
    }
