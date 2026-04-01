from __future__ import annotations

import json
import random
from pathlib import Path
import tempfile
from typing import Any

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        temp_path = Path(handle.name)
    try:
        temp_path.replace(target)
    except PermissionError:
        # Some Windows setups keep transient handles on the destination path long
        # enough to make atomic replace flaky. Fall back to a direct write so the
        # diagnostics can still be regenerated.
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if temp_path.exists():
            temp_path.unlink()
