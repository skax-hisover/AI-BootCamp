"""I/O helpers for safe file persistence."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text by temp-file + replace.

    This reduces partial-write risk under concurrent or interrupted writes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(target)

