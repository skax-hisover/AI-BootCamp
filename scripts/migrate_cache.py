"""One-time migration helper for legacy graph_state_cache.json -> final_answer_cache.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.io import atomic_write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy graph_state_cache.json to final_answer_cache.json")
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Index directory path containing cache files (default: data/index)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing final_answer_cache.json with merged result",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when migration is skipped/fails",
    )
    return parser.parse_args()


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def migrate_cache(index_dir: Path, force: bool = False) -> tuple[bool, str]:
    legacy_path = index_dir / "graph_state_cache.json"
    target_path = index_dir / "final_answer_cache.json"
    backup_path = index_dir / "graph_state_cache.backup.json"

    if not legacy_path.exists():
        return False, f"[SKIP] legacy cache not found: {legacy_path.as_posix()}"

    legacy = _load_json_dict(legacy_path)
    if not legacy:
        return False, f"[SKIP] legacy cache is empty/invalid: {legacy_path.as_posix()}"

    if target_path.exists() and not force:
        return (
            False,
            "[SKIP] final cache already exists. Re-run with --force to merge/overwrite.",
        )

    current = _load_json_dict(target_path) if target_path.exists() else {}
    merged = dict(current)
    migrated_sessions = 0
    for session_id, payload in legacy.items():
        if session_id not in merged:
            merged[session_id] = payload
            migrated_sessions += 1

    if force and current:
        # Keep current keys and fill missing entries from legacy by default.
        # This avoids clobbering newer final-answer payload records.
        pass

    atomic_write_text(target_path, json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_write_text(backup_path, json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    return (
        True,
        f"[OK] migrated {migrated_sessions} session entries to {target_path.as_posix()} "
        f"(legacy backup: {backup_path.as_posix()})",
    )


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    ok, message = migrate_cache(index_dir=index_dir, force=bool(args.force))
    print(message)
    if not ok and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
