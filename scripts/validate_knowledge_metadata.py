"""Validate knowledge metadata sidecar files for operational quality checks.

Usage:
  python scripts/validate_knowledge_metadata.py
  python scripts/validate_knowledge_metadata.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
REQUIRED_FIELDS = ("collected_at", "source_url", "curator", "license")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate knowledge metadata sidecars")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any warning is found",
    )
    return parser.parse_args()


def _iter_knowledge_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def _meta_path_for(doc_path: Path) -> Path:
    # Example: sample.md -> sample.md.meta.json
    return doc_path.with_name(doc_path.name + ".meta.json")


def _is_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def validate() -> tuple[list[str], int]:
    warnings: list[str] = []
    checked = 0
    for doc in _iter_knowledge_files(KNOWLEDGE_DIR):
        checked += 1
        meta_path = _meta_path_for(doc)
        rel_doc = doc.relative_to(PROJECT_ROOT).as_posix()
        rel_meta = meta_path.relative_to(PROJECT_ROOT).as_posix()

        if not meta_path.exists():
            warnings.append(f"[WARN] Missing metadata sidecar: {rel_meta} (for {rel_doc})")
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"[WARN] Invalid JSON in {rel_meta}: {exc}")
            continue

        if not isinstance(meta, dict):
            warnings.append(f"[WARN] Metadata must be JSON object: {rel_meta}")
            continue

        for field in REQUIRED_FIELDS:
            value = str(meta.get(field, "")).strip()
            if not value:
                warnings.append(f"[WARN] Missing required field '{field}' in {rel_meta}")

        collected_at = str(meta.get("collected_at", "")).strip()
        if collected_at and not _is_iso_date(collected_at):
            warnings.append(
                f"[WARN] collected_at should follow YYYY-MM-DD in {rel_meta}: {collected_at}"
            )

        source_url = str(meta.get("source_url", "")).strip()
        if source_url and not (
            source_url.startswith("http://")
            or source_url.startswith("https://")
            or source_url.startswith("internal://")
        ):
            warnings.append(
                f"[WARN] source_url should start with http(s):// or internal:// in {rel_meta}"
            )

    return warnings, checked


def main() -> None:
    args = parse_args()
    warnings, checked = validate()
    print(f"Checked documents: {checked}")
    if warnings:
        for warning in warnings:
            print(warning)
        if args.strict:
            raise SystemExit(1)
    else:
        print("[OK] All knowledge metadata sidecars are valid.")


if __name__ == "__main__":
    main()
