"""Submission safety checks for .env leakage risk.

Usage:
  python scripts/check_env_submission_safety.py
  python scripts/check_env_submission_safety.py --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check .env submission safety")
    parser.add_argument(
        "--env-path",
        default=".env",
        help="Path to env file relative to project root",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when risky conditions are detected",
    )
    return parser.parse_args()


def _load_key_values(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"", "none", "null", "changeme", "your_value_here"} or lowered.startswith("<")


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    sensitive_tokens = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return any(token in upper for token in sensitive_tokens)


def main() -> None:
    args = parse_args()
    env_path = (PROJECT_ROOT / args.env_path).resolve()
    gitignore_path = PROJECT_ROOT / ".gitignore"
    warnings: list[str] = []

    if not gitignore_path.exists():
        warnings.append("[WARN] .gitignore file is missing.")
    else:
        content = gitignore_path.read_text(encoding="utf-8", errors="ignore")
        if ".env" not in content:
            warnings.append("[WARN] .gitignore does not include '.env'.")

    if not env_path.exists():
        print("[OK] .env file does not exist in project root. Submission safety is high.")
    else:
        warnings.append(
            f"[WARN] {env_path.relative_to(PROJECT_ROOT).as_posix()} exists. "
            "Do not include it in submission package."
        )
        values = _load_key_values(env_path)
        populated_sensitive = [
            key
            for key, value in values.items()
            if _is_sensitive_key(key) and not _is_placeholder(value)
        ]
        if populated_sensitive:
            warnings.append(
                "[WARN] Sensitive env keys have non-empty values: "
                + ", ".join(sorted(populated_sensitive))
            )
        else:
            print("[OK] Sensitive env keys appear empty or placeholder-only.")

    if warnings:
        for warning in warnings:
            print(warning)
        if args.strict:
            raise SystemExit(1)
    else:
        print("[OK] No .env submission safety issue detected.")


if __name__ == "__main__":
    main()

