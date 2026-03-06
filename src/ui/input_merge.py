"""Helpers for deterministic text merge policy used by Streamlit upload inputs."""

from __future__ import annotations


def merge_uploaded_text(existing: str, uploaded: str, mode: str) -> str:
    """Merge UI text input with uploaded text according to configured mode.

    Rules:
    - empty uploaded text keeps existing text unchanged
    - mode == "추가하기" appends uploaded text after existing (with blank-line separator)
    - otherwise (including "덮어쓰기"), uploaded text replaces existing text
    """
    existing_clean = (existing or "").strip()
    uploaded_clean = (uploaded or "").strip()
    if not uploaded_clean:
        return existing or ""
    if mode == "추가하기" and existing_clean:
        if existing_clean == uploaded_clean:
            return existing_clean
        return f"{existing_clean}\n\n{uploaded_clean}".strip()
    return uploaded_clean

