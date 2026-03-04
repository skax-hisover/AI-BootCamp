"""Lightweight PII masking helpers."""

from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?82[-\s]?)?0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b")


def mask_pii_text(text: str) -> str:
    masked = _EMAIL_RE.sub("[EMAIL_MASKED]", text)
    masked = _PHONE_RE.sub("[PHONE_MASKED]", masked)
    return masked


def mask_pii_payload(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii_text(value)
    if isinstance(value, list):
        return [mask_pii_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(k): mask_pii_payload(v) for k, v in value.items()}
    return value

