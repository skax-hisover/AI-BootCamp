"""Common application error contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JobPilotError(Exception):
    error_code: str
    detail: str
    status_code: int = 400

    def __str__(self) -> str:
        return self.detail

    def to_payload(self) -> dict[str, Any]:
        return {"error_code": self.error_code, "detail": self.detail}

