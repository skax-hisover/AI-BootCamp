"""Lightweight in-memory session store."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    _messages: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})

    def get(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        return self._messages[session_id][-limit:]
