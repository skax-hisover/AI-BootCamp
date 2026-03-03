"""Session memory store with optional file persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict


@dataclass
class SessionMemory:
    storage_path: str | Path | None = None
    _messages: DefaultDict[str, list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(raw, dict):
            return

        for session_id, messages in raw.items():
            if isinstance(session_id, str) and isinstance(messages, list):
                normalized: list[dict[str, str]] = []
                for message in messages:
                    if isinstance(message, dict):
                        role = str(message.get("role", ""))
                        content = str(message.get("content", ""))
                        normalized.append({"role": role, "content": content})
                self._messages[session_id] = normalized

    def _flush(self) -> None:
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {session_id: messages for session_id, messages in self._messages.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})
        self._flush()

    def get(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        return self._messages[session_id][-limit:]
