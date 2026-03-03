"""Session memory store with optional file persistence."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict


@dataclass
class SessionMemory:
    storage_path: str | Path | None = None
    max_sessions: int = 200
    ttl_seconds: int = 60 * 60 * 24
    _messages: DefaultDict[str, list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _updated_at: dict[str, float] = field(default_factory=dict)

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

        if isinstance(raw, dict) and "sessions" in raw:
            sessions = raw.get("sessions", {})
            updated_at = raw.get("updated_at", {})
            if isinstance(sessions, dict):
                for session_id, messages in sessions.items():
                    if isinstance(session_id, str) and isinstance(messages, list):
                        self._messages[session_id] = self._normalize_messages(messages)
            if isinstance(updated_at, dict):
                for session_id, ts in updated_at.items():
                    if isinstance(session_id, str):
                        try:
                            self._updated_at[session_id] = float(ts)
                        except (TypeError, ValueError):
                            continue
        elif isinstance(raw, dict):
            # Backward-compatible load for old format: {session_id: [messages...]}
            now = time.time()
            for session_id, messages in raw.items():
                if isinstance(session_id, str) and isinstance(messages, list):
                    self._messages[session_id] = self._normalize_messages(messages)
                    self._updated_at[session_id] = now

        self._prune(now=time.time(), flush=False)

    def _normalize_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", ""))
                content = str(message.get("content", ""))
                normalized.append({"role": role, "content": content})
        return normalized

    def _prune_expired(self, now: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        expired = [
            session_id
            for session_id, ts in self._updated_at.items()
            if (now - ts) > self.ttl_seconds
        ]
        for session_id in expired:
            self._messages.pop(session_id, None)
            self._updated_at.pop(session_id, None)
        return bool(expired)

    def _enforce_max_sessions(self) -> bool:
        if self.max_sessions <= 0:
            return False
        removed = False
        while len(self._messages) > self.max_sessions:
            oldest_id = min(self._updated_at, key=self._updated_at.get, default=None)
            if oldest_id is None:
                break
            self._messages.pop(oldest_id, None)
            self._updated_at.pop(oldest_id, None)
            removed = True
        return removed

    def _prune(self, now: float | None = None, flush: bool = True) -> None:
        now = now or time.time()
        changed = self._prune_expired(now=now)
        changed = self._enforce_max_sessions() or changed
        if changed and flush:
            self._flush()

    def _flush(self) -> None:
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessions": {session_id: messages for session_id, messages in self._messages.items()},
            "updated_at": self._updated_at,
            "meta": {"max_sessions": self.max_sessions, "ttl_seconds": self.ttl_seconds},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, session_id: str, role: str, content: str) -> None:
        now = time.time()
        self._prune(now=now, flush=False)
        self._messages[session_id].append({"role": role, "content": content})
        self._updated_at[session_id] = now
        self._enforce_max_sessions()
        self._flush()

    def get(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        self._prune()
        messages = self._messages.get(session_id, [])
        if session_id in self._messages:
            self._updated_at[session_id] = time.time()
            self._flush()
        return messages[-limit:]
