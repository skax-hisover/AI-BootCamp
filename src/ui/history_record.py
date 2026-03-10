"""History record builder for UI persistence with configurable data minimization."""

from __future__ import annotations

import hashlib
from typing import Any

HISTORY_RECORD_VERSION = 1


def _sha256_short(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _clip(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _summary_response(payload: dict[str, Any]) -> dict[str, Any]:
    references = payload.get("references", [])
    compact_refs: list[dict[str, Any]] = []
    if isinstance(references, list):
        for item in references[:3]:
            if isinstance(item, dict):
                compact_refs.append(
                    {
                        "rank": item.get("rank"),
                        "source": item.get("source"),
                        "score": item.get("score"),
                    }
                )
    return {
        "summary": str(payload.get("summary", "") or ""),
        "route": payload.get("route"),
        "routing_reason": payload.get("routing_reason"),
        "rag_low_confidence": payload.get("rag_low_confidence"),
        "cached_state_hit": payload.get("cached_state_hit", False),
        "resume_improvements": list(payload.get("resume_improvements", []) or [])[:5],
        "interview_preparation": list(payload.get("interview_preparation", []) or [])[:5],
        "two_week_plan": list(payload.get("two_week_plan", []) or [])[:5],
        "input_gap_notice": payload.get("input_gap_notice"),
        "references": compact_refs,
    }


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def migrate_history_record(item: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a history record to the latest schema version."""
    normalized = dict(item)
    version = _to_int(normalized.get("record_version", 0), default=0)
    if version <= 0:
        # v0 -> v1: add explicit schema version and normalize minimal required fields.
        normalized["record_version"] = 1
    else:
        normalized["record_version"] = version

    normalized["storage_mode"] = (
        str(normalized.get("storage_mode", "full"))
        if str(normalized.get("storage_mode", "full")) in {"summary", "full"}
        else "full"
    )
    normalized["query"] = str(normalized.get("query", "") or "")
    normalized["target_role"] = str(normalized.get("target_role", "백엔드 개발자") or "백엔드 개발자")
    normalized["session_id"] = str(normalized.get("session_id", "") or "")
    normalized["resume_len"] = _to_int(normalized.get("resume_len", 0), default=0)
    normalized["jd_len"] = _to_int(normalized.get("jd_len", 0), default=0)
    normalized["resume_text"] = str(normalized.get("resume_text", "") or "")
    normalized["jd_text"] = str(normalized.get("jd_text", "") or "")
    if "response" in normalized and not isinstance(normalized.get("response"), dict):
        normalized["response"] = {}
    return normalized


def build_history_record(
    *,
    session_id: str,
    query: str,
    target_role: str,
    resume_text: str,
    jd_text: str,
    response_payload: dict[str, Any],
    storage_mode: str = "summary",
) -> dict[str, Any]:
    mode = storage_mode if storage_mode in {"summary", "full"} else "summary"
    base = {
        "record_version": HISTORY_RECORD_VERSION,
        "session_id": session_id,
        "query": query.strip(),
        "target_role": target_role,
        "resume_len": len(resume_text or ""),
        "jd_len": len(jd_text or ""),
        "resume_hash": _sha256_short(resume_text or ""),
        "jd_hash": _sha256_short(jd_text or ""),
        "storage_mode": mode,
    }
    if mode == "full":
        return {
            **base,
            "resume_text": resume_text,
            "jd_text": jd_text,
            "response": response_payload,
        }
    return {
        **base,
        "resume_text": "",
        "jd_text": "",
        "resume_preview": _clip(resume_text, 500),
        "jd_preview": _clip(jd_text, 500),
        "response": _summary_response(response_payload),
    }

