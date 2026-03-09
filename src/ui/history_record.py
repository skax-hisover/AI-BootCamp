"""History record builder for UI persistence with configurable data minimization."""

from __future__ import annotations

import hashlib
from typing import Any


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

