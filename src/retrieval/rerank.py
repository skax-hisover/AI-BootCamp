"""Rerank strategies for retrieval hits."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def _is_role_relevant(text: str, hint: str) -> bool:
    keywords = [token.strip().lower() for token in hint.split(",") if token.strip()]
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords) if keywords else True


def rerank_hits(
    hits: list[Any],
    query: str,
    role_hint: str,
    route_categories: set[str] | None = None,
    top_k: int = 4,
    provider: str = "heuristic",
    enabled: bool = True,
) -> list[Any]:
    """Rerank retrieval hits with pluggable strategy.

    Notes:
    - `provider=cross_encoder|llm` currently falls back to `heuristic`.
    - Keep this module isolated so higher-quality rerankers can be swapped in later.
    """
    if not hits:
        return []

    ranked = sorted(hits, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
    if not enabled or provider.lower() in {"none", "off", "disabled"}:
        return ranked[:top_k]

    provider_key = provider.lower()
    if provider_key in {"cross_encoder", "llm"}:
        provider_key = "heuristic"

    if provider_key != "heuristic":
        return ranked[:top_k]

    query_tokens = [token for token in query.lower().split() if token]
    lowered_categories = {item.lower() for item in route_categories} if route_categories else set()
    rescored: list[Any] = []

    for hit in ranked:
        source = str(getattr(hit, "source", ""))
        content = str(getattr(hit, "content", ""))
        metadata = dict(getattr(hit, "metadata", {}) or {})
        base_score = float(getattr(hit, "score", 0.0))
        category = str(metadata.get("category", "")).lower()
        source_type = str(metadata.get("source_type", "")).lower()

        role_boost = 0.15 if _is_role_relevant(f"{source} {content}", role_hint) else 0.0
        category_boost = 0.08 if lowered_categories and category in lowered_categories else 0.0
        lexical_boost = 0.05 if any(token in content.lower() for token in query_tokens) else 0.0
        # Prioritize uploaded JD as first-class evidence for gap analysis.
        upload_boost = 0.12 if source_type == "jd_upload" else (0.08 if source_type == "resume_upload" else 0.0)

        new_score = round(base_score + role_boost + category_boost + lexical_boost + upload_boost, 4)
        hit_kwargs = asdict(hit)
        hit_kwargs["score"] = new_score
        rescored.append(type(hit)(**hit_kwargs))

    rescored.sort(key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
    return rescored[:top_k]
