"""Shared settings loader for the final project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from src.common import JobPilotError


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    knowledge_dir: Path
    index_dir: Path
    aoai_endpoint: str
    aoai_api_key: str
    aoai_deployment: str
    aoai_embedding_deployment: str
    aoai_api_version: str
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120
    memory_max_sessions: int = 200
    memory_ttl_seconds: int = 60 * 60 * 24
    index_force_rebuild: bool = False
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rag_evidence_score_threshold: float = 0.45
    rerank_enabled: bool = True
    rerank_provider: str = "heuristic"
    graph_state_cache_enabled: bool = True
    graph_state_cache_bypass_contextual: bool = True
    session_memory_persist_enabled: bool = True
    session_memory_pii_mask_enabled: bool = False
    ui_history_persist_enabled: bool = True
    ui_history_pii_mask_enabled: bool = False


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    endpoint = (
        os.getenv("AOAI_ENDPOINT")
        or os.getenv("\ufeffAOAI_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    api_key = os.getenv("AOAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AOAI_DEPLOY_GPT4O") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
    embedding_deployment = (
        os.getenv("AOAI_EMBEDDING_DEPLOYMENT")
        or os.getenv("AOAI_DEPLOY_EMBED_ADA")
        or "text-embedding-ada-002"
    )
    api_version = os.getenv("AOAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-10-21"
    memory_max_sessions = _int_env("MEMORY_MAX_SESSIONS", 200)
    memory_ttl_seconds = _int_env("MEMORY_TTL_SECONDS", 60 * 60 * 24)
    index_force_rebuild = _bool_env("INDEX_FORCE_REBUILD", False)
    vector_weight = _float_env("VECTOR_WEIGHT", 0.6)
    bm25_weight = _float_env("BM25_WEIGHT", 0.4)
    rag_evidence_score_threshold = _float_env("RAG_EVIDENCE_SCORE_THRESHOLD", 0.45)
    rerank_enabled = _bool_env("RERANK_ENABLED", True)
    rerank_provider = (os.getenv("RERANK_PROVIDER") or "heuristic").strip().lower() or "heuristic"
    graph_state_cache_enabled = _bool_env("GRAPH_STATE_CACHE_ENABLED", True)
    graph_state_cache_bypass_contextual = _bool_env("GRAPH_STATE_CACHE_BYPASS_CONTEXTUAL", True)
    session_memory_persist_enabled = _bool_env(
        "SESSION_MEMORY_PERSIST_ENABLED",
        _bool_env("MEMORY_PERSIST_ENABLED", True),
    )
    session_memory_pii_mask_enabled = _bool_env(
        "SESSION_MEMORY_PII_MASK",
        _bool_env("PII_MASK_ENABLED", False),
    )
    ui_history_persist_enabled = _bool_env("UI_HISTORY_PERSIST_ENABLED", True)
    ui_history_pii_mask_enabled = _bool_env(
        "UI_HISTORY_PII_MASK",
        _bool_env("PII_MASK_ENABLED", False),
    )

    missing = []
    if not endpoint:
        missing.append("AOAI_ENDPOINT")
    if not api_key:
        missing.append("AOAI_API_KEY")

    if missing:
        raise JobPilotError(
            error_code="CONFIG_MISSING_ENV",
            detail=f"Missing environment variables: {', '.join(missing)}",
            status_code=400,
        )

    data_dir = project_root / "data"
    knowledge_dir = data_dir / "knowledge"
    index_dir = data_dir / "index"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        knowledge_dir=knowledge_dir,
        index_dir=index_dir,
        aoai_endpoint=endpoint,
        aoai_api_key=api_key,
        aoai_deployment=deployment,
        aoai_embedding_deployment=embedding_deployment,
        aoai_api_version=api_version,
        memory_max_sessions=memory_max_sessions,
        memory_ttl_seconds=memory_ttl_seconds,
        index_force_rebuild=index_force_rebuild,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rag_evidence_score_threshold=rag_evidence_score_threshold,
        rerank_enabled=rerank_enabled,
        rerank_provider=rerank_provider,
        graph_state_cache_enabled=graph_state_cache_enabled,
        graph_state_cache_bypass_contextual=graph_state_cache_bypass_contextual,
        session_memory_persist_enabled=session_memory_persist_enabled,
        session_memory_pii_mask_enabled=session_memory_pii_mask_enabled,
        ui_history_persist_enabled=ui_history_persist_enabled,
        ui_history_pii_mask_enabled=ui_history_pii_mask_enabled,
    )
