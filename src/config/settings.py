"""Shared settings loader for the final project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
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

    missing = []
    if not endpoint:
        missing.append("AOAI_ENDPOINT")
    if not api_key:
        missing.append("AOAI_API_KEY")

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

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
    )
