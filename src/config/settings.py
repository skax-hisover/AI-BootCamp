"""Shared settings loader for the final project."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    aoai_endpoint: str
    aoai_api_key: str
    aoai_deployment: str
    aoai_api_version: str



def load_settings() -> Settings:
    load_dotenv()

    endpoint = os.getenv("AOAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AOAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AOAI_DEPLOY_GPT4O") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
    api_version = os.getenv("AOAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-10-21"

    missing = []
    if not endpoint:
        missing.append("AOAI_ENDPOINT")
    if not api_key:
        missing.append("AOAI_API_KEY")

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    return Settings(
        aoai_endpoint=endpoint,
        aoai_api_key=api_key,
        aoai_deployment=deployment,
        aoai_api_version=api_version,
    )
