"""Client factories for Azure OpenAI models."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from src.config.settings import load_settings


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = 0.2) -> AzureChatOpenAI:
    settings = load_settings()
    return AzureChatOpenAI(
        api_key=settings.aoai_api_key,
        azure_endpoint=settings.aoai_endpoint,
        azure_deployment=settings.aoai_deployment,
        api_version=settings.aoai_api_version,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> AzureOpenAIEmbeddings:
    settings = load_settings()
    return AzureOpenAIEmbeddings(
        api_key=settings.aoai_api_key,
        azure_endpoint=settings.aoai_endpoint,
        api_version=settings.aoai_api_version,
        model=settings.aoai_embedding_deployment,
    )
