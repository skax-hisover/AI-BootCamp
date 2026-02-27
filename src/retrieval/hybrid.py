"""Hybrid retriever: FAISS vector search + BM25 keyword search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_embedding_model, load_settings
from src.retrieval.documents import chunk_documents, load_documents


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


@dataclass
class SearchHit:
    content: str
    source: str
    score: float


class HybridRetriever:
    """Simple in-memory hybrid retriever."""

    def __init__(self, chunks: list[Document], vector_db: FAISS, bm25: BM25Okapi) -> None:
        self.chunks = chunks
        self.vector_db = vector_db
        self.bm25 = bm25

    @classmethod
    def build(cls) -> "HybridRetriever":
        settings = load_settings()
        docs = load_documents(settings.knowledge_dir)
        if not docs:
            raise ValueError(
                f"No knowledge documents found in {settings.knowledge_dir}. "
                "Add .txt/.md/.csv files before running."
            )

        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        embeddings = get_embedding_model()
        vector_db = FAISS.from_documents(chunks, embedding=embeddings)

        tokenized_chunks = [_tokenize(doc.page_content) for doc in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        return cls(chunks=chunks, vector_db=vector_db, bm25=bm25)

    def _vector_scores(self, query: str, top_k: int) -> dict[int, float]:
        docs_with_scores = self.vector_db.similarity_search_with_score(query, k=top_k)
        if not docs_with_scores:
            return {}

        # FAISS distance: lower is better -> convert to higher-is-better score.
        raw = np.array([distance for _, distance in docs_with_scores], dtype=float)
        normalized = 1.0 / (1.0 + raw)
        scores: dict[int, float] = {}
        for (doc, _), score in zip(docs_with_scores, normalized):
            idx = doc.metadata.get("chunk_id")
            if isinstance(idx, int):
                scores[idx] = float(score)
        return scores

    def _bm25_scores(self, query: str, top_k: int) -> dict[int, float]:
        tokens = _tokenize(query)
        raw_scores = np.array(self.bm25.get_scores(tokens), dtype=float)
        if raw_scores.size == 0:
            return {}
        top_indices = raw_scores.argsort()[::-1][:top_k]
        max_score = raw_scores[top_indices].max() if len(top_indices) else 0.0
        denom = max(max_score, 1e-8)
        return {int(i): float(raw_scores[i] / denom) for i in top_indices}

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        settings = load_settings()
        k = top_k or settings.top_k

        vector_scores = self._vector_scores(query, k)
        bm25_scores = self._bm25_scores(query, k * 2)

        merged: dict[int, float] = {}
        for idx, score in vector_scores.items():
            merged[idx] = merged.get(idx, 0.0) + (0.6 * score)
        for idx, score in bm25_scores.items():
            merged[idx] = merged.get(idx, 0.0) + (0.4 * score)

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
        hits: list[SearchHit] = []
        for idx, score in ranked:
            doc = self.chunks[idx]
            hits.append(
                SearchHit(
                    content=doc.page_content,
                    source=doc.metadata.get("filename", "unknown"),
                    score=round(score, 4),
                )
            )
        return hits

    def search_as_context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        hits = self.search(query, top_k=top_k)
        context_parts = []
        refs: list[dict[str, Any]] = []
        for i, hit in enumerate(hits, start=1):
            context_parts.append(f"[{i}] ({hit.source}, score={hit.score})\n{hit.content}")
            refs.append({"rank": i, "source": hit.source, "score": hit.score})
        return "\n\n".join(context_parts), refs
