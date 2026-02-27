"""Knowledge document loading and chunk preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv_file(path: Path) -> str:
    df = pd.read_csv(path).fillna("")
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join(rows)


def iter_source_files(knowledge_dir: Path) -> Iterable[Path]:
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(knowledge_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in iter_source_files(knowledge_dir):
        if path.suffix.lower() == ".csv":
            text = _read_csv_file(path)
        else:
            text = _read_text_file(path)

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "filename": path.name},
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
