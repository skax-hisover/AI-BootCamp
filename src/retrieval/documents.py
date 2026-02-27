"""Knowledge document loading and chunk preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv_file(path: Path) -> str:
    df = pd.read_csv(path).fillna("")
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join(rows)


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for PDF loader: install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx_file(path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for DOCX loader: install python-docx") from exc

    doc = DocxDocument(str(path))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def _read_xlsx_file(path: Path) -> str:
    xls = pd.ExcelFile(path)
    blocks: list[str] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet).fillna("")
        rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
        blocks.append(f"[sheet: {sheet}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def iter_source_files(knowledge_dir: Path) -> Iterable[Path]:
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(knowledge_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in iter_source_files(knowledge_dir):
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                text = _read_csv_file(path)
            elif suffix == ".pdf":
                text = _read_pdf_file(path)
            elif suffix == ".docx":
                text = _read_docx_file(path)
            elif suffix == ".xlsx":
                text = _read_xlsx_file(path)
            else:
                text = _read_text_file(path)
        except Exception as exc:
            # Keep service available even if one file is malformed or parser dependency is missing.
            print(f"[WARN] Failed to load {path.name}: {exc}")
            continue

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
