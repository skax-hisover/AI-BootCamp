"""Common file extraction utilities shared by retrieval loader and UI upload parser."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def _decode_text_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp949", errors="ignore")


def _split_paragraph_blocks(text: str) -> list[dict[str, object]]:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return [
        {"content": block, "metadata": {"section_type": "paragraph", "section_index": idx + 1}}
        for idx, block in enumerate(blocks)
    ]


def _extract_pdf_sections_from_stream(stream: BytesIO) -> list[dict[str, object]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for PDF loader: install pypdf") from exc
    reader = PdfReader(stream)
    pages: list[dict[str, object]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(
                {
                    "content": page_text,
                    "metadata": {"section_type": "pdf_page", "page_number": page_idx},
                }
            )
    return pages


def _extract_docx_sections_from_stream(stream: BytesIO) -> list[dict[str, object]]:
    try:
        from docx import Document as DocxDocument
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for DOCX loader: install python-docx") from exc
    doc = DocxDocument(stream)
    paragraphs: list[dict[str, object]] = []
    for para_idx, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            paragraphs.append(
                {
                    "content": text,
                    "metadata": {"section_type": "docx_paragraph", "paragraph_number": para_idx},
                }
            )
    return paragraphs


def _extract_xlsx_sections_from_stream(stream: BytesIO) -> list[dict[str, object]]:
    xls = pd.ExcelFile(stream)
    blocks: list[dict[str, object]] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet).fillna("")
        for row_idx, row in enumerate(df.values.tolist(), start=1):
            row_text = " | ".join(map(str, row)).strip()
            if row_text:
                blocks.append(
                    {
                        "content": row_text,
                        "metadata": {
                            "section_type": "xlsx_row",
                            "sheet_name": sheet,
                            "row_number": row_idx,
                        },
                    }
                )
    return blocks


def _extract_csv_sections_from_stream(stream: BytesIO) -> list[dict[str, object]]:
    df = pd.read_csv(stream).fillna("")
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(df.values.tolist(), start=1):
        row_text = " | ".join(map(str, row)).strip()
        if row_text:
            rows.append(
                {
                    "content": row_text,
                    "metadata": {"section_type": "csv_row", "row_number": idx},
                }
            )
    return rows


def extract_sections_from_path(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _split_paragraph_blocks(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return _extract_csv_sections_from_stream(BytesIO(path.read_bytes()))
    if suffix == ".pdf":
        return _extract_pdf_sections_from_stream(BytesIO(path.read_bytes()))
    if suffix == ".docx":
        return _extract_docx_sections_from_stream(BytesIO(path.read_bytes()))
    if suffix == ".xlsx":
        return _extract_xlsx_sections_from_stream(BytesIO(path.read_bytes()))
    raise RuntimeError(f"Unsupported extension: {suffix}")


def extract_text_from_upload(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return _decode_text_bytes(data).strip()
    if name.endswith(".pdf"):
        sections = _extract_pdf_sections_from_stream(BytesIO(data))
        return "\n".join(str(item["content"]) for item in sections).strip()
    if name.endswith(".docx"):
        sections = _extract_docx_sections_from_stream(BytesIO(data))
        return "\n".join(str(item["content"]) for item in sections).strip()
    if name.endswith(".xlsx"):
        sections = _extract_xlsx_sections_from_stream(BytesIO(data))
        # UI preview readability: keep sheet boundary in flattened text.
        by_sheet: dict[str, list[str]] = {}
        for item in sections:
            metadata = item.get("metadata", {})
            sheet_name = str(metadata.get("sheet_name", "sheet"))
            by_sheet.setdefault(sheet_name, []).append(str(item.get("content", "")))
        return "\n\n".join(
            f"[sheet: {sheet}]\n" + "\n".join(rows) for sheet, rows in by_sheet.items()
        ).strip()
    raise RuntimeError("지원하지 않는 파일 형식입니다. txt/md/pdf/docx/xlsx 파일을 업로드해 주세요.")

