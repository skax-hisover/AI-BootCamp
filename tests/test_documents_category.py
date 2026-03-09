from pathlib import Path

from src.retrieval.documents import _infer_root_category_from_filename, load_documents_with_report


def test_infer_root_category_from_filename_job_postings() -> None:
    assert _infer_root_category_from_filename(Path("job_market_guide.md")) == "job_postings"
    assert _infer_root_category_from_filename(Path("채용공고_백엔드.md")) == "job_postings"


def test_infer_root_category_from_filename_jd_and_interview() -> None:
    assert _infer_root_category_from_filename(Path("sample_jd_template.md")) == "jd"
    assert _infer_root_category_from_filename(Path("interview_strategy.md")) == "interview_guides"


def test_infer_root_category_from_filename_portfolio_fallback() -> None:
    assert _infer_root_category_from_filename(Path("resume_writing_tips.md")) == "portfolio_examples"
    assert _infer_root_category_from_filename(Path("notes.md")) == "uncategorized"


def test_load_documents_merges_sidecar_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "interview_strategy.md"
    doc_path.write_text("질문 준비 체크리스트", encoding="utf-8")
    sidecar_path = tmp_path / "interview_strategy.md.meta.json"
    sidecar_path.write_text(
        """
{
  "collected_at": "2026-03-09",
  "source_url": "https://example.com/interview",
  "curator": "jobpilot",
  "license": "CC-BY-4.0"
}
""".strip(),
        encoding="utf-8",
    )

    docs, failures = load_documents_with_report(tmp_path)
    assert failures == []
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["collected_at"] == "2026-03-09"
    assert meta["source_url"] == "https://example.com/interview"
    assert meta["curator"] == "jobpilot"
    assert meta["license"] == "CC-BY-4.0"


def test_load_documents_warns_when_sidecar_missing(tmp_path: Path, capsys) -> None:
    doc_path = tmp_path / "resume_tips.md"
    doc_path.write_text("성과 중심 불릿 작성", encoding="utf-8")

    docs, _ = load_documents_with_report(tmp_path)
    captured = capsys.readouterr()

    assert len(docs) == 1
    assert "Missing metadata sidecar for resume_tips.md" in captured.out
