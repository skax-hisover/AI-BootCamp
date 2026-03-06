from pathlib import Path

from src.retrieval.documents import _infer_root_category_from_filename


def test_infer_root_category_from_filename_job_postings() -> None:
    assert _infer_root_category_from_filename(Path("job_market_guide.md")) == "job_postings"
    assert _infer_root_category_from_filename(Path("채용공고_백엔드.md")) == "job_postings"


def test_infer_root_category_from_filename_jd_and_interview() -> None:
    assert _infer_root_category_from_filename(Path("sample_jd_template.md")) == "jd"
    assert _infer_root_category_from_filename(Path("interview_strategy.md")) == "interview_guides"


def test_infer_root_category_from_filename_portfolio_fallback() -> None:
    assert _infer_root_category_from_filename(Path("resume_writing_tips.md")) == "portfolio_examples"
    assert _infer_root_category_from_filename(Path("notes.md")) == "uncategorized"
