from src.retrieval import SearchHit, rerank_hits


def test_rerank_prioritizes_uploaded_jd_when_enabled() -> None:
    hits = [
        SearchHit(
            content="일반 지식 문서 내용",
            source="knowledge.md",
            score=0.58,
            metadata={"category": "uncategorized"},
        ),
        SearchHit(
            content="백엔드 API 성능 개선과 트랜잭션 최적화 역량이 필요합니다.",
            source="uploaded_jd_text",
            score=0.50,
            metadata={"category": "jd", "source_type": "jd_upload"},
        ),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="백엔드 API 성능 개선",
        role_hint="backend, api, server, database",
        route_categories={"jd", "job_postings"},
        top_k=2,
        provider="heuristic",
        enabled=True,
    )
    assert ranked[0].source == "uploaded_jd_text"


def test_rerank_disabled_keeps_base_score_priority() -> None:
    hits = [
        SearchHit(
            content="일반 지식 문서 내용",
            source="knowledge.md",
            score=0.58,
            metadata={"category": "uncategorized"},
        ),
        SearchHit(
            content="백엔드 API 성능 개선과 트랜잭션 최적화 역량이 필요합니다.",
            source="uploaded_jd_text",
            score=0.50,
            metadata={"category": "jd", "source_type": "jd_upload"},
        ),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="백엔드 API 성능 개선",
        role_hint="backend, api, server, database",
        route_categories={"jd", "job_postings"},
        top_k=2,
        provider="heuristic",
        enabled=False,
    )
    assert ranked[0].source == "knowledge.md"
