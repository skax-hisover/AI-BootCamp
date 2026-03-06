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


def test_rerank_limits_same_source_by_max_per_source() -> None:
    hits = [
        SearchHit(content="문서A-1", source="same.md", score=0.9, metadata={"category": "jd"}),
        SearchHit(content="문서A-2", source="same.md", score=0.88, metadata={"category": "jd"}),
        SearchHit(content="문서A-3", source="same.md", score=0.87, metadata={"category": "jd"}),
        SearchHit(content="문서B-1", source="other.md", score=0.7, metadata={"category": "jd"}),
    ]
    ranked = rerank_hits(
        hits=hits,
        query="백엔드 jd",
        role_hint="backend, api",
        route_categories={"jd"},
        top_k=4,
        max_per_source=2,
        provider="heuristic",
        enabled=True,
    )
    same_count = sum(1 for item in ranked if item.source == "same.md")
    assert same_count <= 2
