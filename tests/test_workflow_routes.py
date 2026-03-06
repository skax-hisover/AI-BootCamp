from src.workflow.engine import (
    enforce_final_answer_policy,
    heuristic_route_from_query,
    normalize_final_answer_by_route,
    route_minimums,
)


def test_route_minimums_for_resume_and_interview_only() -> None:
    resume_rules = route_minimums("resume_only")
    interview_rules = route_minimums("interview_only")
    assert resume_rules["two_week_plan"] == 0
    assert interview_rules["two_week_plan"] == 0


def test_normalize_final_answer_by_route_blanks_irrelevant_sections() -> None:
    payload = {
        "summary": "요약",
        "resume_improvements": ["r1"],
        "interview_preparation": ["i1"],
        "two_week_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("plan_only", payload)
    assert normalized["resume_improvements"] == []
    assert normalized["interview_preparation"] == []
    assert normalized["two_week_plan"] == ["p1"]


def test_plan_only_summary_is_clipped_to_one_or_two_sentences() -> None:
    payload = {
        "summary": "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장은 잘려야 합니다.",
        "resume_improvements": [],
        "interview_preparation": [],
        "two_week_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("plan_only", payload)
    assert "세 번째 문장" not in normalized["summary"]
    assert normalized["summary"].count(".") <= 2


def test_summary_is_not_notice_only_when_model_returns_notice_text() -> None:
    payload = {
        "summary": "법/세무/노무 등 비전문 영역은 별도 확인이 필요하며 최신 공고/회사 정책은 반드시 원문 확인이 필요합니다.",
        "resume_improvements": ["r1", "r2"],
        "interview_preparation": ["i1", "i2"],
        "two_week_plan": ["p1", "p2"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("full", payload)
    assert "이력서 개선" in normalized["summary"]
    assert "면접 준비" in normalized["summary"]
    assert "2주 계획" in normalized["summary"]


def test_enforce_final_answer_policy_adds_refs_and_citation() -> None:
    payload = {
        "summary": "요약",
        "resume_improvements": ["핵심 역량 보강"],
        "interview_preparation": ["질문 대비"],
        "two_week_plan": [],
        "references": [],
    }
    enforced = enforce_final_answer_policy(
        route="full",
        payload=payload,
        rag_refs=[
            {
                "rank": 1,
                "source": "sample.md",
                "chunk_id": 1,
                "location": "n/a",
                "score": 0.8,
                "category": "uncategorized",
                "snippet": "sample snippet",
            }
        ],
    )
    assert enforced["references"]
    assert isinstance(enforced["references"][0], dict)
    assert all("[1]" in item for item in enforced["resume_improvements"])
    assert len(enforced["two_week_plan"]) >= 4
    assert all("추가 권장 액션" not in item for item in enforced["two_week_plan"])
    assert all("근거/입력 정보가 부족해 2주 실행계획" not in item for item in enforced["two_week_plan"])
    assert enforced.get("input_gap_notice")


def test_heuristic_route_from_query_respects_explicit_exclusion() -> None:
    route = heuristic_route_from_query("이력서 개선 포인트만 5개 뽑아줘. 면접 제외, 계획 제외")
    assert route is not None
    assert route[0] == "resume_only"


def test_heuristic_route_from_query_plan_only_keywords() -> None:
    route = heuristic_route_from_query("전체 요약 없이 2주 계획만 간단히 작성해줘")
    assert route is not None
    assert route[0] == "plan_only"


def test_normalize_final_answer_by_route_cleans_none_notice() -> None:
    payload = {
        "summary": "요약",
        "resume_improvements": ["r1"],
        "interview_preparation": ["i1"],
        "two_week_plan": ["p1"],
        "input_gap_notice": None,
        "references": [],
    }
    normalized = normalize_final_answer_by_route("full", payload)
    assert normalized.get("input_gap_notice") is None

