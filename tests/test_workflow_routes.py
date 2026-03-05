from src.workflow.engine import (
    enforce_final_answer_policy,
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

