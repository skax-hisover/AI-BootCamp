from src.workflow.engine import (
    _cache_record_payload_for_request,
    _sanitize_evidence_map,
    _upsert_cache_record,
    enforce_final_answer_policy,
    heuristic_route_from_query,
    normalize_final_answer_by_route,
    route_minimums,
)
from src.workflow.contracts import enforce_chat_response_contract


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


def test_plan_only_summary_applies_line_and_char_guardrail() -> None:
    payload = {
        "summary": "첫 줄 요약\n둘째 줄 요약\n셋째 줄은 제거되어야 함",
        "resume_improvements": [],
        "interview_preparation": [],
        "two_week_plan": ["p1"],
        "references": [],
    }
    normalized = normalize_final_answer_by_route("plan_only", payload)
    assert "셋째 줄" not in normalized["summary"]
    assert len(normalized["summary"]) <= 143  # max_chars + ellipsis allowance


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


def test_heuristic_route_ignores_negated_exclusion_phrase() -> None:
    route = heuristic_route_from_query("면접 제외하고 싶진 않지만 우선 이력서만 먼저 보고 싶어.")
    # Negated exclusion should not force a heuristic route.
    # Let the LLM router decide with full context.
    assert route is None


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


def test_enforce_chat_response_contract_fills_min_schema() -> None:
    payload = enforce_chat_response_contract({"summary": "", "cached_state_hit": "yes"})
    assert payload["summary"]
    assert isinstance(payload["resume_improvements"], list)
    assert isinstance(payload["references"], list)
    assert payload["cached_state_hit"] is True


def test_enforce_chat_response_contract_normalizes_node_status() -> None:
    payload = enforce_chat_response_contract(
        {
            "summary": "ok",
            "node_status": {
                "resume": {
                    "status": "degraded",
                    "error_code": "STRUCTURED_OUTPUT_RESUME_FALLBACK",
                    "detail": "fallback used",
                }
            },
        }
    )
    assert isinstance(payload["node_status"], dict)
    assert payload["node_status"]["resume"]["status"] == "degraded"


def test_sanitize_evidence_map_clips_out_of_range_indices() -> None:
    evidence_map = {
        "항목A": [1, 2, 7, 0, -1, 2],
        "항목B": ["3", "x", None],
        "": [1],
    }
    sanitized = _sanitize_evidence_map(evidence_map, max_ref=3)
    assert sanitized == {"항목A": [1, 2], "항목B": [3]}


def test_sanitize_evidence_map_returns_empty_when_no_refs() -> None:
    evidence_map = {"항목A": [1, 2, 3]}
    assert _sanitize_evidence_map(evidence_map, max_ref=0) == {}


def test_cache_record_supports_signature_keyed_multi_entry() -> None:
    record = _upsert_cache_record({}, "sig-a", {"summary": "A"}, max_items=3)
    record = _upsert_cache_record(record, "sig-b", {"summary": "B"}, max_items=3)
    payload_a = _cache_record_payload_for_request(record, "sig-a")
    payload_b = _cache_record_payload_for_request(record, "sig-b")
    assert payload_a == {"summary": "A"}
    assert payload_b == {"summary": "B"}


def test_cache_record_applies_lru_trim_per_session() -> None:
    record = {}
    for idx in range(1, 5):
        record = _upsert_cache_record(record, f"sig-{idx}", {"summary": str(idx)}, max_items=2)
    assert _cache_record_payload_for_request(record, "sig-4") == {"summary": "4"}
    assert _cache_record_payload_for_request(record, "sig-3") == {"summary": "3"}
    assert _cache_record_payload_for_request(record, "sig-2") is None


def test_enforce_final_answer_policy_keeps_reference_metadata_fields() -> None:
    payload = {
        "summary": "요약",
        "resume_improvements": ["핵심 역량 보강"],
        "interview_preparation": [],
        "two_week_plan": [],
        "references": [
            {
                "rank": 1,
                "source": "sample.md",
                "chunk_id": 3,
                "location": "page=1",
                "score": 0.9,
                "category": "job_postings",
                "snippet": "샘플",
                "collected_at": "2026-03-09",
                "source_url": "https://example.com",
                "curator": "jobpilot-team",
                "license": "CC-BY-4.0",
            }
        ],
    }
    enforced = enforce_final_answer_policy(route="resume_only", payload=payload, rag_refs=[])
    ref = enforced["references"][0]
    assert ref["collected_at"] == "2026-03-09"
    assert ref["source_url"] == "https://example.com"
    assert ref["curator"] == "jobpilot-team"
    assert ref["license"] == "CC-BY-4.0"

