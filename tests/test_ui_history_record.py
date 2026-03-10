from src.ui.history_record import build_history_record, migrate_history_record


def _sample_response() -> dict:
    return {
        "summary": "요약",
        "resume_improvements": ["r1", "r2"],
        "interview_preparation": ["i1"],
        "two_week_plan": ["p1", "p2"],
        "references": [{"rank": 1, "source": "doc.md", "score": 0.9}],
        "route": "full",
        "routing_reason": "test",
        "rag_low_confidence": False,
        "cached_state_hit": False,
    }


def test_build_history_record_summary_mode_minimizes_payload() -> None:
    record = build_history_record(
        session_id="s1",
        query="q",
        target_role="백엔드 개발자",
        resume_text="resume text",
        jd_text="jd text",
        response_payload=_sample_response(),
        run_id="run-1",
        storage_mode="summary",
    )
    assert record["storage_mode"] == "summary"
    assert record["record_version"] == 1
    assert record["run_id"] == "run-1"
    assert record["resume_text"] == ""
    assert record["jd_text"] == ""
    assert record["resume_hash"]
    assert record["jd_hash"]
    assert isinstance(record["response"], dict)
    assert "summary" in record["response"]


def test_build_history_record_full_mode_keeps_raw_text() -> None:
    record = build_history_record(
        session_id="s1",
        query="q",
        target_role="백엔드 개발자",
        resume_text="resume text",
        jd_text="jd text",
        response_payload=_sample_response(),
        run_id="run-2",
        storage_mode="full",
    )
    assert record["storage_mode"] == "full"
    assert record["record_version"] == 1
    assert record["run_id"] == "run-2"
    assert record["resume_text"] == "resume text"
    assert record["jd_text"] == "jd text"
    assert record["response"]["summary"] == "요약"


def test_migrate_history_record_v0_to_v1() -> None:
    legacy = {
        "session_id": "s1",
        "query": "q",
        "target_role": "백엔드 개발자",
        "resume_len": "11",
        "jd_len": "7",
        "storage_mode": "summary",
        "response": "legacy text",
    }
    migrated = migrate_history_record(legacy)
    assert migrated["record_version"] == 1
    assert migrated["resume_len"] == 11
    assert migrated["jd_len"] == 7
    assert migrated["storage_mode"] == "summary"
    assert migrated["run_id"] == ""
    assert isinstance(migrated["response"], dict)

