from src.agents.tools import interview_question_bank, jd_resume_gap_score, resume_keyword_match_score


def test_resume_keyword_match_score() -> None:
    result = resume_keyword_match_score.invoke(
        {
            "resume_text": "Python FastAPI API database cloud docker 경험",
            "target_role": "백엔드 개발자",
        }
    )
    assert '"match_score":' in result


def test_interview_question_bank() -> None:
    result = interview_question_bank.invoke({"target_role": "PM"})
    assert '"questions":' in result
    assert "우선순위" in result


def test_jd_resume_gap_score() -> None:
    result = jd_resume_gap_score.invoke(
        {
            "jd_text": "백엔드 포지션: python, api, sql, docker 필수. kafka 우대.",
            "resume_text": "python api 경험과 sql 프로젝트 경험이 있습니다.",
            "target_role": "백엔드 개발자",
        }
    )
    assert '"required_match_rate":' in result
    assert '"missing_required_top":' in result
