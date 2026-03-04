from src.agents.tools import interview_question_bank, resume_keyword_match_score


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
