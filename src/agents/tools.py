"""Tools used by specialist agents."""

from __future__ import annotations

import re

from langchain_core.tools import tool


@tool
def resume_keyword_match_score(resume_text: str, target_role: str) -> str:
    """Estimate keyword match score of a resume for a target role."""
    role_keywords = {
        "백엔드": ["python", "api", "database", "sql", "cloud", "docker", "fastapi", "django"],
        "데이터": ["pandas", "sql", "분석", "시각화", "모델", "통계", "etl"],
        "pm": ["우선순위", "협업", "지표", "로드맵", "의사결정", "커뮤니케이션"],
    }

    key = "백엔드"
    lowered = target_role.lower()
    if "데이터" in target_role or "data" in lowered:
        key = "데이터"
    elif "pm" in lowered:
        key = "pm"

    tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", resume_text.lower()))
    keywords = role_keywords[key]
    matched = [kw for kw in keywords if kw.lower() in tokens]
    score = int((len(matched) / max(len(keywords), 1)) * 100)
    return f"target_role={target_role}, match_score={score}, matched_keywords={matched}"


@tool
def interview_question_bank(target_role: str) -> str:
    """Return common interview question sets for a role."""
    mapping = {
        "백엔드": [
            "대규모 트래픽 상황에서 API 성능을 개선한 경험을 설명해 주세요.",
            "트랜잭션 격리 수준과 데드락 해결 경험을 말해 주세요.",
            "장애 대응 프로세스를 본인이 주도한 사례가 있나요?",
        ],
        "데이터": [
            "모델 성능보다 비즈니스 임팩트가 중요했던 프로젝트 경험은?",
            "데이터 품질 이슈를 발견하고 해결한 과정을 설명해 주세요.",
            "A/B 테스트 결과를 의사결정에 어떻게 연결했나요?",
        ],
        "pm": [
            "우선순위 충돌 상황에서 어떤 기준으로 의사결정했나요?",
            "실패했던 기능 출시 사례와 학습한 점을 말해 주세요.",
            "개발/디자인/사업 부서 협업 갈등을 해결한 경험은?",
        ],
    }
    key = "백엔드"
    lowered = target_role.lower()
    if "데이터" in target_role or "data" in lowered:
        key = "데이터"
    elif "pm" in lowered:
        key = "pm"
    return "\n".join(f"- {q}" for q in mapping[key])
