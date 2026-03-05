"""Tools used by specialist agents."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from langchain_core.tools import tool


def _infer_role_keywords(target_role: str) -> tuple[str, list[str], list[str]]:
    role_required = {
        "백엔드": ["python", "api", "database", "sql", "cloud", "docker", "fastapi", "django"],
        "데이터": ["sql", "python", "분석", "통계", "모델", "시각화", "etl"],
        "pm": ["우선순위", "협업", "지표", "로드맵", "의사결정", "커뮤니케이션"],
    }
    role_preferred = {
        "백엔드": ["kafka", "redis", "msa", "모니터링", "테스트 자동화"],
        "데이터": ["실험", "a/b", "대시보드", "데이터 파이프라인", "가설 검증"],
        "pm": ["가설", "실험", "백로그", "이해관계자", "리스크 관리"],
    }

    key = "백엔드"
    lowered = target_role.lower()
    if "데이터" in target_role or "data" in lowered:
        key = "데이터"
    elif "pm" in lowered:
        key = "pm"
    return key, role_required[key], role_preferred[key]


_SYNONYM_CANONICAL = {
    "rdbms": "database",
    "db": "database",
    "mysql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "fastapi": "fastapi",
    "fast-api": "fastapi",
    "backend": "백엔드",
    "back-end": "백엔드",
    "pm": "pm",
    "productmanager": "pm",
    "k8s": "kubernetes",
}


def _normalize_token(token: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]+", "", token.lower()).strip()
    if not cleaned:
        return ""
    return _SYNONYM_CANONICAL.get(cleaned, cleaned)


@lru_cache(maxsize=1)
def _get_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ModuleNotFoundError:
        return None
    return Kiwi()


def _tokenize(text: str) -> set[str]:
    kiwi = _get_kiwi()
    if kiwi is not None:
        tokens: set[str] = set()
        for token in kiwi.tokenize(text):
            norm = _normalize_token(token.form)
            if norm:
                tokens.add(norm)
        return tokens
    rough = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return {norm for token in rough if (norm := _normalize_token(token))}


def _expand_keyword_aliases(keyword: str) -> set[str]:
    base = _normalize_token(keyword)
    if not base:
        return set()
    aliases: dict[str, set[str]] = {
        "database": {"database", "db", "rdbms", "mysql", "postgres", "postgresql"},
        "sql": {"sql", "mysql", "postgres", "postgresql"},
        "fastapi": {"fastapi", "fast-api"},
        "백엔드": {"백엔드", "backend", "back-end"},
    }
    for canonical, words in aliases.items():
        if base in words:
            return {_normalize_token(item) for item in words}
    return {base}


@tool
def resume_keyword_match_score(resume_text: str, target_role: str) -> str:
    """Estimate keyword match score of a resume for a target role."""
    key, role_keywords, _ = _infer_role_keywords(target_role)
    tokens = _tokenize(resume_text)
    matched: list[str] = []
    for keyword in role_keywords:
        aliases = _expand_keyword_aliases(keyword)
        if aliases & tokens:
            matched.append(keyword)
    score = int((len(matched) / max(len(role_keywords), 1)) * 100)
    payload = {
        "target_role": target_role,
        "role_key": key,
        "match_score": score,
        "matched_keywords": matched,
        "missing_keywords": [kw for kw in role_keywords if kw not in matched],
    }
    return json.dumps(payload, ensure_ascii=False)


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
    payload = {
        "target_role": target_role,
        "role_key": key,
        "questions": mapping[key],
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def jd_resume_gap_score(jd_text: str, resume_text: str, target_role: str) -> str:
    """Quantify JD-Resume gap with required/preferred keyword matching."""
    _, required_keywords, preferred_keywords = _infer_role_keywords(target_role)
    jd_tokens = _tokenize(jd_text)
    resume_tokens = _tokenize(resume_text)

    jd_required = [
        kw for kw in required_keywords if _expand_keyword_aliases(kw) & jd_tokens
    ] or required_keywords
    jd_preferred = [kw for kw in preferred_keywords if _expand_keyword_aliases(kw) & jd_tokens]

    matched_required = [kw for kw in jd_required if _expand_keyword_aliases(kw) & resume_tokens]
    missing_required = [kw for kw in jd_required if not (_expand_keyword_aliases(kw) & resume_tokens)]
    matched_preferred = [kw for kw in jd_preferred if _expand_keyword_aliases(kw) & resume_tokens]
    missing_preferred = [kw for kw in jd_preferred if not (_expand_keyword_aliases(kw) & resume_tokens)]

    required_match_rate = round(
        (len(matched_required) / max(len(jd_required), 1)) * 100.0,
        2,
    )
    preferred_match_rate = round(
        (len(matched_preferred) / max(len(jd_preferred), 1)) * 100.0,
        2,
    ) if jd_preferred else 0.0

    payload = {
        "target_role": target_role,
        "jd_required_keywords": jd_required,
        "jd_preferred_keywords": jd_preferred,
        "matched_required_keywords": matched_required,
        "missing_required_top": missing_required[:5],
        "matched_preferred_keywords": matched_preferred,
        "missing_preferred_top": missing_preferred[:5],
        "required_match_rate": required_match_rate,
        "preferred_match_rate": preferred_match_rate,
    }
    return json.dumps(payload, ensure_ascii=False)
