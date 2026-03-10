"""Reusable prompt templates and policy lines for workflow nodes."""

from __future__ import annotations

PROMPT_RULE_KOREAN_ONLY = "중요: 답변은 반드시 한국어로만 작성하세요."
PROMPT_RULE_NO_COT = "중요: 생각 과정을 노출하지 말고 결과만 제시하세요."
PROMPT_RULE_PLAN_CITATION = "중요: 근거 번호 citation([1][2])을 계획 항목 끝에 표기하세요."
PROMPT_RULE_NO_GUARANTEE = (
    "중요: 합격 확률/결과를 보장하거나 단정하지 말고, 조건부 표현과 근거 기반 코칭으로 작성하세요."
)

PROMPT_RULE_RESULT_ONLY = "생각 과정을 노출하지 말고 결과만 작성하세요."
PROMPT_RULE_ALL_FIELDS_KOREAN = "모든 필드는 한국어로 작성하세요."

TOOL_LOOP_FINALIZATION_INSTRUCTION = (
    "위 대화와 도구 결과를 종합하여 최종 결과를 작성하세요. "
    "더 이상 도구를 호출하지 말고, 생각 과정을 노출하지 말고 결과만 작성하세요. "
    "반드시 한국어로 작성하세요."
)

STRUCTURED_JSON_REPAIR_INSTRUCTION = (
    "직전 응답의 구조화 파싱이 실패했습니다. "
    "스키마 필드를 빠짐없이 채우되, 생각 과정 없이 결과만 한국어 JSON으로 다시 작성하세요."
)

ROUTE_NAME_DESCRIPTIONS = {
    "resume_only": "이력서 개선 중심",
    "interview_only": "면접 대비 중심",
    "full": "이력서 + 면접 + 통합 실행",
    "plan_only": "종합 실행 계획 위주(간단 조언)",
}

RESUME_ONLY_MARKERS = (
    "이력서만",
    "이력서 개선만",
    "이력서만 봐",
    "이력서 위주로",
    "이력서 중심으로",
    "이력서 개선 포인트만",
)
INTERVIEW_ONLY_MARKERS = (
    "면접만",
    "면접 질문만",
    "면접 준비만",
)
PLAN_ONLY_MARKERS = (
    "계획만",
    "플랜만",
    "2주 계획만",
    "실행계획만",
    "실행 계획만",
    "실행 계획 위주로",
    "실행계획 위주로",
    "전체 요약 없이",
)

EXCLUSION_PATTERNS = ("제외", "빼", "빼줘", "제외해")
NEGATION_PATTERNS = ("말고", "말아", "말자", "않", "아니", "원치 않", "싶지 않", "필요 없")

EXCLUSION_TERMS = {
    "resume": ("이력서", "자소서", "포트폴리오"),
    "interview": ("면접", "질문"),
    "plan": ("플랜", "계획", "2주 계획", "실행계획"),
}

INTENT_KEYWORDS = {
    "resume": ("이력서", "자소서", "포트폴리오"),
    "interview": ("면접", "질문", "답변"),
    "plan": ("계획", "플랜", "로드맵", "2주"),
}

SUPERVISOR_ROUTING_RULE_LINES = (
    '사용자가 제외 요청을 명시(예: "면접 제외", "계획 제외", "이력서 제외")하면 해당 제외 의도를 강하게 반영하라.',
    "이력서 텍스트가 미제공이면 resume_only는 가능한 한 피하라.",
    "이력서 텍스트가 미제공이고 요청이 이력서 중심이면 plan_only 또는 full 중 더 안전한 쪽을 선택하라.",
    "면접 대비 요청이 명확하면 interview_only를 우선 검토하라.",
    "JD/공고 텍스트가 제공된 경우, 공고-이력서 비교 요청은 resume_only 또는 full을 우선 검토하라.",
)


def build_common_policy_block(*, include_plan_citation: bool = False) -> str:
    lines = [PROMPT_RULE_KOREAN_ONLY, PROMPT_RULE_NO_COT, PROMPT_RULE_NO_GUARANTEE]
    if include_plan_citation:
        lines.append(PROMPT_RULE_PLAN_CITATION)
    return "\n".join(lines)


def build_specialist_system_prompt(agent_name: str, focus: str) -> str:
    return (
        f"당신은 {agent_name}입니다. {focus} "
        "출력은 반드시 한국어로, 스키마를 준수한 구조화 결과만 반환하세요."
    )


def build_route_options_block() -> str:
    lines = ["선택 가능한 route:"]
    for route, desc in ROUTE_NAME_DESCRIPTIONS.items():
        lines.append(f"- {route}: {desc}")
    return "\n".join(lines)


def build_supervisor_routing_rules_block() -> str:
    lines = ["[라우팅 규칙]"]
    lines.extend(f"- {item}" for item in SUPERVISOR_ROUTING_RULE_LINES)
    lines.append(
        "- 용어 정의(휴리스틱/LLM 공통): "
        f"resume={INTENT_KEYWORDS['resume']}, interview={INTENT_KEYWORDS['interview']}, plan={INTENT_KEYWORDS['plan']}"
    )
    return "\n".join(lines)

