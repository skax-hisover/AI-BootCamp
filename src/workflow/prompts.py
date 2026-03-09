"""Reusable prompt templates and policy lines for workflow nodes."""

from __future__ import annotations

PROMPT_RULE_KOREAN_ONLY = "중요: 답변은 반드시 한국어로만 작성하세요."
PROMPT_RULE_NO_COT = "중요: 생각 과정을 노출하지 말고 결과만 제시하세요."
PROMPT_RULE_PLAN_CITATION = "중요: 근거 번호 citation([1][2])을 계획 항목 끝에 표기하세요."

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


def build_common_policy_block(*, include_plan_citation: bool = False) -> str:
    lines = [PROMPT_RULE_KOREAN_ONLY, PROMPT_RULE_NO_COT]
    if include_plan_citation:
        lines.append(PROMPT_RULE_PLAN_CITATION)
    return "\n".join(lines)


def build_specialist_system_prompt(agent_name: str, focus: str) -> str:
    return (
        f"당신은 {agent_name}입니다. {focus} "
        "출력은 반드시 한국어로, 스키마를 준수한 구조화 결과만 반환하세요."
    )

