"""Structured output schemas for final response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    summary: str = Field(description="요청에 대한 핵심 요약")
    resume_improvements: list[str] = Field(description="이력서 개선 액션 아이템")
    interview_preparation: list[str] = Field(description="면접 준비 액션 아이템")
    two_week_plan: list[str] = Field(description="2주 실행 계획")
    references: list[str] = Field(description="참고한 문서/근거 출처")
