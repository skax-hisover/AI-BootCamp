"""Structured output schemas for final response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResumeNotes(BaseModel):
    key_findings: list[str] = Field(description="핵심 진단 요약")
    improvement_points: list[str] = Field(description="우선순위 이력서 개선 항목")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")


class InterviewNotes(BaseModel):
    expected_questions: list[str] = Field(description="예상 면접 질문")
    answer_guides: list[str] = Field(description="질문별 답변 방향")
    avoid_patterns: list[str] = Field(description="피해야 할 답변 패턴")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")


class FinalAnswer(BaseModel):
    summary: str = Field(description="요청에 대한 핵심 요약")
    resume_improvements: list[str] = Field(description="이력서 개선 액션 아이템")
    interview_preparation: list[str] = Field(description="면접 준비 액션 아이템")
    two_week_plan: list[str] = Field(description="2주 실행 계획")
    references: list[str] = Field(description="참고한 문서/근거 출처")
