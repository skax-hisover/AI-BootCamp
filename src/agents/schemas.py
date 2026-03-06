"""Structured output schemas for final response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResumeNotes(BaseModel):
    key_findings: list[str] = Field(description="핵심 진단 요약")
    improvement_points: list[str] = Field(description="우선순위 이력서 개선 항목")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="개선 항목/진단 문장을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class InterviewNotes(BaseModel):
    expected_questions: list[str] = Field(description="예상 면접 질문")
    answer_guides: list[str] = Field(description="질문별 답변 방향")
    avoid_patterns: list[str] = Field(description="피해야 할 답변 패턴")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="질문/답변 항목을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class PlanNotes(BaseModel):
    priorities: list[str] = Field(description="우선순위 기반 실행 항목")
    weekly_schedule: list[str] = Field(description="주차/일정 단위 실행 계획")
    validation_checks: list[str] = Field(description="검증 방법 및 체크포인트")
    evidence_snippets: list[str] = Field(description="RAG 근거 문장 또는 근거 요약")
    evidence_map: dict[str, list[int]] = Field(
        default_factory=dict,
        description="계획 항목을 근거 chunk 번호([1],[2]...)와 매핑",
    )


class ReferenceItem(BaseModel):
    rank: int = Field(description="검색 결과 순위")
    source: str = Field(description="출처 파일명")
    chunk_id: int | str | None = Field(default=None, description="원본 문서 내 청크 ID")
    location: str = Field(default="n/a", description="원본 문서 위치")
    score: float = Field(default=0.0, description="검색 점수 (0~1)")
    category: str | None = Field(default=None, description="문서 카테고리")
    snippet: str = Field(default="", description="청크 내용 요약")


class FinalAnswer(BaseModel):
    summary: str = Field(description="요청에 대한 핵심 요약")
    resume_improvements: list[str] = Field(
        default_factory=list,
        description="이력서 개선 액션 아이템(라우트에 따라 비워질 수 있음)",
    )
    interview_preparation: list[str] = Field(
        default_factory=list,
        description="면접 준비 액션 아이템(라우트에 따라 비워질 수 있음)",
    )
    two_week_plan: list[str] = Field(
        default_factory=list,
        description="2주 실행 계획(라우트에 따라 비워질 수 있음)",
    )
    input_gap_notice: str | None = Field(
        default=None,
        description="입력 정보 부족 시 사용자에게 추가 제공을 요청하는 안내 문구",
    )
    references: list[ReferenceItem] = Field(
        default_factory=list,
        description="참고한 문서/근거 출처 구조화 목록",
    )
