"""Request/response contracts used across API/UI/CLI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    user_query: str
    target_role: str = Field(default="백엔드 개발자")
    resume_text: str = Field(default="")
    jd_text: str = Field(default="")


class ChatResponse(BaseModel):
    session_id: str
    summary: str
    resume_improvements: list[str] = Field(default_factory=list)
    interview_preparation: list[str] = Field(default_factory=list)
    two_week_plan: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    route: str | None = None
    routing_reason: str | None = None
    rag_low_confidence: bool | None = None
    cached_state_hit: bool = False