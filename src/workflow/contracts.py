"""Request/response contracts used across API/UI/CLI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    user_query: str
    target_role: str = Field(default="백엔드 개발자")
    resume_text: str = Field(default="")


class ChatResponse(BaseModel):
    session_id: str
    summary: str
    resume_improvements: list[str]
    interview_preparation: list[str]
    two_week_plan: list[str]
    references: list[str]
