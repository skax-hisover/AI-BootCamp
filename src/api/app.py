"""FastAPI app for JobPilot service."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.common import JobPilotError
from src.workflow import ChatRequest, JobPilotService


@lru_cache(maxsize=1)
def get_service() -> JobPilotService:
    return JobPilotService()


app = FastAPI(title="JobPilot AI API", version="1.0.0")


@app.exception_handler(JobPilotError)
def handle_jobpilot_error(_: Request, exc: JobPilotError) -> JSONResponse:
    # Keep error contract flat for API clients.
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_payload(),
    )


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": f"JobPilot service failed: {exc}",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    service = get_service()
    return service.run(req).model_dump()
