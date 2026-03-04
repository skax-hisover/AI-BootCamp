"""FastAPI app for JobPilot service."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from src.common import JobPilotError
from src.workflow import ChatRequest, JobPilotService


@lru_cache(maxsize=1)
def get_service() -> JobPilotService:
    return JobPilotService()


app = FastAPI(title="JobPilot AI API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        service = get_service()
        return service.run(req).model_dump()
    except JobPilotError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_payload()) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INTERNAL_SERVER_ERROR",
                "detail": f"JobPilot service failed: {exc}",
            },
        ) from exc
