"""CLI entry point for JobPilot AI."""

from __future__ import annotations

import argparse
import json
import sys

from src.common import JobPilotError
from src.workflow import ChatRequest, JobPilotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JobPilot AI from CLI")
    parser.add_argument("--session-id", default="cli-default")
    parser.add_argument("--target-role", default="백엔드 개발자")
    parser.add_argument("--query", required=True, help="User request/question")
    parser.add_argument("--resume-text", default="", help="Optional resume plain text")
    parser.add_argument(
        "--jd-text",
        default="",
        help="Optional job posting/JD plain text for gap analysis",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        service = JobPilotService()
        response = service.run(
            ChatRequest(
                session_id=args.session_id,
                user_query=args.query,
                target_role=args.target_role,
                resume_text=args.resume_text,
                jd_text=args.jd_text,
            )
        )
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    except JobPilotError as exc:
        print(json.dumps(exc.to_payload(), ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
