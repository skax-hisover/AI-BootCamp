"""Generate submission evidence files for Step2/Step3 documents.

This script creates:
1) Agent execution log with routing + RAG evidence
2) Final structured response JSON

Streamlit UI screenshot is intentionally collected manually by the user because
it requires an interactive browser view.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow import JobPilotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission evidence artifacts")
    parser.add_argument(
        "--query",
        default="백엔드 이직 준비를 위한 이력서 개선 포인트와 2주 계획을 제시해줘.",
        help="User query to run for evidence generation",
    )
    parser.add_argument("--target-role", default="백엔드 개발자")
    parser.add_argument(
        "--resume-text",
        default="Python/FastAPI 기반 API 개발 경험 2년, 프로젝트 협업 경험 보유",
    )
    parser.add_argument(
        "--jd-text",
        default="백엔드 개발자 채용: Python/FastAPI, RDBMS 튜닝, 장애 대응 경험, 협업 커뮤니케이션",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/evidence",
        help="Directory where artifacts are saved",
    )
    parser.add_argument("--session-id", default="evidence-session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    service = JobPilotService()
    state = {
        "session_id": args.session_id,
        "user_query": args.query,
        "target_role": args.target_role,
        "resume_text": args.resume_text,
        "jd_text": args.jd_text,
    }
    result = service.graph.invoke(
        state,
        config={"configurable": {"thread_id": args.session_id}},
    )
    final_answer = result.get("final_answer", {})

    json_path = output_dir / "agent_final_answer.json"
    json_path.write_text(json.dumps(final_answer, ensure_ascii=False, indent=2), encoding="utf-8")

    route = result.get("route", "unknown")
    routing_reason = result.get("routing_reason", "unknown")
    rag_refs = result.get("rag_refs", [])
    rag_context = result.get("rag_context", "")
    rag_preview = rag_context[:800] + ("..." if len(rag_context) > 800 else "")

    rag_lines: list[str] = []
    for ref in rag_refs if isinstance(rag_refs, list) else []:
        if isinstance(ref, dict):
            rag_lines.append(
                f"- [{ref.get('rank', '?')}] {ref.get('source', 'unknown')} "
                f"(chunk={ref.get('chunk_id', 'na')}, location={ref.get('location', 'n/a')}, score={ref.get('score', 0.0)})"
            )
        else:
            rag_lines.append(f"- {ref}")
    rag_refs_text = "\n".join(rag_lines) if rag_lines else "- (없음)"

    log_path = output_dir / "agent_execution_log.md"
    log_text = f"""# Agent Execution Log

- Generated at: {timestamp}
- Session ID: {args.session_id}
- Target role: {args.target_role}

## Input
- Query: {args.query}
- Resume text length: {len(args.resume_text)}
- JD text length: {len(args.jd_text)}

## Supervisor Routing
- Route: {route}
- Reason: {routing_reason}

## RAG References
{rag_refs_text}

## RAG Context Preview
```
{rag_preview}
```

## Final Answer Artifact
- JSON file: `{json_path.as_posix()}`
"""
    log_path.write_text(log_text, encoding="utf-8")

    print(f"[OK] Created: {log_path.as_posix()}")
    print(f"[OK] Created: {json_path.as_posix()}")


if __name__ == "__main__":
    main()

