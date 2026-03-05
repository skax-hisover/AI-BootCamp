"""Run automatic checks for routing accuracy / reference inclusion / plan quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    AnswerQualitySample,
    RoutingSampleResult,
    plan_quality_rate,
    reference_inclusion_rate,
    routing_accuracy,
)
from src.workflow import JobPilotService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate differentiation metrics")
    parser.add_argument(
        "--cases",
        default="data/eval/sample_queries.json",
        help="JSON file path containing evaluation queries",
    )
    parser.add_argument("--session-prefix", default="eval-session")
    parser.add_argument("--min-routing-accuracy", type=float, default=0.5)
    parser.add_argument("--min-reference-rate", type=float, default=0.8)
    parser.add_argument("--min-plan-quality-rate", type=float, default=0.8)
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. When set, writes the full report with UTF-8 encoding.",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation cases JSON must be a list.")
    return [item for item in raw if isinstance(item, dict)]


def _requires_plan(route: str) -> bool:
    route_key = (route or "").strip().lower()
    return route_key in {"full", "plan_only"}


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    service = JobPilotService()
    cases = _load_cases(cases_path)
    routing_samples: list[RoutingSampleResult] = []
    reference_samples: list[AnswerQualitySample] = []
    plan_quality_samples: list[AnswerQualitySample] = []

    details: list[dict] = []
    for i, case in enumerate(cases, start=1):
        session_id = f"{args.session_prefix}-{i}"
        state = {
            "session_id": session_id,
            "user_query": str(case.get("query", "")),
            "target_role": str(case.get("target_role", "백엔드 개발자")),
            "resume_text": str(case.get("resume_text", "")),
            "jd_text": str(case.get("jd_text", "")),
        }
        result = service.graph.invoke(
            state,
            config={"configurable": {"thread_id": session_id}},
        )
        final_answer = result.get("final_answer", {}) if isinstance(result, dict) else {}
        refs = final_answer.get("references", []) if isinstance(final_answer, dict) else []
        plans = final_answer.get("two_week_plan", []) if isinstance(final_answer, dict) else []
        references = refs if isinstance(refs, list) else []
        two_week_plan = plans if isinstance(plans, list) else []

        predicted_route = str(result.get("route", "unknown"))
        expected_route = str(case.get("expected_route", ""))
        eval_route = (expected_route or predicted_route).strip().lower()
        if expected_route:
            routing_samples.append(
                RoutingSampleResult(expected_route=expected_route, predicted_route=predicted_route)
            )
        sample = AnswerQualitySample(
            references=[str(item) for item in references],
            two_week_plan=[str(item) for item in two_week_plan],
        )
        reference_samples.append(sample)
        if _requires_plan(eval_route):
            plan_quality_samples.append(sample)
        details.append(
            {
                "query": state["user_query"],
                "expected_route": expected_route,
                "predicted_route": predicted_route,
                "eval_route": eval_route,
                "references_count": len(references),
                "plan_items_count": len(two_week_plan),
            }
        )

    routing = routing_accuracy(routing_samples) if routing_samples else 0.0
    ref_rate = reference_inclusion_rate(reference_samples)
    plan_rate = plan_quality_rate(plan_quality_samples)

    report_lines: list[str] = []
    report_lines.append("=== Differentiation Metrics ===")
    report_lines.append(f"Cases: {len(cases)}")
    if routing_samples:
        report_lines.append(f"Routing accuracy: {routing:.2%}")
    else:
        report_lines.append("Routing accuracy: n/a (expected_route not provided)")
    report_lines.append(f"Reference inclusion rate: {ref_rate:.2%}")
    report_lines.append(f"Plan quality rate (full/plan_only): {plan_rate:.2%}")
    report_lines.append("")
    report_lines.append("=== Case Details ===")
    for item in details:
        report_lines.append(json.dumps(item, ensure_ascii=False))

    if routing_samples and routing < args.min_routing_accuracy:
        report_lines.append(
            f"[FAIL] routing accuracy {routing:.2%} < {args.min_routing_accuracy:.2%}"
        )
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])
    if ref_rate < args.min_reference_rate:
        report_lines.append(f"[FAIL] reference inclusion rate {ref_rate:.2%} < {args.min_reference_rate:.2%}")
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])
    if plan_rate < args.min_plan_quality_rate:
        report_lines.append(f"[FAIL] plan quality rate {plan_rate:.2%} < {args.min_plan_quality_rate:.2%}")
        _emit_report(report_lines, args.output)
        raise SystemExit(report_lines[-1])

    report_lines.append("[PASS] All thresholds satisfied.")
    _emit_report(report_lines, args.output)


def _emit_report(lines: list[str], output_path: str) -> None:
    text = "\n".join(lines)
    print(text)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

