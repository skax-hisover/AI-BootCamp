"""LangGraph multi-agent workflow for JobPilot AI."""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents.schemas import FinalAnswer, InterviewNotes, PlanNotes, ResumeNotes
from src.agents.tools import interview_question_bank, resume_keyword_match_score
from src.common import JobPilotError
from src.config import get_chat_model, load_settings
from src.retrieval import HybridRetriever, SearchHit
from src.utils.memory import SessionMemory
from src.workflow.contracts import ChatRequest, ChatResponse


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    target_role: str
    resume_text: str
    jd_text: str
    memory_messages: list[dict[str, str]]
    rag_context: str
    rag_refs: list[str]
    rag_low_confidence: bool
    route: str
    routing_reason: str
    resume_notes: dict[str, Any]
    interview_notes: dict[str, Any]
    plan_notes: dict[str, Any]
    final_answer: dict[str, Any]


class RouteDecision(BaseModel):
    route: Literal["resume_only", "interview_only", "full", "plan_only"] = Field(
        description="Supervisor가 선택한 라우팅 결과"
    )
    reason: str = Field(description="해당 라우팅을 선택한 이유")


class RagPlan(BaseModel):
    rewritten_queries: list[str] = Field(default_factory=list, description="재검색용 쿼리 목록")
    source_hint: str = Field(
        default="", description="우선 확인할 문서/키워드 힌트(예: backend, data, pm)"
    )


TStructured = TypeVar("TStructured", bound=BaseModel)


def _run_tool_loop_structured(
    prompt: str,
    tools: list[BaseTool],
    output_schema: type[TStructured],
    max_steps: int = 4,
    max_tool_rounds: int = 2,
) -> TStructured:
    llm = get_chat_model(temperature=0.2).bind_tools(tools)
    messages = [HumanMessage(content=prompt)]
    seen_calls: set[str] = set()
    tool_round_count = 0
    for _ in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_round_count += 1
        if tool_round_count > max_tool_rounds:
            messages.append(
                HumanMessage(
                    content=(
                        "도구 호출 라운드가 충분히 수행되었습니다. "
                        "이제 도구 호출을 중단하고 최종 결과만 작성하세요."
                    )
                )
            )
            break

        repeated_detected = False

        for call in response.tool_calls:
            matched_tool = next((tool for tool in tools if tool.name == call["name"]), None)
            args = call.get("args", {})
            signature = f"{call['name']}::{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            if not matched_tool:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {call['name']}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue
            if signature in seen_calls:
                repeated_detected = True
                messages.append(
                    ToolMessage(
                        content="Repeated tool+args call detected. Execution skipped.",
                        tool_call_id=call["id"],
                        name=matched_tool.name,
                    )
                )
                continue
            seen_calls.add(signature)

            output = matched_tool.invoke(args)
            messages.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=call["id"],
                    name=matched_tool.name,
                )
            )
        if repeated_detected:
            messages.append(
                HumanMessage(
                    content=(
                        "동일한 도구 호출이 반복 감지되었습니다. "
                        "추가 도구 호출 없이 최종 결과를 정리하세요."
                    )
                )
            )
            break

    parser = get_chat_model(temperature=0.2).with_structured_output(output_schema)
    return parser.invoke(
        [
            *messages,
            HumanMessage(
                content=(
                    "위 대화와 도구 결과를 종합하여 최종 결과를 작성하세요. "
                    "더 이상 도구를 호출하지 말고, 생각 과정을 노출하지 말고 결과만 작성하세요. "
                    "반드시 한국어로 작성하세요."
                )
            ),
        ],
    )


def build_graph(retriever: HybridRetriever, memory: SessionMemory):
    llm = get_chat_model(temperature=0.1)
    resume_few_shot = """
[Few-shot 예시 1]
입력:
- 목표 직무: 백엔드 개발자
- 이력서 요약: "Spring 기반 API 개발, 성능 개선 경험 없음"
출력 스타일:
- 개선 포인트:
  1) 성능 개선 지표(응답시간/처리량) 추가
  2) DB 튜닝 사례(인덱스/쿼리 최적화) 명시
  3) 장애 대응 경험과 재발 방지 액션 포함

[Few-shot 예시 2]
입력:
- 목표 직무: 데이터 분석가
- 이력서 요약: "대시보드 제작 경험 위주"
출력 스타일:
- 개선 포인트:
  1) 문제 정의 -> 분석 -> 인사이트 -> 비즈니스 임팩트 흐름으로 재작성
  2) SQL/Python 사용 범위와 자동화 범위 구체화
  3) 지표 개선 수치(예: 전환율 12% 상승) 추가
""".strip()

    interview_few_shot = """
[Few-shot 예시 1]
입력: 백엔드 개발자 면접 준비
출력 스타일:
- 예상 질문: 대규모 트래픽 환경의 병목 해결 경험?
- 답변 방향: 병목 식별 -> 대안 비교 -> 적용 결과 수치
- 피해야 할 패턴: "그냥 캐시 썼다" 식의 근거 없는 답변

[Few-shot 예시 2]
입력: PM 면접 준비
출력 스타일:
- 예상 질문: 우선순위 충돌 상황 의사결정 사례?
- 답변 방향: 목표 지표 -> 이해관계자 조율 -> 결과/회고
- 피해야 할 패턴: 개인 의견만 강조하고 데이터/지표 근거 누락
""".strip()

    def _format_resume_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("key_findings", []):
            lines.append(f"- 핵심 진단: {item}")
        for item in notes.get("improvement_points", []):
            lines.append(f"- 개선 포인트: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _format_interview_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("expected_questions", []):
            lines.append(f"- 예상 질문: {item}")
        for item in notes.get("answer_guides", []):
            lines.append(f"- 답변 방향: {item}")
        for item in notes.get("avoid_patterns", []):
            lines.append(f"- 피해야 할 패턴: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _format_plan_notes(notes: dict[str, Any]) -> str:
        if not notes:
            return "없음"
        lines: list[str] = []
        for item in notes.get("priorities", []):
            lines.append(f"- 우선순위: {item}")
        for item in notes.get("weekly_schedule", []):
            lines.append(f"- 일정: {item}")
        for item in notes.get("validation_checks", []):
            lines.append(f"- 검증: {item}")
        for item in notes.get("evidence_snippets", []):
            lines.append(f"- 근거: {item}")
        evidence_map = notes.get("evidence_map", {})
        if isinstance(evidence_map, dict) and evidence_map:
            lines.append(f"- 근거 매핑: {evidence_map}")
        return "\n".join(lines) if lines else str(notes)

    def _fallback_resume_notes(state: AgentState, reason: str) -> dict[str, Any]:
        return {
            "key_findings": [
                "구조화 파싱 실패로 최소 진단 결과로 대체되었습니다.",
                reason,
            ],
            "improvement_points": [
                "핵심 역량 3~5개를 상단 요약에 명시",
                "프로젝트를 문제-접근-결과 구조로 재작성",
                "정량 지표(성능/처리량/품질)를 문장에 포함",
                "목표 직무 공고 키워드와 일치율을 높이도록 수정",
            ],
            "evidence_snippets": [state.get("rag_context", "")[:300] or "RAG 근거 없음"],
            "evidence_map": {},
        }

    def _fallback_interview_notes(state: AgentState, reason: str) -> dict[str, Any]:
        return {
            "expected_questions": [
                "해당 직무에서 본인이 가장 잘한 문제 해결 사례는?",
                "프로젝트에서 성과를 수치로 설명할 수 있는가?",
                "협업 갈등 상황을 어떻게 해결했는가?",
                "우선순위 판단 기준은 무엇이었는가?",
            ],
            "answer_guides": [
                "상황-행동-결과 순서로 간결하게 답변",
                "수치/지표를 포함해 신뢰도를 높임",
                "본인 기여 범위를 명확히 구분",
                "회고와 재발 방지 관점까지 포함",
            ],
            "avoid_patterns": [
                "근거 없는 단정형 답변",
                "팀 성과를 본인 성과처럼 과장",
                "기술 나열만 하고 문제 맥락 누락",
                "질문 의도와 무관한 장황한 설명",
            ],
            "evidence_snippets": [
                reason,
                state.get("rag_context", "")[:300] or "RAG 근거 없음",
            ],
            "evidence_map": {},
        }

    def _fallback_plan_notes(state: AgentState, reason: str) -> dict[str, Any]:
        return {
            "priorities": [
                "JD 핵심 역량과 현재 준비 상태의 격차가 큰 항목부터 우선 실행",
                "이력서/포트폴리오 고도화와 면접 리허설을 병행",
                "주 단위 산출물(문서/답변 스크립트) 중심으로 관리",
            ],
            "weekly_schedule": [
                "1주차: JD 키워드 정렬, 이력서 핵심 불릿 재작성, 프로젝트 성과 수치 보강",
                "2주차: 예상 질문 답변 스크립트 완성, 모의 면접 2회, 피드백 반영",
            ],
            "validation_checks": [
                "이력서 항목별로 JD 요구역량 매칭 여부 체크",
                "모의 면접 후 약점 질문 재학습 여부 확인",
                "지원 전 체크리스트 완료율 점검",
            ],
            "evidence_snippets": [
                reason,
                state.get("rag_context", "")[:300] or "RAG 근거 없음",
            ],
            "evidence_map": {},
        }

    def _route_minimums(route: str) -> dict[str, int]:
        route_key = (route or "full").lower()
        if route_key == "resume_only":
            return {
                "resume_improvements": 4,
                "interview_preparation": 0,
                "two_week_plan": 4,
            }
        if route_key == "interview_only":
            return {
                "resume_improvements": 0,
                "interview_preparation": 4,
                "two_week_plan": 4,
            }
        if route_key == "plan_only":
            return {
                "resume_improvements": 0,
                "interview_preparation": 0,
                "two_week_plan": 4,
            }
        return {
            "resume_improvements": 4,
            "interview_preparation": 4,
            "two_week_plan": 4,
        }

    def _normalize_final_answer_by_route(route: str, answer: dict[str, Any]) -> dict[str, Any]:
        route_key = (route or "full").lower()
        payload = {
            "summary": str(answer.get("summary", "")),
            "resume_improvements": list(answer.get("resume_improvements", []) or []),
            "interview_preparation": list(answer.get("interview_preparation", []) or []),
            "two_week_plan": list(answer.get("two_week_plan", []) or []),
            "references": list(answer.get("references", []) or []),
        }
        if route_key in {"interview_only", "plan_only"}:
            payload["resume_improvements"] = []
        if route_key in {"resume_only", "plan_only"}:
            payload["interview_preparation"] = []
        return payload

    def _fallback_final_answer(state: AgentState, reason: str) -> dict[str, Any]:
        refs = state.get("rag_refs") or []
        references = refs[:4] if isinstance(refs, list) else []
        if not references:
            references = ["RAG 근거 부족: 일반 가이드 기반 결과"]
        return {
            "summary": f"일부 구조화 단계 실패로 안전 모드 응답을 제공합니다. ({reason})",
            "resume_improvements": [
                "직무 키워드를 반영해 이력서 요약을 재작성",
                "프로젝트 성과를 수치 중심으로 보강",
                "기술 스택보다 문제 해결 흐름을 강조",
                "JD와 불일치하는 표현을 정리",
            ],
            "interview_preparation": [
                "직무 핵심 질문 10개를 선정해 답변 초안 작성",
                "STAR 구조로 답변 리허설 진행",
                "수치/지표 근거를 포함한 답변으로 보완",
                "모의 면접 후 취약 질문 재학습",
            ],
            "two_week_plan": [
                "1주차: 이력서 핵심 수정 및 공고 키워드 정렬",
                "1주차: 프로젝트 사례 2개를 성과 중심으로 재정리",
                "2주차: 면접 질문별 답변 스크립트 완성",
                "2주차: 모의 면접 및 피드백 반영",
            ],
            "references": references,
        }

    def _route_by_supervisor(state: AgentState) -> AgentState:
        router = llm.with_structured_output(RouteDecision)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        ) or "없음"
        resume_text = (state.get("resume_text") or "").strip()
        jd_text = (state.get("jd_text") or "").strip()
        try:
            decision = router.invoke(
                f"""
너는 Supervisor Planner다. 사용자 요청을 보고 실행 라우트를 고른다.

선택 가능한 route:
- resume_only: 이력서 개선 중심
- interview_only: 면접 대비 중심
- full: 이력서 + 면접 + 통합 실행
- plan_only: 종합 실행 계획 위주(간단 조언)

[사용자 요청]
{state['user_query']}

[목표 직무]
{state['target_role']}

[JD/공고 텍스트 제공 여부]
{"제공됨" if jd_text else "미제공"}

[JD/공고 텍스트 길이]
{len(jd_text)}

[이력서 텍스트 제공 여부]
{"제공됨" if resume_text else "미제공"}

[이력서 텍스트 길이]
{len(resume_text)}

[최근 메모리]
{history_text}

[라우팅 규칙]
- 이력서 텍스트가 미제공이면 resume_only는 가능한 한 피하라.
- 이력서 텍스트가 미제공이고 요청이 이력서 중심이면 plan_only 또는 full 중 더 안전한 쪽을 선택하라.
- 면접 대비 요청이 명확하면 interview_only를 우선 검토하라.
- JD/공고 텍스트가 제공된 경우, 공고-이력서 비교 요청은 resume_only 또는 full을 우선 검토하라.
"""
            )
            route = decision.route
            reason = decision.reason
            if not resume_text and route == "resume_only":
                route = "plan_only"
                reason = (
                    f"{reason} / 이력서 텍스트가 없어 resume_only 대신 plan_only로 조정"
                )
            return {"route": route, "routing_reason": reason}
        except Exception:
            return {"route": "full", "routing_reason": "파싱 실패로 full 라우트 기본 적용"}

    def supervisor_node(state: AgentState) -> AgentState:
        base_state: AgentState = {
            "memory_messages": memory.get(state["session_id"], limit=8),
        }
        base_state.update(_route_by_supervisor(base_state | state))
        return base_state

    def _infer_role_hint(target_role: str) -> str:
        role = target_role.lower()
        if "백엔드" in target_role or "backend" in role:
            return "backend, api, server, database"
        if "데이터" in target_role or "data" in role:
            return "data, sql, analytics, dashboard"
        if "pm" in role or "기획" in target_role or "product" in role:
            return "pm, product, roadmap, metric"
        return target_role

    def _is_role_relevant(text: str, hint: str) -> bool:
        keywords = [token.strip().lower() for token in hint.split(",") if token.strip()]
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords) if keywords else True

    def _category_filter_for_route(route: str) -> set[str] | None:
        route_key = (route or "full").lower()
        if route_key == "resume_only":
            return {"job_postings", "jd", "portfolio_examples", "resume_examples"}
        if route_key == "interview_only":
            return {"interview_guides", "job_postings", "jd"}
        if route_key == "plan_only":
            return {"job_postings", "jd", "interview_guides"}
        return None

    def _rerank_hits(
        hits: list[SearchHit],
        query: str,
        role_hint: str,
        route: str,
        top_k: int = 4,
    ) -> list[SearchHit]:
        """
        Dedicated rerank layer.
        This function is an extension point for future rerankers
        (e.g., cross-encoder / LLM-based rerank) without touching retriever.search().
        """
        if not hits:
            return []
        query_lower = query.lower()
        route_categories = _category_filter_for_route(route)
        rescored: list[SearchHit] = []
        for hit in hits:
            category = str(hit.metadata.get("category", "")).lower()
            role_boost = (
                0.15 if _is_role_relevant(f"{hit.source} {hit.content}", role_hint) else 0.0
            )
            category_boost = 0.08 if route_categories and category in route_categories else 0.0
            lexical_boost = 0.05 if any(token in hit.content.lower() for token in query_lower.split()) else 0.0
            rescored.append(
                SearchHit(
                    content=hit.content,
                    source=hit.source,
                    score=round(hit.score + role_boost + category_boost + lexical_boost, 4),
                    metadata=hit.metadata,
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    def rag_node(state: AgentState) -> AgentState:
        planner = llm.with_structured_output(RagPlan)
        settings = load_settings()
        role_hint = _infer_role_hint(state["target_role"])
        route = state.get("route", "full")
        retrieval_top_k = 2 if route == "plan_only" else 4
        query_limit = 2 if route == "plan_only" else 3
        jd_text = (state.get("jd_text") or "").strip()
        category_filter = _category_filter_for_route(route)
        try:
            rag_plan = planner.invoke(
                f"""
너는 RAG Agent다. 검색 품질을 높이기 위해 쿼리를 2~3개로 재작성하라.
route={route}
user_query={state['user_query']}
target_role={state['target_role']}
jd_text_present={"yes" if jd_text else "no"}
jd_text_preview={jd_text[:400] if jd_text else "none"}
source_hint는 직무 연관 키워드로 짧게 작성하라.
"""
            )
            query_candidates = [state["user_query"], *rag_plan.rewritten_queries]
            source_hint = rag_plan.source_hint or role_hint
        except Exception:
            query_candidates = [state["user_query"], f"{state['target_role']} {state['user_query']}"]
            source_hint = role_hint

        merged_hits: dict[str, SearchHit] = {}
        for candidate in query_candidates[:query_limit]:
            for hit in retriever.search(
                candidate,
                top_k=retrieval_top_k,
                category_filter=category_filter,
            ):
                key = f"{hit.source}::{hit.content[:120]}"
                current = merged_hits.get(key)
                if not current or hit.score > current.score:
                    merged_hits[key] = hit

        ranked_hits = _rerank_hits(
            hits=list(merged_hits.values()),
            query=state["user_query"],
            role_hint=source_hint,
            route=route,
            top_k=retrieval_top_k,
        )
        top_score = ranked_hits[0].score if ranked_hits else 0.0
        low_confidence = (not ranked_hits) or (top_score < settings.rag_evidence_score_threshold)
        context_parts: list[str] = []
        refs: list[dict[str, Any]] = []
        for i, hit in enumerate(ranked_hits, start=1):
            context_parts.append(f"[{i}] ({hit.source}, score={hit.score})\n{hit.content}")
            refs.append(
                {
                    "rank": i,
                    "source": hit.source,
                    "score": hit.score,
                    "category": hit.metadata.get("category"),
                }
            )

        if not context_parts:
            context_parts = ["검색 결과가 부족하여 일반적인 직무 가이드를 기반으로 응답합니다."]
        if low_confidence:
            context_parts.append(
                f"[rag-safety] 근거 점수가 임계치({settings.rag_evidence_score_threshold:.2f}) 미만이므로 "
                "보수적 표현과 조건부 제안을 우선하세요."
            )

        ref_names = [
            f"{r['rank']}. {r['source']} (score={r['score']}, category={r.get('category')})"
            for r in refs
        ]
        rag_context = "\n\n".join(context_parts)
        rag_context += (
            f"\n\n[rag-agent-note] route={route}, source_hint={source_hint}, "
            f"category_filter={sorted(category_filter) if category_filter else 'none'}, "
            f"top_score={top_score:.4f}, low_confidence={low_confidence}"
        )
        return {
            "rag_context": rag_context,
            "rag_refs": ref_names,
            "rag_low_confidence": low_confidence,
        }

    def plan_node(state: AgentState) -> AgentState:
        parser = llm.with_structured_output(PlanNotes)
        resume_notes = _format_resume_notes(state.get("resume_notes", {}))
        interview_notes = _format_interview_notes(state.get("interview_notes", {}))
        try:
            structured = parser.invoke(
                f"""
당신은 Plan Agent입니다.
중요: 답변은 반드시 한국어로만 작성하세요.
중요: 생각 과정을 노출하지 말고 결과만 제시하세요.
중요: 근거 번호 citation([1][2])을 계획 항목 끝에 표기하세요.

[사용자 요청]
{state['user_query']}

[목표 직무]
{state['target_role']}

[Supervisor route]
{state.get('route', 'full')}

[JD/공고 텍스트]
{state.get('jd_text', '') or '미제공'}

[Resume Agent 결과]
{resume_notes if state.get('resume_notes') else '이번 라우트에서는 Resume Agent를 생략했습니다.'}

[Interview Agent 결과]
{interview_notes if state.get('interview_notes') else '이번 라우트에서는 Interview Agent를 생략했습니다.'}

[RAG 근거]
{state.get('rag_context', '')}

요구사항:
- priorities, weekly_schedule, validation_checks를 균형 있게 작성
- evidence_map 필수: key는 계획 항목, value는 근거 번호 목록(예: [1,2])
"""
            )
            return {"plan_notes": structured.model_dump()}
        except Exception as exc:
            return {"plan_notes": _fallback_plan_notes(state, f"plan_node fallback: {exc}")}

    def resume_node(state: AgentState) -> AgentState:
        has_resume_text = bool((state.get("resume_text") or "").strip())
        jd_text = (state.get("jd_text") or "").strip()
        prompt = f"""
당신은 Resume Agent입니다.
중요: 답변은 반드시 한국어로만 작성하세요.
중요: 생각 과정을 노출하지 말고 결과만 제시하세요.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}
JD/공고 텍스트:
{jd_text or 'JD/공고 텍스트가 제공되지 않았습니다.'}
사용자 이력서:
{state['resume_text'] or '이력서 텍스트가 제공되지 않았습니다.'}

RAG 근거:
{state['rag_context']}

지시:
1) resume_keyword_match_score 도구를 활용해 키워드 적합도를 반영하세요.
2) 도구 호출이 필요 없다고 판단되면 즉시 최종 결과를 작성하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.
4) 근거 문장/출처 기반으로만 작성하고, 근거 없는 단정은 금지합니다.
5) 최종 출력은 반드시 ResumeNotes 스키마를 따르세요.
6) 로컬 정책: 이력서 텍스트가 없으면 개인 문장 교정보다 "직무-이력서 갭 분석 및 보강 체크리스트" 중심으로 작성하세요.
7) JD/공고 텍스트가 있으면 JD 요구역량과 이력서 간 갭을 항목별로 명시적으로 비교하세요.
8) evidence_map 필수: key는 개선/진단 문장, value는 관련 근거 번호 목록(예: [1,2])입니다.

[Few-shot]
{resume_few_shot}
"""
        try:
            if not has_resume_text:
                parser = llm.with_structured_output(ResumeNotes)
                structured = parser.invoke(
                    f"""
이력서 원문이 없는 상황입니다.
아래 조건을 반영해 ResumeNotes를 작성하세요.
- 개인 경력 단정 금지
- 직무 공고 기준 갭 분석/보강 우선순위 제시
- 한국어로만 작성
- evidence_map 필수: 각 개선항목을 근거 번호([1],[2]...)와 매핑

[목표 직무]
{state['target_role']}

[JD/공고 텍스트]
{jd_text or '없음'}

[사용자 요청]
{state['user_query']}

[RAG 근거]
{state.get('rag_context', '')}
"""
                )
            else:
                structured = _run_tool_loop_structured(
                    prompt=prompt,
                    tools=[resume_keyword_match_score],
                    output_schema=ResumeNotes,
                )
            return {"resume_notes": structured.model_dump()}
        except Exception as exc:
            return {"resume_notes": _fallback_resume_notes(state, f"resume_node fallback: {exc}")}

    def interview_node(state: AgentState) -> AgentState:
        has_resume_text = bool((state.get("resume_text") or "").strip())
        jd_text = (state.get("jd_text") or "").strip()
        prompt = f"""
당신은 Interview Agent입니다.
중요: 답변은 반드시 한국어로만 작성하세요.
중요: 생각 과정을 노출하지 말고 결과만 제시하세요.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}
JD/공고 텍스트:
{jd_text or 'JD/공고 텍스트가 제공되지 않았습니다.'}

RAG 근거:
{state['rag_context']}

지시:
1) interview_question_bank 도구를 활용해 질문 세트를 참고하세요.
2) 도구 호출이 필요 없다고 판단되면 즉시 최종 결과를 작성하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.
4) 근거 문장/출처 기반으로만 작성하고, 근거 없는 단정은 금지합니다.
5) 최종 출력은 반드시 InterviewNotes 스키마를 따르세요.
6) 로컬 정책: 이력서 텍스트가 없으면 개인 경험 단정 대신 직무 공통 질문/답변 프레임워크 중심으로 작성하세요.
7) JD/공고 텍스트가 있으면 JD 요구역량을 기준으로 질문 우선순위를 조정하세요.
8) evidence_map 필수: key는 질문/답변 문장, value는 관련 근거 번호 목록(예: [1,2])입니다.

[Few-shot]
{interview_few_shot}
"""
        try:
            if not has_resume_text:
                parser = llm.with_structured_output(InterviewNotes)
                structured = parser.invoke(
                    f"""
이력서 원문이 없는 상황입니다.
아래 조건을 반영해 InterviewNotes를 작성하세요.
- 개인 이력 단정 금지
- 직무 공통 질문, 답변 프레임워크, 금지 패턴을 우선 제시
- 한국어로만 작성
- evidence_map 필수: 각 질문/답변 항목을 근거 번호([1],[2]...)와 매핑

[목표 직무]
{state['target_role']}

[JD/공고 텍스트]
{jd_text or '없음'}

[사용자 요청]
{state['user_query']}

[RAG 근거]
{state.get('rag_context', '')}
"""
                )
            else:
                structured = _run_tool_loop_structured(
                    prompt=prompt,
                    tools=[interview_question_bank],
                    output_schema=InterviewNotes,
                )
            return {"interview_notes": structured.model_dump()}
        except Exception as exc:
            return {
                "interview_notes": _fallback_interview_notes(
                    state, f"interview_node fallback: {exc}"
                )
            }

    def synthesis_node(state: AgentState) -> AgentState:
        parser_llm = llm.with_structured_output(FinalAnswer)
        route = state.get("route", "full")
        min_rules = _route_minimums(route)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        )
        resume_notes = (
            _format_resume_notes(state.get("resume_notes", {}))
            if state.get("resume_notes")
            else "이번 라우트에서는 Resume Agent를 생략했습니다."
        )
        interview_notes = (
            _format_interview_notes(state.get("interview_notes", {}))
            if state.get("interview_notes")
            else "이번 라우트에서는 Interview Agent를 생략했습니다."
        )
        plan_notes = (
            _format_plan_notes(state.get("plan_notes", {}))
            if state.get("plan_notes")
            else "이번 라우트에서는 Plan Agent를 생략했습니다."
        )
        try:
            answer = parser_llm.invoke(
                f"""
당신은 JobPilot AI의 Supervisor입니다.
아래 정보를 통합하여 반드시 JSON 스키마에 맞는 결과를 생성하세요.
모든 필드(summary, resume_improvements, interview_preparation, two_week_plan, references)는 반드시 한국어로 작성하세요.
생각 과정을 노출하지 말고 결과만 작성하세요.
각 불릿 항목 끝에 근거 번호 citation을 붙이세요(예: "... [1][2]").

[사용자 요청]
{state['user_query']}

[목표 직무]
{state['target_role']}

[JD/공고 텍스트]
{state.get('jd_text', '') or '미제공'}

[최근 대화 메모리]
{history_text or '없음'}

[Supervisor 라우팅]
- route: {route}
- reason: {state.get('routing_reason', '기본 라우트')}

[Resume Agent 결과]
{resume_notes}

[Interview Agent 결과]
{interview_notes}

[Plan Agent 결과]
{plan_notes}

[RAG 근거]
{state.get('rag_context', 'RAG 단계를 생략했습니다.')}

[RAG 신뢰도 모드]
{"근거 부족 모드(보수적 표현 우선)" if state.get('rag_low_confidence') else "일반 모드"}

[참고 출처]
{state.get('rag_refs', [])}

요구사항:
- summary는 4문장 이내.
- route별 최소 개수:
  - resume_improvements >= {min_rules['resume_improvements']}
  - interview_preparation >= {min_rules['interview_preparation']}
  - two_week_plan >= {min_rules['two_week_plan']}
- route가 full 또는 plan_only인 경우 Plan Agent 결과를 우선 반영해 two_week_plan을 작성하세요.
- 최소 개수가 0인 섹션은 빈 배열로 반환하세요.
- references는 출처 파일명 중심으로 구성.
"""
            )
            normalized = _normalize_final_answer_by_route(route, answer.model_dump())
            return {"final_answer": normalized}
        except Exception as exc:
            fallback = _fallback_final_answer(state, f"synthesis fallback: {exc}")
            normalized = _normalize_final_answer_by_route(route, fallback)
            return {"final_answer": normalized}

    def route_after_supervisor(state: AgentState) -> str:
        route = state.get("route", "full")
        if route in {"full", "resume_only", "interview_only", "plan_only"}:
            return "rag"
        return "rag"

    def route_after_rag(state: AgentState) -> str:
        route = state.get("route", "full")
        if route in {"full", "resume_only"}:
            return "resume"
        if route == "interview_only":
            return "interview"
        if route == "plan_only":
            return "plan"
        return "synthesis"

    def route_after_resume(state: AgentState) -> str:
        return "interview" if state.get("route", "full") == "full" else "synthesis"

    def route_after_interview(state: AgentState) -> str:
        return "plan" if state.get("route", "full") == "full" else "synthesis"

    def route_after_plan(state: AgentState) -> str:
        return "synthesis"

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("resume", resume_node)
    graph.add_node("interview", interview_node)
    graph.add_node("plan", plan_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"rag": "rag"},
    )
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"resume": "resume", "interview": "interview", "plan": "plan", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "resume",
        route_after_resume,
        {"interview": "interview", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "interview",
        route_after_interview,
        {"plan": "plan", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"synthesis": "synthesis"},
    )
    graph.add_edge("synthesis", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


class JobPilotService:
    """High-level service that owns retriever, workflow and memory."""

    def __init__(self) -> None:
        settings = load_settings()
        self.memory = SessionMemory(
            storage_path=settings.index_dir / "session_memory.json",
            max_sessions=settings.memory_max_sessions,
            ttl_seconds=settings.memory_ttl_seconds,
            persist_enabled=settings.memory_persist_enabled,
            pii_mask_enabled=settings.pii_mask_enabled,
        )
        self.retriever = HybridRetriever.build()
        self.graph = build_graph(retriever=self.retriever, memory=self.memory)

    def run(self, req: ChatRequest) -> ChatResponse:
        try:
            self.memory.add(req.session_id, "user", req.user_query)

            state: AgentState = {
                "session_id": req.session_id,
                "user_query": req.user_query,
                "target_role": req.target_role,
                "resume_text": req.resume_text,
                "jd_text": req.jd_text,
            }
            result = self.graph.invoke(
                state,
                config={"configurable": {"thread_id": req.session_id}},
            )
            payload = result["final_answer"]
            response = ChatResponse(session_id=req.session_id, **payload)

            self.memory.add(req.session_id, "assistant", response.summary)
            return response
        except JobPilotError:
            raise
        except Exception as exc:
            raise JobPilotError(
                error_code="SERVICE_RUNTIME_ERROR",
                detail=f"JobPilot service failed: {exc}",
                status_code=500,
            ) from exc
