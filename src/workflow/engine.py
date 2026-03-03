"""LangGraph multi-agent workflow for JobPilot AI."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents.schemas import FinalAnswer
from src.agents.tools import interview_question_bank, resume_keyword_match_score
from src.config import get_chat_model, load_settings
from src.retrieval import HybridRetriever
from src.utils.memory import SessionMemory
from src.workflow.contracts import ChatRequest, ChatResponse


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    target_role: str
    resume_text: str
    memory_messages: list[dict[str, str]]
    rag_context: str
    rag_refs: list[str]
    route: str
    routing_reason: str
    resume_notes: str
    interview_notes: str
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


def _normalize_content(content: Any) -> str:
    text = content if isinstance(content, str) else str(content)
    return text.replace("DONE", "").strip()


def _run_tool_loop(prompt: str, tools: list[BaseTool], max_steps: int = 4) -> str:
    llm = get_chat_model(temperature=0.2).bind_tools(tools)
    messages = [HumanMessage(content=prompt)]
    for _ in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return _normalize_content(response.content)

        for call in response.tool_calls:
            matched_tool = next((tool for tool in tools if tool.name == call["name"]), None)
            if not matched_tool:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {call['name']}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue

            output = matched_tool.invoke(call.get("args", {}))
            messages.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=call["id"],
                    name=matched_tool.name,
                )
            )

    final = get_chat_model(temperature=0.2).invoke(
        [
            *messages,
            HumanMessage(
                content=(
                    "도구 결과를 종합해 최종 답변만 간결히 작성하고 마지막에 DONE을 붙이세요. "
                    "반드시 한국어로만 작성하세요."
                )
            ),
        ]
    )
    return _normalize_content(final.content)


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

    def _route_by_supervisor(state: AgentState) -> AgentState:
        router = llm.with_structured_output(RouteDecision)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        ) or "없음"
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

[최근 메모리]
{history_text}
"""
            )
            return {"route": decision.route, "routing_reason": decision.reason}
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

    def rag_node(state: AgentState) -> AgentState:
        planner = llm.with_structured_output(RagPlan)
        role_hint = _infer_role_hint(state["target_role"])
        try:
            rag_plan = planner.invoke(
                f"""
너는 RAG Agent다. 검색 품질을 높이기 위해 쿼리를 2~3개로 재작성하라.
route={state.get('route', 'full')}
user_query={state['user_query']}
target_role={state['target_role']}
source_hint는 직무 연관 키워드로 짧게 작성하라.
"""
            )
            query_candidates = [state["user_query"], *rag_plan.rewritten_queries]
            source_hint = rag_plan.source_hint or role_hint
        except Exception:
            query_candidates = [state["user_query"], f"{state['target_role']} {state['user_query']}"]
            source_hint = role_hint

        merged_hits: dict[str, dict[str, Any]] = {}
        for candidate in query_candidates[:3]:
            for hit in retriever.search(candidate, top_k=4):
                key = f"{hit.source}::{hit.content[:120]}"
                boosted = hit.score + (
                    0.15 if _is_role_relevant(f"{hit.source} {hit.content}", source_hint) else 0.0
                )
                current = merged_hits.get(key)
                if not current or boosted > current["score"]:
                    merged_hits[key] = {
                        "content": hit.content,
                        "source": hit.source,
                        "score": round(boosted, 4),
                    }

        ranked_hits = sorted(merged_hits.values(), key=lambda item: item["score"], reverse=True)[:4]
        context_parts: list[str] = []
        refs: list[dict[str, Any]] = []
        for i, hit in enumerate(ranked_hits, start=1):
            context_parts.append(f"[{i}] ({hit['source']}, score={hit['score']})\n{hit['content']}")
            refs.append({"rank": i, "source": hit["source"], "score": hit["score"]})

        if not context_parts:
            context_parts = ["검색 결과가 부족하여 일반적인 직무 가이드를 기반으로 응답합니다."]

        ref_names = [f"{r['rank']}. {r['source']} (score={r['score']})" for r in refs]
        rag_context = "\n\n".join(context_parts)
        rag_context += f"\n\n[rag-agent-note] route={state.get('route', 'full')}, source_hint={source_hint}"
        return {"rag_context": rag_context, "rag_refs": ref_names}

    def resume_node(state: AgentState) -> AgentState:
        prompt = f"""
당신은 Resume Agent입니다.
중요: 답변은 반드시 한국어로만 작성하세요.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}
사용자 이력서:
{state['resume_text'] or '이력서 텍스트가 제공되지 않았습니다.'}

RAG 근거:
{state['rag_context']}

지시:
1) resume_keyword_match_score 도구를 활용해 키워드 적합도를 반영하세요.
2) 불필요한 설명 없이, 개선 포인트 중심으로 6줄 이내로 작성하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.

[Few-shot]
{resume_few_shot}

충분한 결론에 도달하면 마지막에 DONE을 붙이세요.
"""
        result = _run_tool_loop(prompt, [resume_keyword_match_score])
        return {"resume_notes": result}

    def interview_node(state: AgentState) -> AgentState:
        prompt = f"""
당신은 Interview Agent입니다.
중요: 답변은 반드시 한국어로만 작성하세요.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}

RAG 근거:
{state['rag_context']}

지시:
1) interview_question_bank 도구를 활용해 질문 세트를 참고하세요.
2) 예상 질문 + 답변 방향 + 피해야 할 답변 패턴을 핵심만 제시하세요.
3) 아래 Few-shot 스타일을 참고해 동일한 형식으로 작성하세요.

[Few-shot]
{interview_few_shot}

충분한 결론에 도달하면 마지막에 DONE을 붙이세요.
"""
        result = _run_tool_loop(prompt, [interview_question_bank])
        return {"interview_notes": result}

    def synthesis_node(state: AgentState) -> AgentState:
        parser_llm = llm.with_structured_output(FinalAnswer)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        )
        resume_notes = state.get("resume_notes", "이번 라우트에서는 Resume Agent를 생략했습니다.")
        interview_notes = state.get("interview_notes", "이번 라우트에서는 Interview Agent를 생략했습니다.")
        answer = parser_llm.invoke(
            f"""
당신은 JobPilot AI의 Supervisor입니다.
아래 정보를 통합하여 반드시 JSON 스키마에 맞는 결과를 생성하세요.
모든 필드(summary, resume_improvements, interview_preparation, two_week_plan, references)는 반드시 한국어로 작성하세요.

[사용자 요청]
{state['user_query']}

[목표 직무]
{state['target_role']}

[최근 대화 메모리]
{history_text or '없음'}

[Supervisor 라우팅]
- route: {state.get('route', 'full')}
- reason: {state.get('routing_reason', '기본 라우트')}

[Resume Agent 결과]
{resume_notes}

[Interview Agent 결과]
{interview_notes}

[RAG 근거]
{state.get('rag_context', 'RAG 단계를 생략했습니다.')}

[참고 출처]
{state.get('rag_refs', [])}

요구사항:
- summary는 4문장 이내.
- resume_improvements, interview_preparation, two_week_plan은 각각 최소 4개.
- references는 출처 파일명 중심으로 구성.
"""
        )
        return {"final_answer": answer.model_dump()}

    def route_after_supervisor(state: AgentState) -> str:
        route = state.get("route", "full")
        if route == "plan_only":
            return "synthesis"
        if route in {"full", "resume_only", "interview_only"}:
            return "rag"
        return "rag"

    def route_after_rag(state: AgentState) -> str:
        route = state.get("route", "full")
        if route in {"full", "resume_only"}:
            return "resume"
        if route == "interview_only":
            return "interview"
        return "synthesis"

    def route_after_resume(state: AgentState) -> str:
        return "interview" if state.get("route", "full") == "full" else "synthesis"

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("resume", resume_node)
    graph.add_node("interview", interview_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"rag": "rag", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"resume": "resume", "interview": "interview", "synthesis": "synthesis"},
    )
    graph.add_conditional_edges(
        "resume",
        route_after_resume,
        {"interview": "interview", "synthesis": "synthesis"},
    )
    graph.add_edge("interview", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


class JobPilotService:
    """High-level service that owns retriever, workflow and memory."""

    def __init__(self) -> None:
        settings = load_settings()
        self.memory = SessionMemory(storage_path=settings.index_dir / "session_memory.json")
        self.retriever = HybridRetriever.build()
        self.graph = build_graph(retriever=self.retriever, memory=self.memory)

    def run(self, req: ChatRequest) -> ChatResponse:
        self.memory.add(req.session_id, "user", req.user_query)

        state: AgentState = {
            "session_id": req.session_id,
            "user_query": req.user_query,
            "target_role": req.target_role,
            "resume_text": req.resume_text,
        }
        result = self.graph.invoke(state)
        payload = result["final_answer"]
        response = ChatResponse(session_id=req.session_id, **payload)

        self.memory.add(req.session_id, "assistant", response.summary)
        return response
