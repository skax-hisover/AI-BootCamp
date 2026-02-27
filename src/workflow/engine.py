"""LangGraph multi-agent workflow for JobPilot AI."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.agents.schemas import FinalAnswer
from src.agents.tools import interview_question_bank, resume_keyword_match_score
from src.config import get_chat_model
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
    resume_notes: str
    interview_notes: str
    final_answer: dict[str, Any]


def _run_tool_loop(prompt: str, tools: list[BaseTool]) -> str:
    llm = get_chat_model(temperature=0.2).bind_tools(tools)
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    tool_messages: list[ToolMessage] = []
    for call in response.tool_calls:
        for tool in tools:
            if tool.name == call["name"]:
                output = tool.invoke(call.get("args", {}))
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=call["id"],
                        name=tool.name,
                    )
                )
                break

    if not tool_messages:
        return response.content if isinstance(response.content, str) else str(response.content)

    final = get_chat_model(temperature=0.2).invoke([messages[0], response, *tool_messages])
    return final.content if isinstance(final.content, str) else str(final.content)


def build_graph(retriever: HybridRetriever, memory: SessionMemory):
    llm = get_chat_model(temperature=0.1)

    def supervisor_node(state: AgentState) -> AgentState:
        return {
            "memory_messages": memory.get(state["session_id"], limit=8),
        }

    def rag_node(state: AgentState) -> AgentState:
        context, refs = retriever.search_as_context(state["user_query"])
        ref_names = [f"{r['rank']}. {r['source']} (score={r['score']})" for r in refs]
        return {"rag_context": context, "rag_refs": ref_names}

    def resume_node(state: AgentState) -> AgentState:
        prompt = f"""
당신은 Resume Agent입니다.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}
사용자 이력서:
{state['resume_text'] or '이력서 텍스트가 제공되지 않았습니다.'}

RAG 근거:
{state['rag_context']}

지시:
1) resume_keyword_match_score 도구를 활용해 키워드 적합도를 반영하세요.
2) 불필요한 설명 없이, 개선 포인트 중심으로 6줄 이내로 작성하세요.
"""
        result = _run_tool_loop(prompt, [resume_keyword_match_score])
        return {"resume_notes": result}

    def interview_node(state: AgentState) -> AgentState:
        prompt = f"""
당신은 Interview Agent입니다.
목표 직무: {state['target_role']}
사용자 요청: {state['user_query']}

RAG 근거:
{state['rag_context']}

지시:
1) interview_question_bank 도구를 활용해 질문 세트를 참고하세요.
2) 예상 질문 + 답변 방향 + 피해야 할 답변 패턴을 핵심만 제시하세요.
"""
        result = _run_tool_loop(prompt, [interview_question_bank])
        return {"interview_notes": result}

    def synthesis_node(state: AgentState) -> AgentState:
        parser_llm = llm.with_structured_output(FinalAnswer)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("memory_messages", [])
        )
        answer = parser_llm.invoke(
            f"""
당신은 JobPilot AI의 Supervisor입니다.
아래 정보를 통합하여 반드시 JSON 스키마에 맞는 결과를 생성하세요.

[사용자 요청]
{state['user_query']}

[목표 직무]
{state['target_role']}

[최근 대화 메모리]
{history_text or '없음'}

[Resume Agent 결과]
{state['resume_notes']}

[Interview Agent 결과]
{state['interview_notes']}

[RAG 근거]
{state['rag_context']}

[참고 출처]
{state['rag_refs']}

요구사항:
- summary는 4문장 이내.
- resume_improvements, interview_preparation, two_week_plan은 각각 최소 4개.
- references는 출처 파일명 중심으로 구성.
"""
        )
        return {"final_answer": answer.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("resume", resume_node)
    graph.add_node("interview", interview_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "rag")
    graph.add_edge("rag", "resume")
    graph.add_edge("resume", "interview")
    graph.add_edge("interview", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


class JobPilotService:
    """High-level service that owns retriever, workflow and memory."""

    def __init__(self) -> None:
        self.memory = SessionMemory()
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
