"""Separate session-scoped retrieval and grounding graph for multi-agent chat."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.multi.chat_agent import GroundedChatDraft
from app.guardrails.chat_input import ChatInputGuardrailResult
from app.rag.models import RetrievedDocument


logger = logging.getLogger(__name__)

CHAT_FAILURE_ANSWER = (
    "The analysis assistant could not answer this question at the moment."
)


class MultiAgentChatState(TypedDict, total=False):
    session_id: str
    user_id: str
    query: str
    dataset: Any
    input_guardrail: ChatInputGuardrailResult
    retrieved_documents: list[RetrievedDocument]
    chat_draft: GroundedChatDraft
    guarded_draft: GroundedChatDraft


ValidateSession = Callable[[str, str], Any]
ValidateInput = Callable[[str], ChatInputGuardrailResult]
RetrieveDocuments = Callable[[str, str], list[RetrievedDocument]]
GenerateDraft = Callable[
    [str, str, list[RetrievedDocument]],
    GroundedChatDraft,
]
GroundDraft = Callable[
    [str, list[RetrievedDocument], GroundedChatDraft],
    GroundedChatDraft,
]


def build_multi_agent_chat_graph(
    *,
    validate_session: ValidateSession,
    validate_input: ValidateInput,
    retrieve_documents: RetrieveDocuments,
    generate_draft: GenerateDraft,
    ground_draft: GroundDraft,
):
    """Compile the chat pipeline with injectable deterministic boundaries."""

    def session_validation_node(state: MultiAgentChatState) -> dict[str, Any]:
        session_id = str(state.get("session_id") or "")
        user_id = str(state.get("user_id") or "")
        return {"dataset": validate_session(session_id, user_id)}

    def input_guardrail_node(state: MultiAgentChatState) -> dict[str, Any]:
        result = validate_input(str(state.get("query") or ""))
        return {
            "query": result.query,
            "input_guardrail": result,
        }

    def route_guardrail(
        state: MultiAgentChatState,
    ) -> Literal["session_filtered_retrieval", "blocked_response"]:
        result = state["input_guardrail"]
        return (
            "session_filtered_retrieval"
            if result.allowed
            else "blocked_response"
        )

    def blocked_response_node(state: MultiAgentChatState) -> dict[str, Any]:
        result = state["input_guardrail"]
        return {
            "guarded_draft": GroundedChatDraft(
                answer=result.blocked_answer
                or "This request was blocked by the input guardrail.",
                source_ids=[],
                insufficient_context=True,
            )
        }

    def retrieval_node(state: MultiAgentChatState) -> dict[str, Any]:
        session_id = str(state.get("session_id") or "")
        try:
            documents = retrieve_documents(session_id, state["query"])
        except Exception:
            logger.exception(
                "Chat retrieval failed session_id=%s",
                session_id,
            )
            documents = []
        return {"retrieved_documents": documents}

    def chat_agent_node(state: MultiAgentChatState) -> dict[str, Any]:
        try:
            draft = generate_draft(
                str(state.get("session_id") or ""),
                state["query"],
                list(state.get("retrieved_documents") or []),
            )
        except Exception:
            logger.exception(
                "Chat agent failed session_id=%s",
                state.get("session_id"),
            )
            draft = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return {"chat_draft": draft}

    def output_grounding_node(state: MultiAgentChatState) -> dict[str, Any]:
        try:
            guarded = ground_draft(
                state["query"],
                list(state.get("retrieved_documents") or []),
                state["chat_draft"],
            )
        except Exception:
            logger.exception(
                "Chat grounding failed session_id=%s",
                state.get("session_id"),
            )
            guarded = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return {"guarded_draft": guarded}

    graph = StateGraph(MultiAgentChatState)
    graph.add_node("session_validation", session_validation_node)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("blocked_response", blocked_response_node)
    graph.add_node("session_filtered_retrieval", retrieval_node)
    graph.add_node("chat_agent", chat_agent_node)
    graph.add_node("output_grounding_guardrail", output_grounding_node)

    graph.add_edge(START, "session_validation")
    graph.add_edge("session_validation", "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_guardrail,
        {
            "session_filtered_retrieval": "session_filtered_retrieval",
            "blocked_response": "blocked_response",
        },
    )
    graph.add_edge("blocked_response", END)
    graph.add_edge("session_filtered_retrieval", "chat_agent")
    graph.add_edge("chat_agent", "output_grounding_guardrail")
    graph.add_edge("output_grounding_guardrail", END)
    return graph.compile()
