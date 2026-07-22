"""Direct validation, retrieval, generation, and grounding for dataset chat."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.multi.chat import (
    INSUFFICIENT_CONTEXT_ANSWER,
    chat_agent,
)
from app.core.config import agent_model_policy, get_rag_config
from app.orchestration.state import ChatState
from app.rag.models import RetrievedDocument
from app.rag.retrieval.retriever import retriever
from app.schemas.specialists import GroundedChatDraft


logger = logging.getLogger(__name__)
_RAG_CONFIG = get_rag_config()
_VECTOR_SEARCH_LIMIT = _RAG_CONFIG.retrieval.vector_search_limit
_CHAT_SEARCH_LIMIT = _RAG_CONFIG.retrieval.chat_search_limit
_CHAT_AGENT_TIMEOUT_SECONDS = agent_model_policy("chat").timeout_seconds
_MAX_HISTORY_MESSAGES = 6
_MAX_HISTORY_CHARACTERS = 2_000

BLOCKED_CHAT_ANSWER = (
    "I cannot follow requests to reveal secrets or override the analysis "
    "assistant's instructions. Please ask a question about this dataset's "
    "analysis instead."
)
CHAT_FAILURE_ANSWER = (
    "The analysis assistant could not answer this question at the moment."
)
CHAT_TIMEOUT_ANSWER = (
    "The detailed response took longer than expected. Please try asking about "
    "a specific product, period, or trend."
)
CAUSAL_TERMS = ("caused", "cause", "because", "responsible", "driver")
CAUSAL_FALLBACK = (
    "The available analysis shows the observed change but does not identify who "
    "caused it."
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?!\w)")
_BLOCKED_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        (
            r"\b(ignore|disregard|override|forget|bypass)\b.{0,80}"
            r"\b(previous|prior|system|developer|assistant|security)\b.{0,40}"
            r"\b(instruction|instructions|prompt|message|rules?)\b"
        ),
        (
            r"\b(reveal|show|print|display|expose|leak|return|give\s+me)\b.{0,100}"
            r"\b(system\s+prompt|developer\s+message|hidden\s+instructions?|"
            r"api[_\s-]?key|service[_\s-]?role|password|access[_\s-]?token|secret)\b"
        ),
        (
            r"\b(system\s+prompt|developer\s+message|hidden\s+instructions?|"
            r"api[_\s-]?key|service[_\s-]?role|password|access[_\s-]?token|secret)\b"
            r".{0,100}\b(reveal|show|print|display|expose|leak|return)\b"
        ),
        r"\b(jailbreak|prompt\s+injection|developer\s+mode)\b",
    )
)


@dataclass(frozen=True)
class ChatResult:
    query: str
    draft: GroundedChatDraft


def _source_ids(document: RetrievedDocument) -> list[str]:
    values = [document.metadata.get("source_id")]
    if isinstance(document.metadata.get("source_ids"), list):
        values.extend(document.metadata["source_ids"])
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _ordered_valid_source_ids(
    source_ids: list[str],
    documents: list[RetrievedDocument],
) -> list[str]:
    requested = {str(value).strip() for value in source_ids if str(value).strip()}
    output: list[str] = []
    for document in documents:
        for source_id in _source_ids(document):
            if source_id in requested and source_id not in output:
                output.append(source_id)
    return output


def _numbers(text: str) -> set[str]:
    return {
        match.group(0).replace(",", "") for match in NUMBER_PATTERN.finditer(text)
    }


def _safe_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    used_characters = 0
    for message in history[-_MAX_HISTORY_MESSAGES:]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "user" and any(pattern.search(content) for pattern in _BLOCKED_PATTERNS):
            continue
        remaining = _MAX_HISTORY_CHARACTERS - used_characters
        if remaining <= 0:
            break
        content = content[:remaining]
        output.append({"role": role, "content": content})
        used_characters += len(content)
    return output


def _retrieval_query(query: str, history: list[dict[str, str]]) -> str:
    if not history:
        return query
    history_text = "\n".join(
        f"{message['role']}: {message['content']}" for message in history
    )
    return (
        f"Current question: {query}\n"
        f"Recent conversation context:\n{history_text}"
    )


def _ground_draft(
    query: str,
    documents: list[RetrievedDocument],
    draft: GroundedChatDraft,
) -> GroundedChatDraft:
    if draft.insufficient_context or not documents:
        return GroundedChatDraft(
            answer=draft.answer or INSUFFICIENT_CONTEXT_ANSWER,
            source_ids=[],
            insufficient_context=True,
        )

    source_ids = _ordered_valid_source_ids(draft.source_ids, documents)
    if not source_ids:
        return GroundedChatDraft(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            source_ids=[],
            insufficient_context=True,
        )

    evidence = "\n".join(document.page_content for document in documents)
    query_is_causal = any(term in query.casefold() for term in CAUSAL_TERMS)
    evidence_supports_causation = any(
        term in evidence.casefold() for term in CAUSAL_TERMS
    )
    if query_is_causal and not evidence_supports_causation:
        return GroundedChatDraft(
            answer=CAUSAL_FALLBACK,
            source_ids=source_ids,
            insufficient_context=False,
        )

    if _numbers(draft.answer) - _numbers(evidence):
        return GroundedChatDraft(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            source_ids=[],
            insufficient_context=True,
        )
    return draft.model_copy(update={"source_ids": source_ids})


def _timeout_fallback(documents: list[RetrievedDocument]) -> GroundedChatDraft:
    """Return retrieved recommendations when model generation exceeds its budget."""
    recommendations = [
        document
        for document in documents
        if document.metadata.get("document_type") == "recommendation"
        and document.page_content.strip()
    ][:3]
    if not recommendations:
        return GroundedChatDraft(
            answer=CHAT_TIMEOUT_ANSWER,
            source_ids=[],
            insufficient_context=True,
        )

    actions = []
    source_ids: list[str] = []
    for document in recommendations:
        title = str(document.metadata.get("title") or "Recommended action").strip()
        actions.append(f"- {title}: {document.page_content.strip()}")
        for source_id in _source_ids(document):
            if source_id not in source_ids:
                source_ids.append(source_id)

    if not source_ids:
        return GroundedChatDraft(
            answer=CHAT_TIMEOUT_ANSWER,
            source_ids=[],
            insufficient_context=True,
        )
    return GroundedChatDraft(
        answer=(
            "The detailed recommendation timed out, so here are the existing "
            "dataset-grounded actions:\n" + "\n".join(actions)
        ),
        source_ids=source_ids,
        insufficient_context=False,
    )


def build_chat_graph(rag: Any | None = None, agent: Any | None = None):
    """Compile the guarded retrieval, generation, and grounding workflow."""

    retrieval = rag or retriever
    chat = agent or chat_agent

    def guardrail(state: ChatState) -> dict[str, Any]:
        query = str(state.get("query") or "").strip()
        if not query:
            raise ValueError("The chat query cannot be empty.")
        blocked = any(pattern.search(query) for pattern in _BLOCKED_PATTERNS)
        update: dict[str, Any] = {
            "query": query,
            "history": _safe_history(state.get("history") or []),
            "blocked": blocked,
        }
        if blocked:
            update["draft"] = GroundedChatDraft(
                answer=BLOCKED_CHAT_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return update

    def route_guardrail(state: ChatState) -> str:
        return "blocked" if state.get("blocked") else "retrieve"

    def retrieve_node(state: ChatState) -> dict[str, Any]:
        contextual_query = _retrieval_query(state["query"], state.get("history") or [])
        candidates = retrieval.retrieve(
            session_id=state["session_id"],
            query=contextual_query,
            limit=_VECTOR_SEARCH_LIMIT,
        )
        return {
            "retrieval_query": contextual_query,
            "retrieved_documents": candidates,
        }

    def rerank_node(state: ChatState) -> dict[str, Any]:
        candidates = state.get("retrieved_documents") or []
        reranked = retrieval.rerank(state["retrieval_query"], candidates)
        return {
            "reranked_documents": (reranked or candidates)[:_CHAT_SEARCH_LIMIT],
        }

    def generation_node(state: ChatState) -> dict[str, Any]:
        documents = state.get("reranked_documents") or []
        try:
            draft = asyncio.run(
                asyncio.wait_for(
                    chat.run(
                        session_id=state["session_id"],
                        query=state["query"],
                        retrieved_documents=documents,
                        history=state.get("history") or [],
                    ),
                    timeout=_CHAT_AGENT_TIMEOUT_SECONDS,
                )
            )
        except TimeoutError:
            logger.warning(
                "Chat agent timed out session_id=%s timeout_seconds=%s",
                state["session_id"],
                _CHAT_AGENT_TIMEOUT_SECONDS,
            )
            draft = _timeout_fallback(documents)
        except Exception:
            logger.exception("Chat agent failed session_id=%s", state["session_id"])
            draft = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return {"draft": draft}

    def grounding_node(state: ChatState) -> dict[str, Any]:
        try:
            draft = _ground_draft(
                state["query"],
                state.get("reranked_documents") or [],
                state["draft"],
            )
        except Exception:
            logger.exception("Chat grounding failed session_id=%s", state["session_id"])
            draft = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return {"draft": draft}

    graph = StateGraph(ChatState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("blocked", lambda state: {})
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generation_node)
    graph.add_node("ground", grounding_node)
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_guardrail,
        {"blocked": "blocked", "retrieve": "retrieve"},
    )
    graph.add_edge("blocked", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "ground")
    graph.add_edge("ground", END)
    return graph.compile()


class ChatGraph:
    def __init__(self, rag: Any | None = None, agent: Any | None = None) -> None:
        self._graph = build_chat_graph(rag, agent)

    def answer(
        self,
        session_id: str,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ChatResult:
        result = self._graph.invoke(
            {"session_id": session_id, "query": query, "history": history or []}
        )
        return ChatResult(query=result["query"], draft=result["draft"])
