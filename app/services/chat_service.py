"""Direct validation, retrieval, generation, and grounding for dataset chat."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.agents.multi.chat_agent import (
    GroundedChatDraft,
    INSUFFICIENT_CONTEXT_ANSWER,
    chat_agent,
)
from app.core.config import get_rag_config
from app.rag.models import RetrievedDocument
from app.rag.rag_service import rag_service


logger = logging.getLogger(__name__)
_RAG_CONFIG = get_rag_config()
_VECTOR_SEARCH_LIMIT = _RAG_CONFIG.retrieval.vector_search_limit
_CHAT_SEARCH_LIMIT = _RAG_CONFIG.retrieval.chat_search_limit
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


class ChatService:
    def __init__(self, rag: Any | None = None, agent: Any | None = None) -> None:
        self._rag = rag or rag_service
        self._agent = agent or chat_agent

    def answer(
        self,
        session_id: str,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ChatResult:
        """Run the multi-agent chat sequence without a graph wrapper."""
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")

        if any(pattern.search(cleaned_query) for pattern in _BLOCKED_PATTERNS):
            return ChatResult(
                query=cleaned_query,
                draft=GroundedChatDraft(
                    answer=BLOCKED_CHAT_ANSWER,
                    source_ids=[],
                    insufficient_context=True,
                ),
            )

        safe_history = _safe_history(history or [])
        contextual_query = _retrieval_query(cleaned_query, safe_history)
        candidates = self._rag.retrieve(
            session_id=session_id,
            query=contextual_query,
            limit=_VECTOR_SEARCH_LIMIT,
        )
        reranked = self._rag.rerank(contextual_query, candidates)
        documents = (reranked or candidates)[:_CHAT_SEARCH_LIMIT]
        try:
            draft = asyncio.run(
                self._agent.run(
                    session_id=session_id,
                    query=cleaned_query,
                    retrieved_documents=documents,
                    history=safe_history,
                )
            )
        except Exception:
            logger.exception("Chat agent failed session_id=%s", session_id)
            draft = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )

        try:
            guarded_draft = _ground_draft(cleaned_query, documents, draft)
        except Exception:
            logger.exception("Chat grounding failed session_id=%s", session_id)
            guarded_draft = GroundedChatDraft(
                answer=CHAT_FAILURE_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return ChatResult(query=cleaned_query, draft=guarded_draft)
