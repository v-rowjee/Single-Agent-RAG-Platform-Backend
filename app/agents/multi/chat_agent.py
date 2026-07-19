"""Grounded chat generation over session-scoped retrieval evidence only."""
from __future__ import annotations

import logging
import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import agent_model_policy, get_rag_config
from app.core.groq_structured import request_structured
from app.core.prompts import render_agent_prompts
from app.rag.models import RetrievedDocument


logger = logging.getLogger(__name__)

_RAG_CONFIG = get_rag_config()
MAX_RETRIEVED_DOCUMENTS = _RAG_CONFIG.retrieval.chat_search_limit
MAX_CONTEXT_CHARACTERS = _RAG_CONFIG.retrieval.max_context_chars
MAX_DOCUMENT_CHARACTERS = _RAG_CONFIG.chunking.size
INSUFFICIENT_CONTEXT_ANSWER = (
    "The available analysis does not contain enough information to answer that "
    "question."
)
FALLBACK_ANSWER = (
    "The analysis assistant could not produce a grounded answer from the "
    "available evidence."
)


class GroundedChatDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    insufficient_context: bool


def _document_source_ids(document: RetrievedDocument) -> list[str]:
    metadata = document.metadata
    values = [metadata.get("source_id")]
    source_ids = metadata.get("source_ids")
    if isinstance(source_ids, list):
        values.extend(source_ids)
    output: list[str] = []
    for value in values:
        source_id = str(value or "").strip()
        if source_id and source_id not in output:
            output.append(source_id)
    return output


def _compact_context(documents: list[RetrievedDocument]) -> str:
    sections: list[str] = []
    used_characters = 0
    for document in documents[:MAX_RETRIEVED_DOCUMENTS]:
        metadata = document.metadata
        source_ids = _document_source_ids(document)
        content = document.page_content.strip()[:MAX_DOCUMENT_CHARACTERS]
        if not content:
            continue
        header = (
            f"Document ID: {source_ids[0] if source_ids else 'unknown'}\n"
            f"Dataset: {metadata.get('file_name') or 'active workspace'}\n"
            f"Type: {metadata.get('document_type') or 'unknown'}\n"
            f"Title: {metadata.get('title') or source_ids[0] if source_ids else 'Business intelligence evidence'}\n"
            f"Source IDs: {', '.join(source_ids) or 'not available'}\n"
            "Content:\n"
        )
        remaining = MAX_CONTEXT_CHARACTERS - used_characters - len(header) - 2
        if remaining <= 0:
            break
        section = f"{header}{content[:remaining].rstrip()}"
        sections.append(section)
        used_characters += len(section) + 2
    return "\n\n".join(sections)


def _validated_source_ids(
    draft: GroundedChatDraft,
    documents: list[RetrievedDocument],
) -> list[str]:
    selected = {str(value).strip() for value in draft.source_ids if str(value).strip()}
    output: list[str] = []
    for document in documents[:MAX_RETRIEVED_DOCUMENTS]:
        for source_id in _document_source_ids(document):
            if source_id in selected and source_id not in output:
                output.append(source_id)
    return output


async def _request_groq_draft(
    query: str,
    context: str,
    history: list[dict[str, str]],
) -> GroundedChatDraft:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    prompts = render_agent_prompts(
        "multi/chat",
        payload={
            "query": query,
            "conversation_history": history,
            "documents": context,
        },
    )
    return await request_structured(
        api_key=api_key,
        policy=agent_model_policy("chat"),
        response_model=GroundedChatDraft,
        schema_name="grounded_chat_draft",
        temperature=0.1,
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


class ChatAgent:
    async def run(
        self,
        session_id: str,
        query: str,
        retrieved_documents: list[RetrievedDocument],
        history: list[dict[str, str]] | None = None,
    ) -> GroundedChatDraft:
        documents = retrieved_documents[:MAX_RETRIEVED_DOCUMENTS]
        logger.info(
            "Chat agent started session_id=%s retrieved_document_count=%s",
            session_id,
            len(documents),
        )
        if not documents:
            logger.info("Insufficient context detected session_id=%s", session_id)
            return GroundedChatDraft(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )

        try:
            draft = await _request_groq_draft(
                query,
                _compact_context(documents),
                history or [],
            )
            validated = draft.model_copy(
                update={"source_ids": _validated_source_ids(draft, documents)}
            )
            if validated.insufficient_context:
                logger.info("Insufficient context detected session_id=%s", session_id)
            logger.info("Grounded draft generated session_id=%s", session_id)
            return validated
        except (ValidationError, RuntimeError, ValueError):
            logger.info("Chat agent fallback used session_id=%s", session_id)
            return GroundedChatDraft(
                answer=FALLBACK_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        except Exception:
            logger.exception("Chat agent fallback used session_id=%s", session_id)
            return GroundedChatDraft(
                answer=FALLBACK_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )


chat_agent = ChatAgent()
