from __future__ import annotations

import asyncio

import pytest

from app.agents.multi import chat_agent as chat_module
from app.agents.multi.chat_agent import GroundedChatDraft
from app.guardrails.chat_grounding import chat_grounding_guardrail
from app.rag.models import RetrievedDocument


SESSION_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SESSION_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def document(
    source_id: str,
    content: str,
    *,
    source_ids: list[str] | None = None,
    session_id: str = SESSION_A,
) -> RetrievedDocument:
    return RetrievedDocument(
        page_content=content,
        metadata={
            "source_id": source_id,
            "source_ids": source_ids or [source_id],
            "session_id": session_id,
            "dataset_id": session_id,
            "document_type": "trend",
            "title": source_id,
        },
        score=0.9,
    )


def test_chat_agent_returns_a_grounded_draft_from_supplied_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        document("kpi_total_revenue", "Revenue was 120."),
        document(
            "trend_document",
            "Revenue increased across the analysed period.",
            source_ids=["trend_monthly_revenue"],
        ),
    ]
    received: dict[str, str] = {}

    async def groq(query: str, context: str) -> GroundedChatDraft:
        received.update({"query": query, "context": context})
        return GroundedChatDraft(
            answer="Revenue increased across the analysed period.",
            source_ids=["trend_monthly_revenue"],
            insufficient_context=False,
        )

    monkeypatch.setattr(chat_module, "_request_groq_draft", groq)

    result = asyncio.run(
        chat_module.chat_agent.run(SESSION_A, "What is the trend?", documents)
    )

    assert result.source_ids == ["trend_monthly_revenue"]
    assert result.insufficient_context is False
    assert received["query"] == "What is the trend?"
    assert "Revenue was 120." in received["context"]
    assert "Revenue increased across the analysed period." in received["context"]


def test_empty_retrieval_skips_groq_and_returns_insufficient_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_module,
        "_request_groq_draft",
        lambda **kwargs: pytest.fail("Groq must not be called for empty retrieval"),
    )

    result = asyncio.run(chat_module.chat_agent.run(SESSION_A, "What changed?", []))

    assert result.insufficient_context is True
    assert result.source_ids == []
    assert "does not contain enough information" in result.answer


def test_unknown_llm_source_ids_are_removed_before_guarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        document(
            "trend_document",
            "Revenue increased.",
            source_ids=["trend_monthly_revenue"],
        )
    ]

    async def groq(query: str, context: str) -> GroundedChatDraft:
        return GroundedChatDraft(
            answer="Revenue increased.",
            source_ids=["trend_monthly_revenue", "unknown_document"],
            insufficient_context=False,
        )

    monkeypatch.setattr(chat_module, "_request_groq_draft", groq)

    draft = asyncio.run(chat_module.chat_agent.run(SESSION_A, "What changed?", documents))
    guarded = chat_grounding_guardrail.validate("What changed?", documents, draft)

    assert draft.source_ids == ["trend_monthly_revenue"]
    assert guarded.source_ids == ["trend_monthly_revenue"]


def test_unsupported_causal_question_is_sanitised_by_final_guardrail() -> None:
    documents = [document("trend_document", "Revenue declined in June.")]
    draft = GroundedChatDraft(
        answer="A manager caused the revenue decline.",
        source_ids=["trend_document"],
        insufficient_context=False,
    )

    guarded = chat_grounding_guardrail.validate(
        "Who caused the revenue decline?", documents, draft
    )

    assert "does not identify who caused it" in guarded.answer
    assert guarded.source_ids == ["trend_document"]


def test_groq_failure_returns_safe_agent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(query: str, context: str) -> GroundedChatDraft:
        raise RuntimeError("private provider failure")

    monkeypatch.setattr(chat_module, "_request_groq_draft", fail)

    result = asyncio.run(
        chat_module.chat_agent.run(SESSION_A, "What changed?", [document("trend", "Revenue rose.")])
    )

    assert result.insufficient_context is True
    assert result.source_ids == []
    assert "private provider failure" not in result.answer


def test_agent_uses_only_documents_supplied_by_session_filtered_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = [document("session-a-trend", "Revenue rose for session A.")]
    other_session = document("session-b-trend", "Never use this.", session_id=SESSION_B)
    received: dict[str, str] = {}

    async def groq(query: str, context: str) -> GroundedChatDraft:
        received["context"] = context
        return GroundedChatDraft(
            answer="Revenue rose for session A.",
            source_ids=["session-a-trend"],
            insufficient_context=False,
        )

    monkeypatch.setattr(chat_module, "_request_groq_draft", groq)
    result = asyncio.run(chat_module.chat_agent.run(SESSION_A, "What changed?", supplied))

    assert result.source_ids == ["session-a-trend"]
    assert "session A" in received["context"]
    assert other_session.page_content not in received["context"]


def test_numeric_claim_not_present_in_evidence_is_rejected() -> None:
    documents = [document("kpi_revenue", "Revenue was 120.")]
    supported = chat_grounding_guardrail.validate(
        "What was revenue?",
        documents,
        GroundedChatDraft(
            answer="Revenue was 120.",
            source_ids=["kpi_revenue"],
            insufficient_context=False,
        ),
    )
    draft = GroundedChatDraft(
        answer="Revenue was 999.",
        source_ids=["kpi_revenue"],
        insufficient_context=False,
    )

    guarded = chat_grounding_guardrail.validate("What was revenue?", documents, draft)

    assert supported.answer == "Revenue was 120."
    assert guarded.insufficient_context is True
    assert guarded.source_ids == []
    assert "does not contain enough information" in guarded.answer
