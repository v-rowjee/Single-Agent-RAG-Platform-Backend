from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.agents.multi.chat_agent import GroundedChatDraft
from app.api import business_intelligence as chat_api
from app.core.config import Settings
from app.rag.models import RetrievedDocument
from app.rag.rag_service import RagService
from app.schemas.business_intelligence import ChatRequest, ChatResponse
from app.services.business_intelligence_service import (
    BusinessIntelligenceService,
    SessionNotFoundError,
)
from app.services.supabase_service import DatasetRecord, MessageRecord


SESSION_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SESSION_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


class FakeStorage:
    def __init__(self, datasets: dict[str, DatasetRecord], rows: list[dict[str, object]]) -> None:
        self.datasets = datasets
        self.rows = rows
        self.vector_calls: list[dict[str, object]] = []
        self.messages: list[MessageRecord] = []

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        return self.datasets.get(dataset_id)

    def match_document_chunks(
        self,
        dataset_id: str,
        query_embedding: list[float],
        match_count: int,
        match_threshold: float,
    ) -> list[dict[str, object]]:
        self.vector_calls.append(
            {
                "dataset_id": dataset_id,
                "query_embedding": query_embedding,
                "match_count": match_count,
                "match_threshold": match_threshold,
            }
        )
        return self.rows

    def save_message(
        self,
        dataset_id: str,
        role: str,
        content: str,
        sources: list[str] | None = None,
    ) -> MessageRecord:
        message = MessageRecord(
            id=f"message-{len(self.messages) + 1}",
            dataset_id=dataset_id,
            role=role,
            content=content,
            sources=sources or [],
            created_at="2026-07-14T10:00:00Z",
        )
        self.messages.append(message)
        return message


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def dataset(session_id: str) -> DatasetRecord:
    return DatasetRecord(
        id=session_id,
        file_name="sales.csv",
        storage_path=f"{session_id}/sales.csv",
        mime_type="text/csv",
        file_size=32,
        file_hash="hash",
        description=None,
        status="ready",
        rag_status="ready",
        error_message=None,
    )


def retrieval_row(
    session_id: str,
    document_type: str,
    title: str,
    content: str,
) -> dict[str, object]:
    return {
        "dataset_id": session_id,
        "source_id": f"{document_type}-{session_id}",
        "document_type": document_type,
        "content": content,
        "similarity": 0.9,
        "metadata": {
            "session_id": session_id,
            "dataset_id": session_id,
            "title": title,
            "source_ids": [f"{document_type}-source"],
        },
    }


def configure_service(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
) -> tuple[BusinessIntelligenceService, FakeStorage]:
    storage = FakeStorage({SESSION_A: dataset(SESSION_A)}, rows)
    rag = RagService(storage=storage)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    monkeypatch.setattr("app.services.business_intelligence_service.rag_service", rag)
    return (
        BusinessIntelligenceService(
            storage=storage,  # type: ignore[arg-type]
            settings=Settings("", "", bi_pipeline_mode="multi"),
        ),
        storage,
    )


def test_chat_uses_session_scoped_vector_retrieval_and_canonical_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, storage = configure_service(
        monkeypatch,
        [
            retrieval_row(SESSION_A, "kpi", "Revenue KPI", "Revenue was 120."),
            retrieval_row(SESSION_B, "trend", "Other dataset trend", "Never use this."),
        ],
    )
    received: dict[str, object] = {}

    async def run(session_id, query, retrieved_documents):
        received.update({"session_id": session_id, "query": query, "documents": retrieved_documents})
        return GroundedChatDraft(
            answer="grounded answer",
            source_ids=["kpi-source"],
            insufficient_context=False,
        )

    monkeypatch.setattr("app.services.business_intelligence_service.chat_agent.run", run)

    def guard(query, retrieved_documents, draft):
        received["guarded_draft"] = draft
        return draft

    monkeypatch.setattr(
        "app.services.business_intelligence_service.chat_grounding_guardrail.validate",
        guard,
    )

    response = service.chat(SESSION_A, "What is the revenue trend?")

    assert isinstance(response, ChatResponse)
    assert response.answer == "grounded answer"
    assert response.grounding == "Retrieved dataset sources: `kpi-source`."
    assert storage.vector_calls[0]["dataset_id"] == SESSION_A
    assert received["session_id"] == SESSION_A
    assert [document.metadata["title"] for document in received["documents"]] == ["Revenue KPI"]
    assert received["guarded_draft"].source_ids == ["kpi-source"]


def test_chat_passes_compact_context_for_all_multi_agent_document_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        retrieval_row(SESSION_A, "kpi", "Revenue KPI", "Revenue rose by 10%."),
        retrieval_row(SESSION_A, "trend", "Revenue trend", "Trend is upward."),
        retrieval_row(SESSION_A, "anomaly", "Revenue anomaly", "A spike occurred."),
        retrieval_row(SESSION_A, "forecast", "Revenue forecast", "Forecast remains positive."),
    ]
    service, _ = configure_service(monkeypatch, rows)
    received: dict[str, object] = {}

    async def run(session_id, query, retrieved_documents):
        received["documents"] = retrieved_documents
        return GroundedChatDraft(
            answer="answer",
            source_ids=["kpi-source"],
            insufficient_context=False,
        )

    monkeypatch.setattr("app.services.business_intelligence_service.chat_agent.run", run)

    service.chat(SESSION_A, "Summarise revenue evidence")

    assert [document.metadata["document_type"] for document in received["documents"]] == [
        "kpi", "trend", "anomaly", "forecast"
    ]


def test_unknown_chat_session_preserves_not_found_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BusinessIntelligenceService(
        storage=FakeStorage({}, [])  # type: ignore[arg-type]
    )

    with pytest.raises(SessionNotFoundError):
        service.chat("missing-session", "What happened?")

    monkeypatch.setattr(chat_api, "business_intelligence_service", service)
    with pytest.raises(HTTPException) as error:
        chat_api.chat(ChatRequest(sessionId="missing-session", query="What happened?"))
    assert error.value.status_code == 404


def test_chat_returns_normal_response_when_no_documents_are_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = configure_service(monkeypatch, [])
    monkeypatch.setattr(
        "app.agents.multi.chat_agent._request_groq_draft",
        lambda **kwargs: pytest.fail("Groq should not run"),
    )

    response = service.chat(SESSION_A, "What is the revenue trend?")

    assert isinstance(response, ChatResponse)
    assert "does not contain enough information" in response.answer


def test_chat_handles_vector_retrieval_failure_without_unfiltered_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({SESSION_A: dataset(SESSION_A)}, [])
    calls = 0

    def fail_retrieval(**kwargs: object) -> list[RetrievedDocument]:
        nonlocal calls
        calls += 1
        raise RuntimeError("private vector failure")

    monkeypatch.setattr(
        "app.services.business_intelligence_service.rag_service",
        SimpleNamespace(retrieve_for_session=fail_retrieval),
    )
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
    )

    response = service.chat(SESSION_A, "What is the revenue trend?")

    assert calls == 1
    assert "could not answer" in response.answer
    assert "private vector failure" not in response.answer


def test_chat_handles_model_failure_with_safe_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = configure_service(
        monkeypatch,
        [retrieval_row(SESSION_A, "insight", "Revenue insight", "Revenue improved.")],
    )
    monkeypatch.setattr(
        "app.services.business_intelligence_service.chat_agent.run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private model failure")),
    )

    response = service.chat(SESSION_A, "What improved?")

    assert isinstance(response, ChatResponse)
    assert "could not answer" in response.answer
    assert "private model failure" not in response.answer


def test_chat_route_uses_canonical_request_and_response_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chat_api,
        "business_intelligence_service",
        SimpleNamespace(
            chat=lambda session_id, query: ChatResponse(
                answer="answer",
                grounding="Dataset overview.",
            )
        ),
    )

    response = chat_api.chat(ChatRequest(sessionId=SESSION_A, query="Question"))

    assert isinstance(response, ChatResponse)
    assert response.model_dump() == {
        "answer": "answer",
        "grounding": "Dataset overview.",
    }
