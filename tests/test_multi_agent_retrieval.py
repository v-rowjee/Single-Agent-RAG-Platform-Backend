from __future__ import annotations

from typing import Any

from app.rag.rag_service import RagService


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class LeakyStorage:
    def match_document_chunks(
        self,
        dataset_id: str,
        query_embedding: list[float],
        match_count: int,
        match_threshold: float,
    ) -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": dataset_id,
                "source_id": "valid",
                "document_type": "kpi",
                "content": "Valid session evidence.",
                "metadata": {
                    "session_id": dataset_id,
                    "dataset_id": dataset_id,
                },
                "similarity": 0.9,
            },
            {
                "dataset_id": "other-session",
                "source_id": "leaked",
                "document_type": "kpi",
                "content": "Evidence from another session.",
                "metadata": {
                    "session_id": "other-session",
                    "dataset_id": "other-session",
                },
                "similarity": 0.99,
            },
        ]


def test_retrieve_for_session_defensively_removes_cross_session_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    service = RagService(storage=LeakyStorage())  # type: ignore[arg-type]

    documents = service.retrieve_for_session(
        session_id="requested-session",
        query="revenue",
    )

    assert [document.metadata["source_id"] for document in documents] == ["valid"]

