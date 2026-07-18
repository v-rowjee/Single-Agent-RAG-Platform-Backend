from __future__ import annotations

from typing import Any

from app.rag.rag_service import RagService


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


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


class IndexStorage:
    def __init__(self) -> None:
        self.deleted_dataset_id: str | None = None
        self.rows: list[dict[str, object]] = []

    def delete_document_chunks(self, dataset_id: str) -> None:
        self.deleted_dataset_id = dataset_id

    def insert_document_chunks(
        self,
        rows: list[dict[str, object]],
        batch_size: int,
    ) -> None:
        assert batch_size == 50
        self.rows = rows


def test_index_documents_is_the_shared_session_scoped_indexer(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    storage = IndexStorage()
    service = RagService(storage=storage)  # type: ignore[arg-type]

    result = service.index_documents(
        session_id="session",
        dataset_id="session",
        retrieval_documents=[
            {
                "id": "revenue_kpi",
                "content": "Revenue is 120.",
                "document_type": "kpi",
                "source_ids": ["revenue"],
                "metadata": {"title": "Revenue KPI"},
            }
        ],
    )

    assert result["status"] == "success"
    assert result["indexed_count"] == 1
    assert storage.deleted_dataset_id == "session"
    assert storage.rows[0]["source_id"] == "revenue_kpi"
    assert storage.rows[0]["dataset_id"] == "session"
    assert storage.rows[0]["metadata"] == {
        "title": "Revenue KPI",
        "session_id": "session",
        "dataset_id": "session",
        "source_ids": ["revenue"],
        "chunk_index": 0,
    }


def test_retrieve_defensively_removes_cross_session_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    service = RagService(storage=LeakyStorage())  # type: ignore[arg-type]

    documents = service.retrieve(
        session_id="requested-session",
        query="revenue",
    )

    assert [document.metadata["source_id"] for document in documents] == ["valid"]

