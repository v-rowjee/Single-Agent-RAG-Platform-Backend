from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from app.agents.multi.retrieval_preparation import RetrievalPreparationAgent
from app.rag.retrieval.retriever import Retriever


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


class InvalidEmbeddingService(FakeEmbeddingService):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


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
        self.replaced_dataset_id: str | None = None
        self.rows: list[dict[str, object]] = []

    def replace_document_chunks(
        self,
        dataset_id: str,
        rows: list[dict[str, object]],
    ) -> int:
        self.replaced_dataset_id = dataset_id
        self.rows = rows
        return len(rows)


def test_index_documents_is_the_shared_session_scoped_indexer(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.retrieval.retriever.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    storage = IndexStorage()
    service = Retriever(storage=storage)  # type: ignore[arg-type]

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
    assert result["vector_size"] == 384
    assert storage.replaced_dataset_id == "session"
    assert storage.rows[0]["source_id"] == "revenue_kpi"
    assert storage.rows[0]["dataset_id"] == "session"
    assert storage.rows[0]["metadata"] == {
        "title": "Revenue KPI",
        "session_id": "session",
        "dataset_id": "session",
        "source_ids": ["revenue"],
        "chunk_index": 0,
        "source_id": "revenue_kpi",
        "document_type": "kpi",
    }


def test_index_documents_applies_shared_hard_chunk_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.retrieval.retriever.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    storage = IndexStorage()
    service = Retriever(storage=storage)  # type: ignore[arg-type]

    result = service.index_documents(
        session_id="session",
        dataset_id="session",
        retrieval_documents=[
            {
                "id": "long_insight",
                "content": "x" * 1_900,
                "document_type": "insight",
            }
        ],
    )

    assert result["status"] == "success"
    assert result["document_count"] == 1
    assert result["chunk_count"] == len(storage.rows)
    assert len(storage.rows) >= 3
    assert all(len(str(row["content"])) <= 800 for row in storage.rows)
    assert [row["chunk_index"] for row in storage.rows] == list(
        range(len(storage.rows))
    )


def test_index_documents_rejects_wrong_embedding_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.retrieval.retriever.get_embedding_service",
        lambda: InvalidEmbeddingService(),
    )
    service = Retriever(storage=IndexStorage())  # type: ignore[arg-type]

    result = service.index_documents(
        session_id="session",
        dataset_id="session",
        retrieval_documents=[
            {
                "id": "revenue_kpi",
                "content": "Revenue is 120.",
                "document_type": "kpi",
            }
        ],
    )

    assert result["status"] == "failed"
    assert result["message"] == (
        "The embedding model must return 384-dimensional vectors."
    )


def test_retrieve_defensively_removes_cross_session_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.retrieval.retriever.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    service = Retriever(storage=LeakyStorage())  # type: ignore[arg-type]

    documents = service.retrieve(
        session_id="requested-session",
        query="revenue",
    )

    assert [document.metadata["source_id"] for document in documents] == ["valid"]


def test_multi_retrieval_preparation_includes_prepared_row_evidence(
    tmp_path,
) -> None:
    prepared_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        [
            {"transaction_id": "TX-001", "region": "North", "revenue": 120},
            {"transaction_id": "TX-002", "region": "South", "revenue": 80},
        ]
    ).to_csv(prepared_path, index=False)

    result = asyncio.run(
        RetrievalPreparationAgent().run(
            prepared_dataset={
                "prepared_file_path": str(prepared_path),
                "file_name": "sales.csv",
                "dataset_profile": {
                    "row_count": 2,
                    "column_count": 3,
                    "currency": "GBP",
                },
                "primary_measures": ["revenue"],
                "dimension_candidates": ["transaction_id", "region"],
                "date_column": None,
            },
            kpi_trend_output=None,
            anomaly_output=None,
            forecasting_output=None,
            synthesis_output={},
        )
    )

    row_documents = [
        document
        for document in result.documents
        if document.document_type == "row_batch"
    ]
    assert row_documents
    assert "TX-001" in "\n".join(document.content for document in row_documents)
    assert "TX-002" in "\n".join(document.content for document in row_documents)
    summary = next(
        document
        for document in result.documents
        if document.document_type == "dataset_summary"
    )
    assert "Currency: GBP" in summary.content
    assert summary.metadata["currency"] == "GBP"


def test_multi_retrieval_preparation_indexes_dashboard_recommendations() -> None:
    result = asyncio.run(
        RetrievalPreparationAgent().run(
            prepared_dataset={},
            kpi_trend_output=None,
            anomaly_output=None,
            forecasting_output=None,
            synthesis_output={},
            dashboard_output={
                "dashboard": {
                    "recommendedActions": [
                        {
                            "id": "protect_margin",
                            "title": "Protect margin",
                            "description": "Review discounting in the South region.",
                            "priority": "high",
                            "sourceIds": ["anomaly_margin"],
                        }
                    ]
                }
            },
        )
    )

    recommendation = next(
        document
        for document in result.documents
        if document.id == "recommendation_protect_margin"
    )
    assert recommendation.document_type == "recommendation"
    assert "Review discounting" in recommendation.content
    assert recommendation.source_ids == ["anomaly_margin"]

