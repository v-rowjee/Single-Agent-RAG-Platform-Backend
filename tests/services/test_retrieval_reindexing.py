from __future__ import annotations

from typing import Any

import pytest

from app.services.retrieval_indexing_service import RetrievalIndexingService


class FakeStorage:
    def __init__(self) -> None:
        self.documents: dict[str, list[dict[str, object]]] = {}
        self.events: list[tuple[str, str]] = []
        self.statuses: list[tuple[str, str]] = []
        self.fail_delete = False

    def update_dataset_status(self, dataset_id: str, *, rag_status: str) -> None:
        self.statuses.append((dataset_id, rag_status))

    def delete_document_chunks(self, dataset_id: str) -> None:
        self.events.append(("delete", dataset_id))
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.documents[dataset_id] = []

    def insert_document_chunks(
        self,
        chunks: list[dict[str, object]],
        batch_size: int = 50,
    ) -> None:
        assert batch_size == 50
        dataset_id = str(chunks[0]["dataset_id"]) if chunks else ""
        self.events.append(("upsert", dataset_id))
        self.documents.setdefault(dataset_id, []).extend(chunks)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


def documents(*ids: str) -> list[dict[str, Any]]:
    return [
        {
            "id": item_id,
            "content": f"Evidence for {item_id}",
            "document_type": "kpi",
            "source_ids": [item_id],
        }
        for item_id in ids
    ]


@pytest.fixture
def indexing_service(monkeypatch: pytest.MonkeyPatch) -> tuple[RetrievalIndexingService, FakeStorage]:
    storage = FakeStorage()
    monkeypatch.setattr(
        "app.services.retrieval_indexing_service.get_embedding_service",
        lambda: FakeEmbeddings(),
    )
    return RetrievalIndexingService(storage=storage), storage  # type: ignore[arg-type]


def test_existing_session_documents_are_deleted_before_replacements(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service

    result = service.index_documents("session-a", "session-a", documents("new"))

    assert result["status"] == "success"
    assert storage.events == [("delete", "session-a"), ("upsert", "session-a")]


def test_delete_is_scoped_to_the_requested_session(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service

    service.index_documents("session-a", "session-a", documents("a"))

    assert ("delete", "session-a") in storage.events
    assert ("delete", "session-b") not in storage.events


def test_reindexing_replaces_instead_of_duplicating_documents(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service

    service.index_documents("session-a", "session-a", documents("old-one", "old-two"))
    service.index_documents("session-a", "session-a", documents("new-one"))

    assert storage.events.count(("delete", "session-a")) == 2
    assert [row["source_id"] for row in storage.documents["session-a"]] == ["new-one"]


def test_reindexing_one_session_preserves_other_sessions(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service

    service.index_documents("session-a", "session-a", documents("a-old"))
    service.index_documents("session-b", "session-b", documents("b-current"))
    service.index_documents("session-a", "session-a", documents("a-current"))

    assert [row["source_id"] for row in storage.documents["session-a"]] == ["a-current"]
    assert [row["source_id"] for row in storage.documents["session-b"]] == ["b-current"]


def test_delete_failure_does_not_insert_replacements(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service
    storage.fail_delete = True

    result = service.index_documents("session-a", "session-a", documents("new"))

    assert result["status"] == "failed"
    assert result["message"] == "Retrieval indexing failed."
    assert ("upsert", "session-a") not in storage.events
    assert storage.statuses[-1] == ("session-a", "failed")


def test_empty_replacement_set_removes_stale_session_documents(
    indexing_service: tuple[RetrievalIndexingService, FakeStorage],
) -> None:
    service, storage = indexing_service
    storage.documents["session-a"] = [{"source_id": "stale"}]

    result = service.index_documents("session-a", "session-a", [])

    assert result == {
        "status": "success",
        "document_count": 0,
        "indexed_count": 0,
        "failed_count": 0,
    }
    assert storage.documents["session-a"] == []
    assert storage.events == [("delete", "session-a")]
