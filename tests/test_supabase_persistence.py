from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.agents.multi.chat_agent import GroundedChatDraft
from app.rag.models import RagDocument, RetrievedDocument
from app.rag.rag_service import RagService
from app.schemas.business_intelligence import DashboardResponse
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import DatasetRecord, MessageRecord


DATASET_ID = "15dc222f-bdfa-4e32-9252-19d9f57cc28a"


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes, str]] = []
        self.deleted_files: list[str] = []
        self.datasets: dict[str, DatasetRecord] = {}
        self.dashboards: dict[str, dict[str, object]] = {}
        self.messages: list[MessageRecord] = []
        self.deleted_chunk_dataset: str | None = None
        self.inserted_chunks: list[dict[str, object]] = []
        self.rpc_calls: list[dict[str, object]] = []
        self.fail_dataset_insert = False

    def upload_file(self, storage_path: str, content: bytes, mime_type: str) -> None:
        self.uploads.append((storage_path, content, mime_type))

    def download_file(self, storage_path: str) -> bytes:
        return b"Region,Revenue\nNorth,10\nSouth,20\n"

    def delete_file(self, storage_path: str) -> None:
        self.deleted_files.append(storage_path)

    def create_dataset(
        self,
        dataset_id: str,
        file_name: str,
        storage_path: str,
        mime_type: str,
        file_size: int,
        file_hash: str,
        description: str | None,
    ) -> DatasetRecord:
        if self.fail_dataset_insert:
            raise RuntimeError("database unavailable")
        record = dataset_record(
            dataset_id=dataset_id,
            file_name=file_name,
            storage_path=storage_path,
            mime_type=mime_type,
            file_size=file_size,
            file_hash=file_hash,
            description=description,
        )
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        return self.datasets.get(dataset_id)

    def update_dataset_status(self, dataset_id: str, **kwargs: object) -> None:
        return None

    def save_dashboard(
        self,
        dataset_id: str,
        status: str,
        response: dict[str, object],
    ) -> object:
        self.dashboards[dataset_id] = {
            "status": status,
            "response": response,
        }
        return SimpleNamespace()

    def get_dashboard(self, dataset_id: str) -> object | None:
        item = self.dashboards.get(dataset_id)
        if item is None:
            return None
        return SimpleNamespace(response=item["response"])

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
            created_at=f"2026-07-13T10:00:0{len(self.messages)}Z",
        )
        self.messages.append(message)
        return message

    def get_recent_messages(
        self,
        dataset_id: str,
        limit: int = 6,
    ) -> list[MessageRecord]:
        return [item for item in self.messages if item.dataset_id == dataset_id][-limit:]

    def delete_document_chunks(self, dataset_id: str) -> None:
        self.deleted_chunk_dataset = dataset_id

    def insert_document_chunks(
        self,
        chunks: list[dict[str, object]],
        batch_size: int = 50,
    ) -> None:
        self.inserted_chunks.extend(chunks)

    def match_document_chunks(
        self,
        dataset_id: str,
        query_embedding: list[float],
        match_count: int = 12,
        match_threshold: float = 0.2,
    ) -> list[dict[str, object]]:
        self.rpc_calls.append(
            {
                "dataset_id": dataset_id,
                "query_embedding": query_embedding,
                "match_count": match_count,
                "match_threshold": match_threshold,
            }
        )
        return [
            {
                "source_id": "dataset_overview",
                "document_type": "dataset_overview",
                "content": "Rows: 2. Revenue exists.",
                "metadata": {"source_id": "dataset_overview"},
                "similarity": 0.91,
            }
        ]


class FakeEmbeddingService:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.2] * 384


def dataset_record(
    dataset_id: str = DATASET_ID,
    file_name: str = "sales.csv",
    storage_path: str | None = None,
    mime_type: str = "text/csv",
    file_size: int = 32,
    file_hash: str = "hash",
    description: str | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        id=dataset_id,
        file_name=file_name,
        storage_path=storage_path or f"{dataset_id}/{file_name}",
        mime_type=mime_type,
        file_size=file_size,
        file_hash=file_hash,
        description=description,
        status="processing",
        rag_status="pending",
        error_message=None,
    )


def dashboard_response(session_id: str = DATASET_ID) -> DashboardResponse:
    dataset = dataset_record(dataset_id=session_id)
    service = BusinessIntelligenceService(storage=FakeStorage())  # type: ignore[arg-type]
    payload = service._build_placeholder_dashboard(
        dataset=dataset,
        dataset_info={
            "rowCount": 2,
            "columnCount": 2,
            "measures": ["Revenue"],
            "dimensions": ["Region"],
            "missingValueCount": 0,
            "duplicateRowCount": 0,
            "completenessPercent": 100.0,
        },
    )
    return DashboardResponse.model_validate(payload)


def upload_file() -> UploadFile:
    return UploadFile(
        filename="sales.csv",
        file=io.BytesIO(b"Region,Revenue\nNorth,10\nSouth,20\n"),
        headers=Headers({"content-type": "text/csv"}),
    )


def test_upload_persists_file_dataset_dashboard_and_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    service = BusinessIntelligenceService(storage=storage)  # type: ignore[arg-type]
    monkeypatch.setattr(
        service,
        "_generate_dashboard_from_bytes",
        lambda dataset, content: dashboard_response(dataset.id),
    )
    monkeypatch.setattr(service, "_try_index_rag", lambda dataset, content: None)

    result = asyncio.run(service.create_analysis(upload_file()))

    assert set(result) == {"status", "sessionId", "fileName", "message"}
    assert result["status"] == "success"
    assert storage.uploads
    storage_path, content, mime_type = storage.uploads[0]
    assert storage_path == f"{result['sessionId']}/sales.csv"
    assert content.startswith(b"Region,Revenue")
    assert mime_type == "text/csv"
    assert result["sessionId"] in storage.datasets
    assert storage.dashboards[result["sessionId"]]["status"] == "partial"
    assert not (Path("app/storage") / str(result["sessionId"])).exists()


def test_dashboard_endpoint_reads_saved_dashboard_without_local_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    storage.datasets[DATASET_ID] = dataset_record()
    storage.dashboards[DATASET_ID] = {
        "status": "partial",
        "response": dashboard_response().model_dump(mode="json"),
    }
    service = BusinessIntelligenceService(storage=storage)  # type: ignore[arg-type]
    monkeypatch.setattr(
        service,
        "_generate_dashboard_from_bytes",
        lambda dataset, content: pytest.fail("dashboard should not regenerate"),
    )

    result = asyncio.run(service.get_dashboard(DATASET_ID))

    assert result.sessionId == DATASET_ID
    assert result.status == "partial"


def test_chat_messages_are_saved_and_recent_history_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    storage.datasets[DATASET_ID] = dataset_record()
    service = BusinessIntelligenceService(storage=storage)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "app.services.business_intelligence_service.rag_service.retrieve_for_session",
        lambda **kwargs: [
            RetrievedDocument(
                page_content="Revenue increased.",
                metadata={
                    "source_id": "dataset_overview",
                    "document_type": "dataset_summary",
                    "session_id": DATASET_ID,
                    "dataset_id": DATASET_ID,
                },
                score=0.9,
            )
        ],
    )
    async def run(session_id, query, retrieved_documents):
        return GroundedChatDraft(
            answer="answer",
            source_ids=["dataset_overview"],
            insufficient_context=False,
        )

    monkeypatch.setattr(
        "app.services.business_intelligence_service.chat_agent.run",
        run,
    )

    result = service.chat(DATASET_ID, "What is revenue?")

    assert result.response == "answer"
    assert [message.role for message in storage.messages] == ["user", "assistant"]
    assert storage.messages[1].sources == ["dataset_overview"]


def test_rag_chunks_are_inserted_and_retrieval_calls_rpc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("Region,Revenue\nNorth,10\nSouth,20\n", encoding="utf-8")
    storage = FakeStorage()
    rag = RagService(storage=storage)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )

    status = rag.index_dataset(
        agent_input=SimpleNamespace(
            sessionId=DATASET_ID,
            filePath=str(csv_path),
            fileName="sales.csv",
        ),
        profile={
            "summary": {
                "measures": ["Revenue"],
                "dimensions": ["Region"],
                "quality": {},
            }
        },
        force=True,
    )
    results = rag.retrieve(
        agent_input=SimpleNamespace(sessionId=DATASET_ID),
        query="revenue",
    )

    assert status.collection_name == "document_chunks"
    assert storage.deleted_chunk_dataset == DATASET_ID
    assert storage.inserted_chunks
    assert len(storage.inserted_chunks[0]["embedding"]) == 384
    assert storage.rpc_calls[0]["match_threshold"] == 0.2
    assert results[0].metadata["source_id"] == "dataset_overview"


def test_storage_upload_is_removed_when_dataset_insert_fails() -> None:
    storage = FakeStorage()
    storage.fail_dataset_insert = True
    service = BusinessIntelligenceService(storage=storage)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        asyncio.run(service.create_analysis(upload_file()))

    assert storage.deleted_files == [storage.uploads[0][0]]


def test_rag_failure_marks_rag_failed_without_deleting_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        storage,
        "update_dataset_status",
        lambda dataset_id, **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        "app.agents.single.business_intelligence_agent.business_intelligence_agent.profile_for_session",
        lambda agent_input: {},
    )
    monkeypatch.setattr(
        "app.rag.rag_service.rag_service.index_dataset",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("index failed")),
    )
    service = BusinessIntelligenceService(storage=storage)  # type: ignore[arg-type]

    service._try_index_rag(
        dataset=dataset_record(),
        content=b"Region,Revenue\nNorth,10\n",
    )

    assert {"rag_status": "indexing"} in updates
    assert updates[-1]["rag_status"] == "failed"


def test_document_builder_can_be_stubbed_for_chunk_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("Region,Revenue\nNorth,10\n", encoding="utf-8")
    storage = FakeStorage()
    rag = RagService(storage=storage)  # type: ignore[arg-type]
    rag._builder = SimpleNamespace(
        build=lambda **kwargs: [
            RagDocument(
                page_content="Revenue by region.",
                metadata={
                    "source_id": "stable_source",
                    "document_type": "numeric_summary",
                    "chunk_index": 0,
                },
            )
        ]
    )
    monkeypatch.setattr(
        "app.rag.rag_service.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )

    rag.index_dataset(
        agent_input=SimpleNamespace(
            sessionId=DATASET_ID,
            filePath=str(csv_path),
            fileName="sales.csv",
        ),
        profile={},
        force=True,
    )

    assert storage.inserted_chunks == [
        {
            "dataset_id": DATASET_ID,
            "source_id": "stable_source",
            "document_type": "numeric_summary",
            "chunk_index": 0,
            "content": "Revenue by region.",
            "metadata": {
                "source_id": "stable_source",
                "document_type": "numeric_summary",
                "chunk_index": 0,
            },
            "embedding": [0.1] * 384,
        }
    ]
