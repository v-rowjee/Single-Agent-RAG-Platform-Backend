from __future__ import annotations

import asyncio
import io
from dataclasses import replace
from types import MethodType
from typing import Any

import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.config import Settings
from app.schemas.business_intelligence import DashboardResponse
from app.services.business_intelligence_service import (
    BusinessIntelligenceService,
    InvalidUploadError,
    PipelineExecution,
)
from app.services.supabase_service import AnalysisSessionRecord, DatasetRecord


USER_ID = "59b3d0fc-2d4a-40a0-8bb1-99e19da406ee"


class WorkspaceStorage:
    def __init__(self) -> None:
        self.sessions: dict[str, AnalysisSessionRecord] = {}
        self.datasets: dict[str, DatasetRecord] = {}
        self.files: dict[str, bytes] = {}
        self.dashboard: dict[str, Any] | None = None
        self.processing: dict[str, Any] | None = None

    def get_active_session(self, user_id: str) -> AnalysisSessionRecord | None:
        return next(
            (session for session in self.sessions.values() if session.user_id == user_id),
            None,
        )

    def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> AnalysisSessionRecord | None:
        session = self.sessions.get(session_id)
        return session if session and session.user_id == user_id else None

    def create_session(self, **values: Any) -> AnalysisSessionRecord:
        session = AnalysisSessionRecord(
            id=values["session_id"],
            user_id=values["user_id"],
            description=values.get("description"),
            status="processing",
            rag_status="pending",
            error_message=None,
        )
        self.sessions[session.id] = session
        return session

    def create_dataset(self, **values: Any) -> DatasetRecord:
        dataset = DatasetRecord(
            id=values["dataset_id"],
            session_id=values["session_id"],
            user_id=values["user_id"],
            file_name=values["file_name"],
            storage_path=values["storage_path"],
            mime_type=values["mime_type"],
            file_size=values["file_size"],
            file_hash=values["file_hash"],
            description=values.get("description"),
            status="processing",
            rag_status="pending",
            error_message=None,
            row_count=values["row_count"],
            column_count=values["column_count"],
        )
        self.datasets[dataset.id] = dataset
        return dataset

    def get_session_datasets(
        self,
        session_id: str,
        user_id: str,
    ) -> list[DatasetRecord]:
        return [
            dataset
            for dataset in self.datasets.values()
            if dataset.session_id == session_id and dataset.user_id == user_id
        ]

    def upload_file(self, storage_path: str, content: bytes, mime_type: str) -> None:
        self.files[storage_path] = content

    def download_file(self, storage_path: str) -> bytes:
        return self.files[storage_path]

    def delete_file(self, storage_path: str) -> None:
        self.files.pop(storage_path, None)

    def delete_session(self, session_id: str, user_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.datasets = {
            dataset_id: dataset
            for dataset_id, dataset in self.datasets.items()
            if dataset.session_id != session_id
        }
        self.dashboard = None
        self.processing = None

    def update_session_status(self, session_id: str, **values: Any) -> None:
        session = self.sessions[session_id]
        self.sessions[session_id] = replace(
            session,
            **{
                key: value
                for key, value in values.items()
                if key in {"status", "rag_status", "error_message"}
            },
        )

    def save_dashboard(self, **values: Any) -> None:
        self.dashboard = values

    def get_dashboard(self, session_id: str) -> None:
        return None

    def save_session_processing(self, **values: Any) -> None:
        self.processing = values


class WorkspaceRag:
    def __init__(self) -> None:
        self.indexed: dict[str, Any] | None = None

    def index_documents(self, **values: Any) -> dict[str, Any]:
        self.indexed = values
        return {
            "status": "success",
            "document_count": len(values["retrieval_documents"]),
            "chunk_count": len(values["retrieval_documents"]),
            "indexed_count": len(values["retrieval_documents"]),
            "failed_count": 0,
        }


def upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(
        filename=name,
        file=io.BytesIO(content),
        headers=Headers({"content-type": "text/csv"}),
    )


def test_mixed_schema_batch_creates_one_workspace_and_uses_every_dataset() -> None:
    storage = WorkspaceStorage()
    rag = WorkspaceRag()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
        rag=rag,
    )
    analyzed: list[str] = []

    async def run_dataset(
        self: BusinessIntelligenceService,
        dataset: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
    ) -> PipelineExecution:
        analyzed.append(dataset.id)
        info = self._inspect_file(dataset.file_name, content or b"")
        response = DashboardResponse.model_validate(
            self._build_placeholder_dashboard(dataset, info)
        )
        return PipelineExecution(
            response=response,
            workflow={"dataset_id": dataset.id},
            retrieval_documents=[
                {
                    "id": "dataset_overview",
                    "content": f"Overview for {dataset.file_name}",
                    "document_type": "dataset_overview",
                }
            ],
        )

    service._run_multi_agent_pipeline = MethodType(run_dataset, service)
    response = asyncio.run(
        service.create_analysis(
            [
                upload("sales.csv", b"region,revenue\nNorth,100\n"),
                upload("inventory.csv", b"sku,stock\nA-1,8\n"),
            ],
            user_id=USER_ID,
            description="Analyse sales and inventory",
        )
    )

    assert len(response["datasetIds"]) == 2
    assert set(analyzed) == set(response["datasetIds"])
    assert len(storage.files) == 2
    assert all(
        f"/{dataset_id}/" in path
        for dataset_id, path in zip(
            response["datasetIds"],
            storage.files,
            strict=True,
        )
    )

    details = service.get_active_dataset_details(USER_ID)
    assert details["sessionId"] == response["sessionId"]
    assert [item["fileName"] for item in details["datasets"]] == [
        "sales.csv",
        "inventory.csv",
    ]
    assert storage.dashboard is not None
    summaries = storage.dashboard["response"]["dashboard"]["datasetSummaries"]
    assert [summary["fileName"] for summary in summaries] == [
        "sales.csv",
        "inventory.csv",
    ]
    assert rag.indexed is not None
    documents = rag.indexed["retrieval_documents"]
    assert {document["metadata"]["file_name"] for document in documents} == {
        "sales.csv",
        "inventory.csv",
    }
    assert len({document["id"] for document in documents}) == 2

    first_preview = service.get_dataset_preview(
        USER_ID,
        response["datasetIds"][0],
        1,
        50,
    )
    second_preview = service.get_dataset_preview(
        USER_ID,
        response["datasetIds"][1],
        1,
        50,
    )
    assert first_preview["columns"] == ["region", "revenue"]
    assert second_preview["columns"] == ["sku", "stock"]

    datasets = storage.get_session_datasets(response["sessionId"], USER_ID)
    selected, ambiguous = service._select_chat_datasets(
        "What is the total revenue?",
        datasets,
    )
    assert selected is not None and selected.file_name == "sales.csv"
    assert ambiguous == []
    selected, ambiguous = service._select_chat_datasets(
        "What is the total?",
        datasets,
    )
    assert selected is None
    assert {dataset.file_name for dataset in ambiguous} == {
        "sales.csv",
        "inventory.csv",
    }

    service.reset_active_dataset(USER_ID)
    assert storage.sessions == {}
    assert storage.datasets == {}
    assert storage.files == {}


def test_batch_validation_rejects_duplicate_content_before_persistence() -> None:
    storage = WorkspaceStorage()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
    )
    content = b"region,revenue\nNorth,100\n"

    with pytest.raises(InvalidUploadError, match="duplicates another file"):
        asyncio.run(
            service.create_analysis(
                [
                    upload("sales.csv", content),
                    upload("sales-copy.csv", content),
                ],
                user_id=USER_ID,
            )
        )

    assert storage.sessions == {}
    assert storage.datasets == {}
    assert storage.files == {}
