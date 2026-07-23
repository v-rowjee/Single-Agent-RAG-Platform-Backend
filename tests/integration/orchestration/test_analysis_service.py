from __future__ import annotations

import asyncio
import io
from types import MethodType
from typing import Any

import pytest
from fastapi import BackgroundTasks
from starlette.datastructures import Headers, UploadFile

from app.core.config import Settings
from app.schemas.api import DashboardResponse
from app.services import business_intelligence as service_module
from app.services.business_intelligence import (
    BusinessIntelligenceService,
    PipelineExecution,
)
from app.services.persistence.analysis import DatasetRecord


USER_ID = "59b3d0fc-2d4a-40a0-8bb1-99e19da406ee"


class UploadStorage:
    def __init__(self) -> None:
        self.saved_dashboards = 0
        self.saved_processing = 0
        self.status_updates: list[dict[str, Any]] = []

    def upload_file(self, storage_path: str, content: bytes, mime_type: str) -> None:
        return None

    def get_active_dataset(self, user_id: str) -> DatasetRecord | None:
        return None

    def create_dataset(self, **values: Any) -> DatasetRecord:
        return DatasetRecord(
            id=str(values["dataset_id"]),
            user_id=str(values["user_id"]),
            file_name=str(values["file_name"]),
            storage_path=str(values["storage_path"]),
            mime_type=str(values["mime_type"]),
            file_size=int(values["file_size"]),
            file_hash=str(values["file_hash"]),
            description=values.get("description"),
            status="processing",
            rag_status="pending",
            error_message=None,
            row_count=int(values["row_count"]),
            column_count=int(values["column_count"]),
        )

    def save_dashboard(self, **values: Any) -> None:
        self.saved_dashboards += 1

    def save_session_processing(self, **values: Any) -> None:
        self.saved_processing += 1

    def update_dataset_status(self, dataset_id: str, **values: Any) -> None:
        self.status_updates.append({"dataset_id": dataset_id, **values})

    def delete_file(self, storage_path: str) -> None:
        return None

    def delete_dataset(self, dataset_id: str, user_id: str) -> None:
        return None


class IndexingRag:
    def index_documents(self, **values: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "document_count": len(values["retrieval_documents"]),
            "indexed_count": len(values["retrieval_documents"]),
            "failed_count": 0,
        }


def test_dashboard_persists_before_background_retrieval_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = UploadStorage()

    class RecordingRag(IndexingRag):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def index_documents(self, **values: Any) -> dict[str, Any]:
            self.calls.append(values)
            return super().index_documents(**values)

    rag = RecordingRag()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
        rag=rag,
    )

    async def run_multi(
        self: BusinessIntelligenceService,
        dataset: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
    ) -> PipelineExecution:
        info = self._inspect_file(dataset.file_name, content or b"")
        return PipelineExecution(
            response=DashboardResponse.model_validate(
                self._build_placeholder_dashboard(dataset, info)
            ),
            retrieval_documents=[
                {
                    "id": "dataset_overview",
                    "content": "Revenue overview.",
                    "document_type": "dataset_overview",
                }
            ],
        )

    monkeypatch.setattr(
        service,
        "_run_multi_agent_pipeline",
        MethodType(run_multi, service),
    )
    background_tasks = BackgroundTasks()
    upload = UploadFile(
        filename="sales.csv",
        file=io.BytesIO(b"date,revenue\n2025-01-01,100\n"),
        headers=Headers({"content-type": "text/csv"}),
    )

    response = asyncio.run(
        service.create_analysis(
            upload,
            user_id=USER_ID,
            background_tasks=background_tasks,
        )
    )

    assert response["status"] == "success"
    assert rag.calls == []
    assert storage.saved_dashboards == 1
    assert storage.status_updates[-1]["status"] == "ready"
    assert storage.status_updates[-1]["rag_status"] == "indexing"
    assert len(background_tasks.tasks) == 1

    asyncio.run(background_tasks())

    assert len(rag.calls) == 1
    assert storage.saved_dashboards == 2
    assert storage.status_updates[-1]["status"] == "ready"
    assert storage.status_updates[-1]["rag_status"] == "ready"


def test_multi_upload_uses_service_owned_persistence_and_never_single_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = UploadStorage()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
        rag=IndexingRag(),
    )
    multi_calls: list[str] = []

    async def run_multi(
        self: BusinessIntelligenceService,
        dataset: DatasetRecord,
        content: bytes | None = None,
    ) -> DashboardResponse:
        multi_calls.append(dataset.id)
        info = self._inspect_file(dataset.file_name, content or b"")
        return DashboardResponse.model_validate(
            self._build_placeholder_dashboard(dataset, info)
        )

    async def single_must_not_run(*args: Any, **kwargs: Any) -> DashboardResponse:
        raise AssertionError("Single-agent analysis must not run in multi mode.")

    def single_rag_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Single-agent RAG indexing must not run in multi mode.")

    monkeypatch.setattr(
        service,
        "_run_multi_agent_pipeline",
        MethodType(run_multi, service),
    )
    monkeypatch.setattr(service, "_run_single_agent_pipeline", single_must_not_run)
    monkeypatch.setattr(service, "_try_index_rag", single_rag_must_not_run)

    upload = UploadFile(
        filename="sales.csv",
        file=io.BytesIO(b"date,revenue\n2025-01-01,100\n"),
        headers=Headers({"content-type": "text/csv"}),
    )
    response = asyncio.run(service.create_analysis(upload, user_id=USER_ID))

    assert response["status"] == "success"
    assert len(multi_calls) == 1
    assert storage.saved_dashboards == 1
    assert storage.saved_processing == 1
    assert len(storage.status_updates) == 2
    assert storage.status_updates[0]["rag_status"] == "indexing"
    assert storage.status_updates[-1]["status"] == "ready"
    assert storage.status_updates[-1]["rag_status"] == "ready"


def test_failed_graph_returns_failed_dashboard_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = DatasetRecord(
        id="9d719abc-9e09-4c14-b2d6-ed8308a1b85d",
        user_id=USER_ID,
        file_name="sales.csv",
        storage_path="session/sales.csv",
        mime_type="text/csv",
        file_size=30,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )
    service = BusinessIntelligenceService(
        storage=UploadStorage(),  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
    )

    class FailedWorkflowGraph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            return {
                **state,
                "workflow_status": "failed",
                "dashboard_output": {
                    "status": "partial",
                    "sessionId": dataset.id,
                    "dashboard": None,
                    "warnings": [],
                    "errors": [],
                },
            }

    monkeypatch.setattr(
        service_module,
        "analysis_graph",
        FailedWorkflowGraph(),
    )

    execution = asyncio.run(
        service._run_multi_agent_pipeline(
            dataset,
            b"date,revenue\n2025-01-01,100\n",
        )
    )

    assert execution.response.status == "failed"
    assert execution.response.dashboard is None
    assert execution.response.errors[0].code == "MULTI_AGENT_PIPELINE_FAILED"
