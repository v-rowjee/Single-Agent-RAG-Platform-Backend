from __future__ import annotations

import asyncio
import io
from types import MethodType
from typing import Any

import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.config import Settings
from app.schemas.business_intelligence import DashboardResponse
from app.services import business_intelligence_service as service_module
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import DatasetRecord


class UploadStorage:
    def __init__(self) -> None:
        self.saved_dashboards = 0
        self.status_updates: list[dict[str, Any]] = []

    def upload_file(self, storage_path: str, content: bytes, mime_type: str) -> None:
        return None

    def create_dataset(self, **values: Any) -> DatasetRecord:
        return DatasetRecord(
            id=str(values["dataset_id"]),
            file_name=str(values["file_name"]),
            storage_path=str(values["storage_path"]),
            mime_type=str(values["mime_type"]),
            file_size=int(values["file_size"]),
            file_hash=str(values["file_hash"]),
            description=values.get("description"),
            status="processing",
            rag_status="pending",
            error_message=None,
        )

    def save_dashboard(self, **values: Any) -> None:
        self.saved_dashboards += 1

    def update_dataset_status(self, dataset_id: str, **values: Any) -> None:
        self.status_updates.append({"dataset_id": dataset_id, **values})

    def delete_file(self, storage_path: str) -> None:
        return None


def test_multi_upload_uses_graph_owned_persistence_and_never_single_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = UploadStorage()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
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
    response = asyncio.run(service.create_analysis(upload))

    assert response["status"] == "success"
    assert len(multi_calls) == 1
    assert storage.saved_dashboards == 0
    assert storage.status_updates[-1]["status"] == "ready"


def test_failed_graph_persistence_returns_failed_dashboard_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = DatasetRecord(
        id="9d719abc-9e09-4c14-b2d6-ed8308a1b85d",
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

    class FailedPersistenceGraph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            return {
                **state,
                "workflow_status": "failed",
                "persistence_result": {"status": "failed"},
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
        "business_intelligence_graph",
        FailedPersistenceGraph(),
    )

    response = asyncio.run(
        service._run_multi_agent_pipeline(
            dataset,
            b"date,revenue\n2025-01-01,100\n",
        )
    )

    assert response.status == "failed"
    assert response.dashboard is None
    assert response.errors[0].code == "MULTI_AGENT_PIPELINE_FAILED"
