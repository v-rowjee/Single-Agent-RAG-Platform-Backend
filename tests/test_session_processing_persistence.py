from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services import business_intelligence_service as service_module
from app.services.business_intelligence_persistence_service import (
    BusinessIntelligencePersistenceService,
)
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import DatasetRecord


SESSION_ID = "9d719abc-9e09-4c14-b2d6-ed8308a1b85d"
USER_ID = "59b3d0fc-2d4a-40a0-8bb1-99e19da406ee"


class PersistenceStorage:
    def __init__(self) -> None:
        self.processing: dict[str, Any] | None = None
        self.dashboard: dict[str, Any] | None = None
        self.status: dict[str, Any] | None = None

    def save_session_processing(self, **values: Any) -> None:
        self.processing = values

    def save_dashboard(self, **values: Any) -> None:
        self.dashboard = values

    def update_dataset_status(self, dataset_id: str, **values: Any) -> None:
        self.status = {"dataset_id": dataset_id, **values}


def test_persistence_stores_processing_metadata_without_temporary_paths() -> None:
    storage = PersistenceStorage()
    service = BusinessIntelligencePersistenceService(storage=storage)  # type: ignore[arg-type]

    result = service.persist_workflow(
        {
            "session_id": SESSION_ID,
            "dataset_id": SESSION_ID,
            "dashboard_output": {
                "status": "failed",
                "sessionId": SESSION_ID,
                "dashboard": None,
                "warnings": [],
                "errors": [],
            },
            "generic_cleaning_report": {
                "cleaned_file_path": "C:/temp/processing/generic_cleaned_dataset.csv",
                "cleaned_row_count": 3,
            },
            "prepared_dataset": {
                "prepared_file_path": "C:/temp/processing/prepared_dataset.csv",
                "temporal_dataset_path": "C:/temp/processing/prepared_temporal_dataset.csv",
                "dataset_profile": {"row_count": 3},
                "cleaning_report": {
                    "cleaned_file_path": "C:/temp/processing/generic_cleaned_dataset.csv"
                },
            },
        }
    )

    assert result["status"] == "success"
    assert storage.processing == {
        "dataset_id": SESSION_ID,
        "workflow_status": "failed",
        "generic_cleaning_report": {"cleaned_row_count": 3},
        "prepared_dataset": {
            "dataset_profile": {"row_count": 3},
            "cleaning_report": {},
        },
    }
    assert storage.dashboard is not None
    workflow = storage.dashboard["response"]["workflow"]
    assert "prepared_file_path" not in workflow["prepared_dataset"]
    assert "temporal_dataset_path" not in workflow["prepared_dataset"]
    assert "cleaned_file_path" not in workflow["generic_cleaning_report"]


class WorkspaceGraph:
    def __init__(self) -> None:
        self.workspace: Path | None = None

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.workspace = Path(state["working_directory"])
        assert self.workspace.is_dir()
        assert "app/storage/sessions" not in self.workspace.as_posix()
        return {
            "workflow_status": "partial",
            "persistence_result": {"status": "success"},
            "dashboard_output": {
                "status": "failed",
                "sessionId": state["session_id"],
                "dashboard": None,
                "warnings": [],
                "errors": [],
            },
        }


class PipelineStorage:
    def download_file(self, storage_path: str) -> bytes:
        raise AssertionError("The supplied content should be used.")


def test_multi_agent_workspace_is_temporary_and_not_session_storage(
    monkeypatch: Any,
) -> None:
    graph = WorkspaceGraph()
    monkeypatch.setattr(service_module, "business_intelligence_graph", graph)
    service = BusinessIntelligenceService(
        storage=PipelineStorage(),  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="multi"),
    )
    dataset = DatasetRecord(
        id=SESSION_ID,
        user_id=USER_ID,
        file_name="sales.csv",
        storage_path=f"{SESSION_ID}/sales.csv",
        mime_type="text/csv",
        file_size=21,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )

    response = asyncio.run(
        service._run_multi_agent_pipeline(
            dataset,
            b"date,revenue\n2026-01-01,10\n",
        )
    )

    assert response.status == "failed"
    assert graph.workspace is not None
    assert not graph.workspace.exists()
