from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.schemas.business_intelligence import DashboardResponse
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import DatasetRecord


SESSION_ID = "15dc222f-bdfa-4e32-9252-19d9f57cc28a"


class FakeStorage:
    def __init__(self, dataset: DatasetRecord) -> None:
        self.dataset = dataset

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        return self.dataset if dataset_id == self.dataset.id else None

    def get_dashboard(self, dataset_id: str) -> None:
        return None

    def download_file(self, storage_path: str) -> bytes:
        return b"Region,Revenue\nNorth,10\nSouth,20\n"


def dataset_record() -> DatasetRecord:
    return DatasetRecord(
        id=SESSION_ID,
        file_name="sales.csv",
        storage_path=f"{SESSION_ID}/sales.csv",
        mime_type="text/csv",
        file_size=32,
        file_hash="hash",
        description="Monthly regional sales performance.",
        status="processing",
        rag_status="pending",
        error_message=None,
    )


def service() -> BusinessIntelligenceService:
    dataset = dataset_record()
    return BusinessIntelligenceService(  # type: ignore[arg-type]
        storage=FakeStorage(dataset),
        settings=Settings("", "", bi_pipeline_mode="multi"),
    )


def valid_dashboard_output(
    service: BusinessIntelligenceService,
    *,
    status: str,
    session_id: str = "wrong-session-id",
) -> dict[str, object]:
    payload = service._build_placeholder_dashboard(
        dataset=dataset_record(),
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
    payload["status"] = status
    payload["sessionId"] = session_id
    return payload


def mock_graph(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, object] | Exception,
) -> AsyncMock:
    async def invoke(initial_state: dict[str, object]) -> dict[str, object]:
        graph.source_content = Path(  # type: ignore[attr-defined]
            str(initial_state["uploaded_file_path"])
        ).read_bytes()
        if isinstance(result, Exception):
            raise result
        return result

    graph = AsyncMock(side_effect=invoke)
    monkeypatch.setattr(
        "app.services.business_intelligence_service.business_intelligence_graph.ainvoke",
        graph,
    )
    return graph


def test_multi_agent_pipeline_returns_validated_success_and_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bi_service = service()
    output = valid_dashboard_output(bi_service, status="success")
    graph = mock_graph(monkeypatch, {"dashboard_output": output})

    response = asyncio.run(bi_service.get_dashboard(SESSION_ID))

    assert isinstance(response, DashboardResponse)
    assert response.status == "success"
    assert response.sessionId == SESSION_ID
    assert response.dashboard == DashboardResponse.model_validate(output).dashboard
    state = graph.await_args.args[0]
    assert state["session_id"] == SESSION_ID
    assert state["dataset_id"] == SESSION_ID
    assert state["business_description"] == "Monthly regional sales performance."
    assert Path(state["uploaded_file_path"]).name == "sales.csv"
    assert graph.source_content.startswith(b"Region,Revenue")  # type: ignore[attr-defined]
    assert state["warnings"] == []
    assert state["errors"] == []
    assert state["completed_agents"] == []
    assert state["failed_agents"] == []
    assert state["skipped_agents"] == []


def test_multi_agent_pipeline_returns_partial_dashboard_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bi_service = service()
    output = valid_dashboard_output(bi_service, status="partial")
    output["warnings"] = [
        {
            "code": "FORECAST_UNAVAILABLE",
            "message": "Forecasting was unavailable for this dataset.",
            "component": "forecasting",
            "recoverable": True,
        }
    ]
    mock_graph(monkeypatch, {"dashboard_output": output})

    response = asyncio.run(bi_service._run_multi_agent_pipeline(dataset_record()))

    assert response.status == "partial"
    assert response.dashboard is not None
    assert response.warnings[0].code == "FORECAST_UNAVAILABLE"


def test_multi_agent_pipeline_returns_deliberate_failed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bi_service = service()
    output = {
        "status": "failed",
        "sessionId": "wrong-session-id",
        "dashboard": None,
        "warnings": [],
        "errors": [
            {
                "code": "DATA_PREPARATION_FAILED",
                "message": "The uploaded file could not be prepared.",
                "component": "data_preparation",
                "recoverable": False,
            }
        ],
    }
    mock_graph(monkeypatch, {"dashboard_output": output})

    response = asyncio.run(bi_service._run_multi_agent_pipeline(dataset_record()))

    assert response.status == "failed"
    assert response.sessionId == SESSION_ID
    assert response.errors[0].code == "DATA_PREPARATION_FAILED"


def test_multi_agent_pipeline_converts_unexpected_graph_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bi_service = service()
    mock_graph(monkeypatch, RuntimeError("raw graph failure must stay private"))

    response = asyncio.run(bi_service._run_multi_agent_pipeline(dataset_record()))

    assert response.status == "failed"
    assert response.dashboard is None
    assert response.errors[0].code == "MULTI_AGENT_PIPELINE_FAILED"
    assert "raw graph failure" not in response.model_dump_json()


def test_multi_agent_pipeline_converts_invalid_graph_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bi_service = service()
    mock_graph(monkeypatch, {"dashboard_output": {"status": "success"}})

    response = asyncio.run(bi_service._run_multi_agent_pipeline(dataset_record()))

    assert response.status == "failed"
    assert response.dashboard is None
    assert response.errors[0].code == "MULTI_AGENT_PIPELINE_FAILED"
