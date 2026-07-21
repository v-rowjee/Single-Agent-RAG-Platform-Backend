from __future__ import annotations

import sys
from io import BytesIO
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pandas as pd
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.agents.single import business_intelligence_agent as single_agent_module
from app.api import business_intelligence as business_intelligence_api
from app.core.auth import CurrentUser, get_current_user
from app.core.config import Settings
from app.main import app
from app.schemas.business_intelligence import DashboardResponse
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import (
    DashboardRecord,
    DatasetRecord,
    MessageRecord,
)


CSV_CONTENT = b"Region,Revenue\nNorth,10\nSouth,20\n"
USER_ID = "59b3d0fc-2d4a-40a0-8bb1-99e19da406ee"


class InMemoryStorage:
    """Network-free storage substitute for the complete API flow."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.datasets: dict[str, DatasetRecord] = {}
        self.dashboards: dict[str, DashboardRecord] = {}
        self.messages: list[MessageRecord] = []
        self.status_updates: list[tuple[str, dict[str, object]]] = []

    def upload_file(self, storage_path: str, content: bytes, mime_type: str) -> None:
        self.files[storage_path] = content

    def download_file(self, storage_path: str) -> bytes:
        return self.files[storage_path]

    def delete_file(self, storage_path: str) -> None:
        self.files.pop(storage_path, None)

    def create_dataset(
        self,
        dataset_id: str,
        user_id: str,
        file_name: str,
        storage_path: str,
        mime_type: str,
        file_size: int,
        file_hash: str,
        description: str | None,
        row_count: int,
        column_count: int,
    ) -> DatasetRecord:
        dataset = DatasetRecord(
            id=dataset_id,
            user_id=user_id,
            file_name=file_name,
            storage_path=storage_path,
            mime_type=mime_type,
            file_size=file_size,
            file_hash=file_hash,
            description=description,
            status="processing",
            rag_status="pending",
            error_message=None,
            row_count=row_count,
            column_count=column_count,
        )
        self.datasets[dataset_id] = dataset
        return dataset

    def get_dataset(self, dataset_id: str, user_id: str) -> DatasetRecord | None:
        dataset = self.datasets.get(dataset_id)
        return dataset if dataset is not None and dataset.user_id == user_id else None

    def get_active_dataset(self, user_id: str) -> DatasetRecord | None:
        return next(
            (
                dataset
                for dataset in self.datasets.values()
                if dataset.user_id == user_id
            ),
            None,
        )

    def delete_dataset(self, dataset_id: str, user_id: str) -> None:
        dataset = self.get_dataset(dataset_id, user_id)
        if dataset is None:
            return
        self.datasets.pop(dataset_id, None)
        self.dashboards.pop(dataset_id, None)
        self.messages = [
            message
            for message in self.messages
            if message.dataset_id != dataset_id
        ]

    def update_dataset_status(self, dataset_id: str, **updates: object) -> None:
        self.status_updates.append((dataset_id, updates))
        dataset = self.datasets[dataset_id]
        allowed = {
            key: value
            for key, value in updates.items()
            if key in {"status", "rag_status", "error_message"}
        }
        self.datasets[dataset_id] = replace(dataset, **allowed)

    def save_dashboard(
        self,
        dataset_id: str,
        status: str,
        response: dict[str, object],
    ) -> DashboardRecord:
        dashboard = DashboardRecord(
            id=f"dashboard-{dataset_id}",
            dataset_id=dataset_id,
            status=status,
            response=response,
        )
        self.dashboards[dataset_id] = dashboard
        return dashboard

    def get_dashboard(self, dataset_id: str) -> DashboardRecord | None:
        return self.dashboards.get(dataset_id)

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
            created_at=f"2026-07-17T10:00:0{len(self.messages)}Z",
        )
        self.messages.append(message)
        return message

    def get_recent_messages(
        self,
        dataset_id: str,
        limit: int = 6,
    ) -> list[MessageRecord]:
        messages = [
            message for message in self.messages if message.dataset_id == dataset_id
        ]
        return messages[-limit:]


class DummyRagService:
    """Records single-agent indexing without embedding or network access."""

    def __init__(self) -> None:
        self.index_calls: list[dict[str, object]] = []

    def index_dataset(self, **kwargs: object) -> SimpleNamespace:
        agent_input = kwargs["agent_input"]
        self.index_calls.append(
            {
                "session_id": agent_input.sessionId,
                "file_name": agent_input.fileName,
                "content": Path(agent_input.filePath).read_bytes(),
                "force": kwargs["force"],
            }
        )
        return SimpleNamespace(collection_name="document_chunks")


class DummySingleAgent:
    """Deterministic stand-in for all single-agent LLM work."""

    def __init__(self) -> None:
        self.dashboard_calls: list[dict[str, object]] = []
        self.profile_calls: list[str] = []
        self.chat_calls: list[dict[str, object]] = []
        self._source_ids: dict[str, list[str]] = {}

    def generate_dashboard(self, agent_input: Any) -> DashboardResponse:
        self.dashboard_calls.append(
            {
                "session_id": agent_input.sessionId,
                "file_name": agent_input.fileName,
                "content": Path(agent_input.filePath).read_bytes(),
            }
        )
        return _dashboard_response(agent_input.sessionId, agent_input.fileName)

    def profile_for_session(self, agent_input: Any) -> dict[str, object]:
        self.profile_calls.append(agent_input.sessionId)
        return {"summary": {"measures": ["Revenue"], "dimensions": ["Region"]}}

    def chat(
        self,
        agent_input: Any,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        self.chat_calls.append(
            {
                "session_id": agent_input.sessionId,
                "query": query,
                "history": history.copy(),
            }
        )
        if len(self.chat_calls) == 1:
            self._source_ids[agent_input.sessionId] = ["dataset_summary"]
            return (
                "**Answer:** Revenue totals 30 across North and South.\n\n"
                "**Grounding:** The uploaded sales dataset."
            )

        self._source_ids[agent_input.sessionId] = ["dataset_summary", "revenue_by_region"]
        return (
            "**Answer:** North contributes 10 and South contributes 20.\n\n"
            "**Grounding:** The prior revenue result and the regional rows."
        )

    def source_ids_for_session(self, session_id: str) -> list[str]:
        return self._source_ids[session_id]


def _dashboard_response(session_id: str, file_name: str) -> DashboardResponse:
    template_service = BusinessIntelligenceService(
        storage=SimpleNamespace(),  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="single"),
    )
    payload = template_service._build_placeholder_dashboard(
        dataset=DatasetRecord(
            id=session_id,
            user_id=USER_ID,
            file_name=file_name,
            storage_path=f"{session_id}/{file_name}",
            mime_type="text/csv",
            file_size=len(CSV_CONTENT),
            file_hash="test-hash",
            description="Quarterly sales",
            status="processing",
            rag_status="pending",
            error_message=None,
        ),
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
    payload["status"] = "success"
    payload["warnings"] = []
    return DashboardResponse.model_validate(payload)


@pytest.fixture
def full_flow(monkeypatch: pytest.MonkeyPatch):
    storage = InMemoryStorage()
    agent = DummySingleAgent()
    rag = DummyRagService()
    service = BusinessIntelligenceService(
        storage=storage,  # type: ignore[arg-type]
        settings=Settings("", "", bi_pipeline_mode="single"),
        rag=rag,
    )

    async def multi_agent_pipeline_must_not_run(*args: object, **kwargs: object):
        raise AssertionError("The multi-agent pipeline must not run in single mode.")

    monkeypatch.setattr(
        service,
        "_run_multi_agent_pipeline",
        multi_agent_pipeline_must_not_run,
    )
    monkeypatch.setattr(
        business_intelligence_api,
        "business_intelligence_service",
        service,
    )
    monkeypatch.setattr(single_agent_module, "business_intelligence_agent", agent)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(id=USER_ID)

    try:
        with TestClient(app) as client:
            yield client, storage, agent, rag
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_single_agent_api_full_flow_uses_only_deterministic_fakes(full_flow) -> None:
    client, storage, agent, rag = full_flow

    upload = client.post(
        "/api/upload",
        files={"file": ("sales.csv", CSV_CONTENT, "text/csv")},
        data={"description": "Quarterly sales"},
    )

    assert upload.status_code == 200
    upload_body = upload.json()
    assert upload_body["status"] == "success"
    assert upload_body["fileName"] == "sales.csv"
    session_id = upload_body["sessionId"]
    assert storage.files[f"{USER_ID}/{session_id}/sales.csv"] == CSV_CONTENT
    assert storage.datasets[session_id].description == "Quarterly sales"
    assert storage.datasets[session_id].status == "ready"
    assert storage.datasets[session_id].rag_status == "ready"
    assert len(agent.dashboard_calls) == 1
    assert agent.dashboard_calls[0]["content"] == CSV_CONTENT
    assert agent.profile_calls == [session_id]
    assert rag.index_calls == [
        {
            "session_id": session_id,
            "file_name": "sales.csv",
            "content": CSV_CONTENT,
            "force": True,
        }
    ]

    dashboard = client.get(f"/api/dashboard/{session_id}")

    assert dashboard.status_code == 200
    assert dashboard.json()["status"] == "success"
    assert dashboard.json()["sessionId"] == session_id
    assert dashboard.json()["dashboard"]["datasetSummary"] == {
        "fileName": "sales.csv",
        "rowCount": 2,
        "columnCount": 2,
        "timeField": None,
        "period": None,
        "measures": ["Revenue"],
        "dimensions": ["Region"],
        "quality": {
            "completenessPercent": 100.0,
            "missingValueCount": 0,
            "duplicateRowCount": 0,
        },
        "generatedAt": dashboard.json()["dashboard"]["datasetSummary"]["generatedAt"],
    }
    assert len(agent.dashboard_calls) == 1

    first_chat = client.post(
        "/api/chat",
        json={"sessionId": session_id, "query": "What is total revenue?"},
    )
    second_chat = client.post(
        "/api/chat",
        json={"sessionId": session_id, "query": "How is it split by region?"},
    )

    assert first_chat.status_code == 200
    assert first_chat.json() == {
        "answer": "Revenue totals 30 across North and South.",
        "grounding": "The uploaded sales dataset.",
        "agentMetadata": {
            "agent": "Chat assistant",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
        },
    }
    assert second_chat.status_code == 200
    assert second_chat.json() == {
        "answer": "North contributes 10 and South contributes 20.",
        "grounding": "The prior revenue result and the regional rows.",
        "agentMetadata": {
            "agent": "Chat assistant",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
        },
    }
    assert agent.chat_calls[0]["history"] == []
    assert agent.chat_calls[1]["history"] == [
        {"role": "user", "content": "What is total revenue?"},
        {
            "role": "assistant",
            "content": (
                "**Answer:** Revenue totals 30 across North and South.\n\n"
                "**Grounding:** The uploaded sales dataset."
            ),
        },
    ]
    assert storage.messages[1].sources == ["dataset_summary"]
    assert storage.messages[3].sources == ["dataset_summary", "revenue_by_region"]

    history = client.get(f"/api/chat/{session_id}/history")

    assert history.status_code == 200
    assert history.json()["sessionId"] == session_id
    assert [message["role"] for message in history.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["grounded"] for message in history.json()["messages"]] == [
        False,
        True,
        False,
        True,
    ]
    assert history.json()["messages"][1]["agentMetadata"] == {
        "agent": "Chat assistant",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
    }


def test_analysis_routes_reject_requests_without_a_bearer_token() -> None:
    with TestClient(app) as client:
        assert client.post(
            "/api/upload",
            files={"file": ("sales.csv", CSV_CONTENT, "text/csv")},
        ).status_code == 401
        assert client.get("/api/dashboard/9d719abc-9e09-4c14-b2d6-ed8308a1b85d").status_code == 401
        assert client.post(
            "/api/chat",
            json={
                "sessionId": "9d719abc-9e09-4c14-b2d6-ed8308a1b85d",
                "query": "What is revenue?",
            },
        ).status_code == 401
        assert client.get(
            "/api/chat/9d719abc-9e09-4c14-b2d6-ed8308a1b85d/history"
        ).status_code == 401
        assert client.get("/api/dataset").status_code == 401
        assert client.get("/api/dataset/preview").status_code == 401
        assert client.post("/api/dataset/reset").status_code == 401


def test_other_users_cannot_access_an_existing_session(full_flow) -> None:
    client, _, _, _ = full_flow
    upload = client.post(
        "/api/upload",
        files={"file": ("sales.csv", CSV_CONTENT, "text/csv")},
    )
    session_id = upload.json()["sessionId"]
    other_user = "6bd2f47e-f81a-4fa6-a8e2-8af53fd2a6f0"
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(id=other_user)

    assert client.get(f"/api/dashboard/{session_id}").status_code == 404
    assert client.post(
        "/api/chat",
        json={"sessionId": session_id, "query": "What is revenue?"},
    ).status_code == 404
    assert client.get(f"/api/chat/{session_id}/history").status_code == 404
    assert client.get("/api/dataset").status_code == 404
    assert client.get("/api/dataset/preview").status_code == 404
    assert client.post("/api/dataset/reset").status_code == 404


def test_active_dataset_details_preview_second_upload_and_reset(full_flow) -> None:
    client, storage, agent, _ = full_flow

    assert client.get("/api/dataset").status_code == 404

    upload = client.post(
        "/api/upload",
        files={"file": ("sales.csv", CSV_CONTENT, "text/csv")},
        data={"description": "Quarterly sales analysis"},
    )
    assert upload.status_code == 200
    session_id = upload.json()["sessionId"]

    details = client.get("/api/dataset")
    assert details.status_code == 200
    assert details.json() == {
        "sessionId": session_id,
        "fileName": "sales.csv",
        "fileSize": len(CSV_CONTENT),
        "uploadedAt": "",
        "rowCount": 2,
        "columnCount": 2,
        "analysisStatus": "ready",
        "ragStatus": "ready",
        "originalPrompt": "Quarterly sales analysis",
    }

    preview = client.get("/api/dataset/preview?page=1&page_size=50")
    assert preview.status_code == 200
    assert preview.json() == {
        "columns": ["Region", "Revenue"],
        "rows": [
            {"Region": "North", "Revenue": 10},
            {"Region": "South", "Revenue": 20},
        ],
        "page": 1,
        "page_size": 50,
        "total_rows": 2,
        "total_pages": 1,
    }

    duplicate = client.post(
        "/api/upload",
        files={"file": ("replacement.csv", CSV_CONTENT, "text/csv")},
    )
    assert duplicate.status_code == 409
    assert len(agent.dashboard_calls) == 1

    assert client.post("/api/chat", json={"sessionId": session_id, "query": "What is revenue?"}).status_code == 200
    reset = client.post("/api/dataset/reset")
    assert reset.status_code == 204
    assert storage.datasets == {}
    assert storage.files == {}
    assert storage.dashboards == {}
    assert storage.messages == []
    assert client.get("/api/dataset").status_code == 404


def test_xlsx_preview_returns_only_the_requested_page(full_flow) -> None:
    client, _, _, _ = full_flow
    workbook = BytesIO()
    pd.DataFrame(
        [
            {"Region": "North", "Revenue": 10},
            {"Region": "South", "Revenue": 20},
        ]
    ).to_excel(workbook, index=False)

    upload = client.post(
        "/api/upload",
        files={
            "file": (
                "sales.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert upload.status_code == 200

    preview = client.get("/api/dataset/preview?page=2&page_size=1")
    assert preview.status_code == 200
    assert preview.json() == {
        "columns": ["Region", "Revenue"],
        "rows": [{"Region": "South", "Revenue": 20}],
        "page": 2,
        "page_size": 1,
        "total_rows": 2,
        "total_pages": 2,
    }


def test_empty_dataset_preview_returns_an_empty_page(full_flow) -> None:
    client, _, _, _ = full_flow
    upload = client.post(
        "/api/upload",
        files={"file": ("empty.csv", b"Region,Revenue\n", "text/csv")},
    )
    assert upload.status_code == 200

    preview = client.get("/api/dataset/preview")
    assert preview.status_code == 200
    assert preview.json() == {
        "columns": ["Region", "Revenue"],
        "rows": [],
        "page": 1,
        "page_size": 50,
        "total_rows": 0,
        "total_pages": 0,
    }
