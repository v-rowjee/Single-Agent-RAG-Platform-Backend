from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import UUID, uuid4

import pandas as pd
from fastapi import UploadFile

from app.core.config import Settings, get_settings
from app.core.model_usage import (
    multi_dashboard_model_usage,
    single_dashboard_model_usage,
)
from app.schemas.business_intelligence import (
    ApiMessage,
    BusinessIntelligenceAgentInput,
    ChatResponse,
    DashboardResponse,
)
from app.orchestration.business_intelligence_graph import (
    business_intelligence_graph,
)
from app.rag.rag_service import rag_service
from app.services.chat_service import ChatService
from app.services.supabase_service import (
    AnalysisSessionRecord,
    DatasetRecord,
    MessageRecord,
    SupabaseService,
    supabase_service,
)


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_FILES = 5
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class SessionNotFoundError(Exception):
    """Raised when an analysis session does not exist."""


class InvalidUploadError(Exception):
    """Raised when an uploaded file cannot be processed."""


class DatasetAlreadyExistsError(Exception):
    """Raised when a user attempts to upload a second active dataset."""


@dataclass
class PipelineExecution:
    response: DashboardResponse
    workflow: dict[str, Any] | None = None
    retrieval_documents: list[dict[str, Any]] | None = None


class BusinessIntelligenceService:
    def __init__(
        self,
        storage: SupabaseService | None = None,
        settings: Settings | None = None,
        *,
        rag: Any | None = None,
        multi_chat_agent: Any | None = None,
        chat_service: ChatService | None = None,
    ) -> None:
        self.storage = storage or supabase_service
        self.settings = settings or get_settings()
        self._rag_service = rag or rag_service
        self._chat_service = chat_service or ChatService(
            rag=self._rag_service,
            agent=multi_chat_agent,
        )
        self._legacy_contract_sessions: set[str] = set()

    def uses_legacy_contract(self, session_id: str) -> bool:
        return session_id in self._legacy_contract_sessions

    async def create_analysis(
        self,
        files: list[UploadFile] | UploadFile,
        user_id: str,
        description: str | None = None,
        legacy_contract: bool = False,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("An authenticated user is required.")
        uploaded_files = (
            [files]
            if hasattr(files, "read") and not isinstance(files, (list, tuple))
            else list(files)
        )
        if not 1 <= len(uploaded_files) <= MAX_UPLOAD_FILES:
            raise InvalidUploadError(
                f"Upload between 1 and {MAX_UPLOAD_FILES} files at once."
            )

        get_active_session = getattr(self.storage, "get_active_session", None)
        active_session = (
            get_active_session(user_id)
            if callable(get_active_session)
            else self.storage.get_active_dataset(user_id)
        )
        if active_session is not None:
            raise DatasetAlreadyExistsError(
                "You already have an active workspace. Use Start Over before uploading another batch."
            )

        inspected: list[dict[str, Any]] = []
        names: set[str] = set()
        hashes: set[str] = set()
        for file in uploaded_files:
            original_name = Path(file.filename or "").name
            file_name = self._sanitize_file_name(original_name)
            mime_type = (file.content_type or "").strip()
            extension = Path(file_name).suffix.lower()
            self._validate_upload_metadata(file_name, mime_type, extension)
            normalized_name = file_name.casefold()
            if normalized_name in names:
                raise InvalidUploadError(
                    f"Duplicate filename '{file_name}' is not allowed."
                )

            content = await file.read()
            if not content:
                raise InvalidUploadError(f"'{file_name}' is empty.")
            if len(content) > MAX_UPLOAD_BYTES:
                raise InvalidUploadError(
                    f"'{file_name}' is larger than the 25 MiB limit."
                )

            file_hash = hashlib.sha256(content).hexdigest()
            if file_hash in hashes:
                raise InvalidUploadError(
                    f"'{file_name}' duplicates another file in this batch."
                )

            dataset_info = self._inspect_file(file_name, content)
            names.add(normalized_name)
            hashes.add(file_hash)
            inspected.append(
                {
                    "file_name": file_name,
                    "mime_type": mime_type,
                    "content": content,
                    "file_hash": file_hash,
                    "dataset_info": dataset_info,
                }
            )

        session_id = str(uuid4())
        session: AnalysisSessionRecord
        datasets: list[DatasetRecord] = []
        uploaded_paths: list[str] = []
        session_created = False
        try:
            create_session = getattr(self.storage, "create_session", None)
            if callable(create_session):
                session = create_session(
                    session_id=session_id,
                    user_id=user_id,
                    description=description,
                )
                session_created = True
            else:
                session = AnalysisSessionRecord(
                    id=session_id,
                    user_id=user_id,
                    description=description,
                    status="processing",
                    rag_status="pending",
                    error_message=None,
                )

            for index, item in enumerate(inspected):
                dataset_id = (
                    session_id
                    if not session_created and index == 0
                    else str(uuid4())
                )
                storage_path = (
                    f"{user_id}/{session_id}/{item['file_name']}"
                    if not session_created and len(inspected) == 1
                    else f"{user_id}/{session_id}/{dataset_id}/{item['file_name']}"
                )
                create_values = {
                    "dataset_id": dataset_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "file_name": item["file_name"],
                    "storage_path": storage_path,
                    "mime_type": item["mime_type"],
                    "file_size": len(item["content"]),
                    "file_hash": item["file_hash"],
                    "description": description,
                    "row_count": int(item["dataset_info"]["rowCount"]),
                    "column_count": int(item["dataset_info"]["columnCount"]),
                }
                try:
                    dataset = self.storage.create_dataset(**create_values)
                except TypeError:
                    create_values.pop("session_id")
                    dataset = self.storage.create_dataset(**create_values)
                    dataset = DatasetRecord(
                        **{
                            **dataset.__dict__,
                            "session_id": session_id,
                        }
                    )
                datasets.append(dataset)

                self.storage.upload_file(
                    storage_path=storage_path,
                    content=item["content"],
                    mime_type=item["mime_type"],
                )
                uploaded_paths.append(storage_path)

            await self._run_workspace_pipeline(
                session=session,
                datasets=datasets,
                contents=[item["content"] for item in inspected],
            )
            if legacy_contract:
                self._legacy_contract_sessions.add(session_id)

        except Exception as error:
            if not datasets and self._is_duplicate_dataset_error(error):
                raise DatasetAlreadyExistsError(
                    "You already have an active workspace. Use Start Over before uploading another batch."
                ) from error

            logger.exception(
                "Workspace processing failed session_id=%s operation=upload",
                session_id,
            )
            for storage_path in reversed(uploaded_paths):
                try:
                    self.storage.delete_file(storage_path)
                except Exception:
                    logger.exception(
                        "Unable to roll back uploaded object path=%s",
                        storage_path,
                    )
            if session_created and hasattr(self.storage, "delete_session"):
                self.storage.delete_session(session_id, user_id)
            else:
                for dataset in datasets:
                    self.storage.delete_dataset(dataset.id, user_id)
            raise

        result = {
            "status": "success",
            "sessionId": session_id,
            "datasetIds": [dataset.id for dataset in datasets],
            "message": "Files uploaded and analysis workspace created successfully.",
        }
        if len(datasets) == 1:
            result["fileName"] = datasets[0].file_name
        return result

    def get_active_dataset(self, user_id: str) -> DatasetRecord:
        _, datasets = self._active_workspace(user_id)
        return datasets[0]

    def _active_workspace(
        self,
        user_id: str,
    ) -> tuple[AnalysisSessionRecord, list[DatasetRecord]]:
        if not user_id:
            raise SessionNotFoundError("No active dataset was found.")
        get_active_session = getattr(self.storage, "get_active_session", None)
        if callable(get_active_session):
            session = get_active_session(user_id)
            if session is None:
                raise SessionNotFoundError("No active dataset was found.")
            datasets = self.storage.get_session_datasets(session.id, user_id)
            if not datasets:
                raise SessionNotFoundError("No active dataset was found.")
            return session, datasets

        dataset = self.storage.get_active_dataset(user_id)
        if dataset is None:
            raise SessionNotFoundError("No active dataset was found.")
        session = AnalysisSessionRecord(
            id=dataset.session_id or dataset.id,
            user_id=dataset.user_id,
            description=dataset.description,
            status=dataset.status,
            rag_status=dataset.rag_status,
            error_message=dataset.error_message,
        )
        return session, [dataset]

    def get_active_dataset_details(self, user_id: str) -> dict[str, Any]:
        session, datasets = self._active_workspace(user_id)
        metadata: list[dict[str, Any]] = []
        for dataset in datasets:
            row_count = dataset.row_count
            column_count = dataset.column_count
            if row_count is None or column_count is None:
                dataset_info = self._inspect_file(
                    dataset.file_name,
                    self.storage.download_file(dataset.storage_path),
                )
                row_count = int(dataset_info["rowCount"])
                column_count = int(dataset_info["columnCount"])
            metadata.append(
                {
                    "datasetId": dataset.id,
                    "fileName": dataset.file_name,
                    "fileSize": dataset.file_size,
                    "uploadedAt": dataset.created_at or "",
                    "rowCount": row_count,
                    "columnCount": column_count,
                }
            )

        if session.id in self._legacy_contract_sessions and len(metadata) == 1:
            item = metadata[0]
            return {
                "sessionId": session.id,
                "fileName": item["fileName"],
                "fileSize": item["fileSize"],
                "uploadedAt": item["uploadedAt"],
                "rowCount": item["rowCount"],
                "columnCount": item["columnCount"],
                "analysisStatus": session.status,
                "ragStatus": session.rag_status,
                "originalPrompt": session.description,
            }

        return {
            "sessionId": session.id,
            "analysisStatus": session.status,
            "ragStatus": session.rag_status,
            "originalPrompt": session.description,
            "requiresReset": session.requires_reset,
            "datasets": metadata,
        }

    def get_dataset_preview(
        self,
        user_id: str,
        dataset_id: str | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        session, datasets = self._active_workspace(user_id)
        dataset = (
            datasets[0]
            if dataset_id is None
            else next(
                (candidate for candidate in datasets if candidate.id == dataset_id),
                None,
            )
        )
        if dataset is None:
            raise SessionNotFoundError(
                f"Dataset '{dataset_id}' was not found in the active workspace."
            )
        if session.requires_reset:
            raise InvalidUploadError(
                "This legacy workspace must be reset before previews are available."
            )
        content = self.storage.download_file(dataset.storage_path)
        total_rows = (
            dataset.row_count
            if dataset.row_count is not None
            else int(self._inspect_file(dataset.file_name, content)["rowCount"])
        )
        columns, rows = self._read_preview_page(
            dataset.file_name,
            content,
            page,
            page_size,
        )
        total_pages = math.ceil(total_rows / page_size) if total_rows else 0
        return {
            "columns": columns,
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        }

    def reset_active_dataset(self, user_id: str) -> None:
        session, datasets = self._active_workspace(user_id)
        for dataset in datasets:
            self.storage.delete_file(dataset.storage_path)
        if hasattr(self.storage, "delete_session"):
            self.storage.delete_session(session.id, user_id)
        else:
            for dataset in datasets:
                self.storage.delete_dataset(dataset.id, user_id)
        self._legacy_contract_sessions.discard(session.id)

    async def get_dashboard(self, session_id: str, user_id: str) -> DashboardResponse:
        session, datasets = self._load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before its dashboard is available."
            )
        dashboard = self.storage.get_dashboard(session.id)

        if dashboard is not None:
            try:
                return self._with_dashboard_model_metadata(
                    DashboardResponse.model_validate(dashboard.response)
                )
            except Exception:
                logger.exception(
                    "Saved dashboard validation failed session_id=%s operation=get_dashboard",
                    session.id,
                )

        contents = [
            self.storage.download_file(dataset.storage_path)
            for dataset in datasets
        ]
        return await self._run_workspace_pipeline(session, datasets, contents)

    async def _run_workspace_pipeline(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> DashboardResponse:
        if len(datasets) != len(contents) or not datasets:
            raise ValueError("Every workspace dataset requires source content.")

        executions: list[PipelineExecution] = []
        for dataset, content in zip(datasets, contents, strict=True):
            if self.settings.bi_pipeline_mode == "multi":
                try:
                    execution = await self._run_multi_agent_pipeline(
                        dataset,
                        content,
                        workspace_session_id=session.id,
                    )
                except TypeError as error:
                    if "workspace_session_id" not in str(error):
                        raise
                    execution = await self._run_multi_agent_pipeline(
                        dataset,
                        content,
                    )
            else:
                execution = await self._run_single_agent_pipeline(dataset, content)
            if isinstance(execution, DashboardResponse):
                execution = PipelineExecution(response=execution)
            executions.append(execution)

        combined = self._combine_workspace_executions(session, datasets, executions)
        return self._persist_workspace_execution(
            session=session,
            datasets=datasets,
            contents=contents,
            execution=combined,
        )

    def _combine_workspace_executions(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        executions: list[PipelineExecution],
    ) -> PipelineExecution:
        if any(execution.response.status == "failed" for execution in executions):
            return PipelineExecution(
                response=self._with_dashboard_model_metadata(
                    DashboardResponse(
                        status="failed",
                        sessionId=session.id,
                        dashboard=None,
                        warnings=[
                            warning
                            for execution in executions
                            for warning in execution.response.warnings
                        ],
                        errors=[
                            *[
                                error
                                for execution in executions
                                for error in execution.response.errors
                            ],
                            ApiMessage(
                                code="DATASET_PREPARATION_FAILED",
                                message=(
                                    "At least one dataset could not be prepared, so "
                                    "the workspace analysis was stopped."
                                ),
                                component="data_preparation",
                                recoverable=True,
                            ),
                        ],
                    )
                ),
                workflow={
                    "session_id": session.id,
                    "dataset_workflows": [
                        execution.workflow or {}
                        for execution in executions
                    ],
                    "workflow_status": "failed",
                },
                retrieval_documents=[],
            )

        dashboards = [
            execution.response.dashboard
            for execution in executions
            if execution.response.dashboard is not None
        ]
        if not dashboards:
            raise ValueError("Workspace processing returned no dashboard data.")

        base = dashboards[0].model_dump(mode="json")
        summaries: list[dict[str, Any]] = []
        for dataset, dashboard in zip(datasets, dashboards, strict=True):
            summary = dashboard.datasetSummary.model_dump(mode="json")
            summary["datasetId"] = dataset.id
            summary["fileName"] = dataset.file_name
            summaries.append(summary)

        def namespace(
            dataset: DatasetRecord,
            item: dict[str, Any],
        ) -> dict[str, Any]:
            payload = dict(item)
            payload["id"] = f"{dataset.id}:{payload.get('id', 'item')}"
            if isinstance(payload.get("sourceIds"), list):
                payload["sourceIds"] = [
                    f"{dataset.id}:{source_id}"
                    for source_id in payload["sourceIds"]
                ]
            return payload

        kpis: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        chart_types: set[str] = set()
        findings: list[str] = []
        business_summaries: list[str] = []
        executive_summaries: list[str] = []
        insights = {
            "criticalAnomalies": [],
            "warnings": [],
            "limitations": [],
            "opportunities": [],
        }
        actions: list[dict[str, Any]] = []

        for dataset, dashboard in zip(datasets, dashboards, strict=True):
            executive_summaries.append(
                f"{dataset.file_name}: {dashboard.executiveSummary}"
            )
            business_summaries.append(
                f"{dataset.file_name}: {dashboard.analysis.businessSummary}"
            )
            findings.extend(
                f"{dataset.file_name}: {finding}"
                for finding in dashboard.analysis.keyFindings
            )
            kpis.extend(
                namespace(dataset, kpi.model_dump(mode="json"))
                for kpi in dashboard.kpis
            )
            for chart in dashboard.supportingCharts:
                if chart.type in chart_types or len(charts) >= 4:
                    continue
                chart_types.add(chart.type)
                charts.append(namespace(dataset, chart.model_dump(mode="json")))
            dashboard_insights = dashboard.insights.model_dump(mode="json")
            for category in insights:
                insights[category].extend(
                    namespace(dataset, item)
                    for item in dashboard_insights.get(category, [])
                )
            actions.extend(
                namespace(dataset, action.model_dump(mode="json"))
                for action in dashboard.recommendedActions
            )

        base.update(
            {
                "title": (
                    dashboards[0].title
                    if len(datasets) == 1
                    else "Multi-Dataset Business Intelligence Dashboard"
                ),
                "executiveSummary": "\n\n".join(executive_summaries),
                "kpis": kpis[:8],
                "supportingCharts": charts,
                "analysis": {
                    "businessSummary": "\n\n".join(business_summaries),
                    "keyFindings": findings,
                },
                "insights": insights,
                "recommendedActions": actions[:5],
                "datasetSummaries": summaries,
            }
        )

        status = (
            "partial"
            if any(execution.response.status == "partial" for execution in executions)
            else "success"
        )
        response = DashboardResponse.model_validate(
            {
                "status": status,
                "sessionId": session.id,
                "dashboard": base,
                "warnings": [
                    warning.model_dump(mode="json")
                    for execution in executions
                    for warning in execution.response.warnings
                ],
                "errors": [
                    error.model_dump(mode="json")
                    for execution in executions
                    for error in execution.response.errors
                ],
                "pipelineMode": self.settings.bi_pipeline_mode,
                "agentModels": self._combined_agent_models(executions),
            }
        )

        retrieval_documents: list[dict[str, Any]] = []
        for dataset, execution in zip(datasets, executions, strict=True):
            for index, document in enumerate(execution.retrieval_documents or []):
                payload = dict(document)
                source_id = str(payload.get("id") or f"document_{index}")
                payload["id"] = f"{dataset.id}:{source_id}"
                metadata = dict(payload.get("metadata") or {})
                metadata.update(
                    {
                        "session_id": session.id,
                        "dataset_id": dataset.id,
                        "file_name": dataset.file_name,
                    }
                )
                payload["metadata"] = metadata
                retrieval_documents.append(payload)

        return PipelineExecution(
            response=response,
            workflow={
                "session_id": session.id,
                "dataset_ids": [dataset.id for dataset in datasets],
                "dataset_workflows": [
                    execution.workflow or {}
                    for execution in executions
                ],
                "workflow_status": status,
            },
            retrieval_documents=retrieval_documents,
        )

    @staticmethod
    def _combined_agent_models(
        executions: list[PipelineExecution],
    ) -> list[dict[str, str]]:
        unique: dict[tuple[str, str, str], dict[str, str]] = {}
        for execution in executions:
            for usage in execution.response.agentModels:
                payload = usage.model_dump(mode="json")
                key = (
                    str(payload.get("agent") or ""),
                    str(payload.get("model") or ""),
                    str(payload.get("provider") or ""),
                )
                unique[key] = payload
        return list(unique.values())

    async def _run_selected_pipeline(
        self,
        session: DatasetRecord,
        content: bytes | None = None,
    ) -> DashboardResponse:
        logger.info(
            "Selected BI pipeline mode=%s session_id=%s",
            self.settings.bi_pipeline_mode,
            session.id,
        )
        source_content = (
            content
            if content is not None
            else self.storage.download_file(session.storage_path)
        )
        if self.settings.bi_pipeline_mode == "multi":
            execution = await self._run_multi_agent_pipeline(session, source_content)
        else:
            execution = await self._run_single_agent_pipeline(session, source_content)

        # Keep test doubles and extensions returning the pre-refactor type working.
        if isinstance(execution, DashboardResponse):
            execution = PipelineExecution(response=execution)
        return self._persist_pipeline_execution(
            session=session,
            content=source_content,
            execution=execution,
        )

    async def _run_single_agent_pipeline(
        self,
        session: DatasetRecord,
        content: bytes | None = None,
    ) -> PipelineExecution:
        return PipelineExecution(
            response=self._with_dashboard_model_metadata(
                self._generate_dashboard_from_bytes(
                    dataset=session,
                    content=(
                        content
                        if content is not None
                        else self.storage.download_file(session.storage_path)
                    ),
                )
            )
        )

    async def _run_multi_agent_pipeline(
        self,
        session: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
    ) -> PipelineExecution:
        """Run the compiled workflow and return its dashboard and durable artifacts."""
        session_id = workspace_session_id or session.session_id or session.id
        content = (
            content
            if content is not None
            else self.storage.download_file(session.storage_path)
        )

        with self._temporary_agent_workspace(
            session,
            content,
        ) as (agent_input, workspace):
            initial_state = {
                "session_id": session_id,
                "dataset_id": session.id,
                "file_name": session.file_name,
                "business_description": session.description,
                "uploaded_file_path": agent_input.filePath,
                "working_directory": str(workspace),
                "warnings": [],
                "errors": [],
                "completed_agents": [],
                "failed_agents": [],
                "skipped_agents": [],
            }
            logger.info("Multi-agent pipeline started session_id=%s", session_id)

            try:
                result = await business_intelligence_graph.ainvoke(initial_state)
                dashboard_output = result.get("dashboard_output")
                if not isinstance(dashboard_output, dict):
                    raise ValueError("The workflow did not return a dashboard output.")

                dashboard_output = dict(dashboard_output)
                dashboard_output["sessionId"] = session_id
                workflow_status = str(result.get("workflow_status") or "failed")
                if workflow_status == "failed":
                    raise RuntimeError("The multi-agent workflow failed.")
                dashboard_output["status"] = workflow_status
                response = self._with_dashboard_model_metadata(
                    DashboardResponse.model_validate(dashboard_output),
                    selected_agents=(
                        result.get("orchestration_plan") or {}
                    ).get("selected_agents", []),
                )
                workflow = {
                    key: result.get(key)
                    for key in (
                        "session_id",
                        "dataset_id",
                        "file_name",
                        "prepared_dataset",
                        "orchestration_plan",
                        "generic_cleaning_report",
                        "kpi_trend_output",
                        "anomaly_output",
                        "forecasting_output",
                        "synthesis_output",
                        "retrieval_documents",
                        "warnings",
                        "errors",
                        "completed_agents",
                        "failed_agents",
                        "skipped_agents",
                        "workflow_status",
                    )
                }
            except Exception:
                logger.exception(
                    "Unexpected multi-agent pipeline failure session_id=%s",
                    session_id,
                )
                failed_response = DashboardResponse(
                    status="failed",
                    sessionId=session_id,
                    dashboard=None,
                    warnings=[],
                    errors=[
                        ApiMessage(
                            code="MULTI_AGENT_PIPELINE_FAILED",
                            message=(
                                "The business intelligence workflow could not be "
                                "completed."
                            ),
                            component="business_intelligence",
                            recoverable=True,
                        )
                    ],
                )
                logger.info(
                    "Multi-agent final workflow status session_id=%s status=%s",
                    session_id,
                    failed_response.status,
                )
                return PipelineExecution(
                    response=self._with_dashboard_model_metadata(failed_response),
                    workflow={
                        "session_id": session_id,
                        "dataset_id": session.id,
                        "workflow_status": "failed",
                        "warnings": [],
                        "errors": ["The multi-agent workflow failed."],
                        "completed_agents": [],
                        "failed_agents": ["business_intelligence"],
                        "skipped_agents": [],
                    },
                    retrieval_documents=[],
                )

        logger.info("Multi-agent pipeline completed session_id=%s", session_id)
        logger.info(
            "Multi-agent final workflow status session_id=%s status=%s",
            session_id,
            response.status,
        )
        return PipelineExecution(
            response=response,
            workflow=workflow,
            retrieval_documents=list(result.get("retrieval_documents") or []),
        )

    def _with_dashboard_model_metadata(
        self,
        response: DashboardResponse,
        selected_agents: list[str] | tuple[str, ...] = (),
    ) -> DashboardResponse:
        """Add model provenance to old and newly generated dashboard records."""
        if response.agentModels:
            return response

        payload = response.model_dump(mode="json")
        if self.settings.bi_pipeline_mode == "single":
            usage = single_dashboard_model_usage()
            payload.update(
                {
                    "pipelineMode": "single",
                    "model": usage[0]["model"],
                    "agentModels": usage,
                }
            )
        else:
            payload.update(
                {
                    "pipelineMode": "multi",
                    "agentModels": multi_dashboard_model_usage(selected_agents),
                }
            )
        return DashboardResponse.model_validate(payload)

    def chat(self, session_id: str, query: str, user_id: str) -> ChatResponse:
        session, datasets = self._load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat is available."
            )
        if (
            self.settings.bi_pipeline_mode == "multi"
            or len(datasets) > 1
            or datasets[0].session_id is not None
        ):
            return self._chat_with_multi_agent_pipeline(
                session_id=session_id,
                query=query,
                user_id=user_id,
            )

        dataset = datasets[0]
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")

        history = self._chat_history(dataset.id)

        self.storage.save_message(
            dataset_id=dataset.id,
            role="user",
            content=cleaned_query,
            sources=[],
        )

        return self._chat_with_single_agent(
            dataset=dataset,
            query=cleaned_query,
            history=history,
        )

    def _chat_with_multi_agent_pipeline(
        self,
        session_id: str,
        query: str,
        user_id: str,
    ) -> ChatResponse:
        session, datasets = self._load_workspace(session_id, user_id)
        history = self._chat_history(session.id)
        selected, ambiguous = self._select_chat_datasets(query, datasets)
        if ambiguous:
            file_names = ", ".join(f"`{dataset.file_name}`" for dataset in ambiguous)
            cleaned_query = query.strip()
            self.storage.save_message(
                dataset_id=session.id,
                role="user",
                content=cleaned_query,
                sources=[],
            )
            return self._save_chat_response(
                session.id,
                (
                    "**Answer:** This calculation could apply to more than one "
                    f"dataset ({file_names}). Name the dataset or field you want "
                    "me to calculate so I do not combine incompatible tables.\n\n"
                    "**Grounding:** The active workspace contains multiple "
                    "matching dataset schemas."
                ),
                [],
            )

        retrieval_query = (
            f"In dataset `{selected.file_name}`, {query}"
            if selected is not None
            else query
        )
        result = self._chat_service.answer(
            session.id,
            retrieval_query,
            history=history,
        )

        self.storage.save_message(
            dataset_id=session.id,
            role="user",
            content=query.strip(),
            sources=[],
        )
        return self._save_chat_response(
            session.id,
            result.draft.answer,
            result.draft.source_ids,
        )

    def _select_chat_datasets(
        self,
        query: str,
        datasets: list[DatasetRecord],
    ) -> tuple[DatasetRecord | None, list[DatasetRecord]]:
        if len(datasets) <= 1:
            return None, []
        normalized_query = query.casefold()
        explicitly_named = [
            dataset
            for dataset in datasets
            if dataset.file_name.casefold() in normalized_query
            or Path(dataset.file_name).stem.casefold() in normalized_query
        ]
        if len(explicitly_named) == 1:
            return explicitly_named[0], []

        calculation_terms = (
            "total",
            "sum",
            "average",
            "avg",
            "minimum",
            "maximum",
            "highest",
            "lowest",
            "how many",
            "count",
        )
        if not any(term in normalized_query for term in calculation_terms):
            return None, []

        matching: list[DatasetRecord] = []
        for dataset in datasets:
            info = self._inspect_file(
                dataset.file_name,
                self.storage.download_file(dataset.storage_path),
            )
            fields = [
                str(field).casefold()
                for field in [*info["measures"], *info["dimensions"]]
            ]
            if any(field in normalized_query for field in fields):
                matching.append(dataset)

        if len(matching) == 1:
            return matching[0], []
        return None, matching or datasets

    def _chat_with_single_agent(
        self,
        dataset: DatasetRecord,
        query: str,
        history: list[dict[str, str]],
    ) -> ChatResponse:
        """Answer through the single-agent RAG graph for single pipeline sessions."""
        try:
            content = self.storage.download_file(dataset.storage_path)
            response, source_ids = self._chat_with_agent(
                dataset=dataset,
                content=content,
                query=query,
                history=history,
            )
        except Exception:
            logger.exception(
                "Single-agent chat preparation failed session_id=%s",
                dataset.id,
            )
            response = (
                "The analysis assistant could not answer this question at the moment."
            )
            source_ids = []

        return self._save_chat_response(dataset.id, response, source_ids)

    def _chat_history(self, dataset_id: str) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.storage.get_recent_messages(dataset_id, limit=10)
        ]

    def _save_chat_response(
        self,
        dataset_id: str,
        response_text: str,
        source_ids: list[str],
    ) -> ChatResponse:
        answer, grounding = self._split_chat_response(response_text, source_ids)
        self.storage.save_message(
            dataset_id=dataset_id,
            role="assistant",
            content=f"**Answer:** {answer}\n\n**Grounding:** {grounding}",
            sources=source_ids,
        )
        return ChatResponse(answer=answer, grounding=grounding)

    @staticmethod
    def _split_chat_response(
        response_text: str,
        source_ids: list[str],
    ) -> tuple[str, str]:
        """Split the agent's Markdown sections into the public chat contract."""
        grounding_match = re.search(
            r"(?:^|\n)\s*\*\*Grounding:\*\*\s*([\s\S]*)$",
            response_text,
            flags=re.IGNORECASE,
        )
        if grounding_match is not None:
            answer = response_text[: grounding_match.start()].strip()
            grounding = grounding_match.group(1).strip()
        else:
            answer = response_text.strip()
            source_text = ", ".join(f"`{source_id}`" for source_id in source_ids)
            grounding = (
                f"Retrieved dataset sources: {source_text}."
                if source_text
                else "No supporting dataset evidence was available."
            )

        answer = re.sub(
            r"^\s*\*\*Answer:\*\*\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return (
            answer or "The analysis assistant could not answer this question.",
            grounding or "No supporting dataset evidence was available.",
        )

    def get_chat_history(self, session_id: str, user_id: str) -> dict[str, Any]:
        session, _ = self._load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat history is available."
            )
        return {
            "sessionId": session.id,
            "messages": [
                self._chat_message(message)
                for message in self.storage.get_recent_messages(session.id, limit=50)
            ],
        }

    def _load_workspace(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[AnalysisSessionRecord, list[DatasetRecord]]:
        normalized_id = self._validate_session_id(session_id)
        if not user_id:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            )
        get_session = getattr(self.storage, "get_session", None)
        if callable(get_session):
            session = get_session(normalized_id, user_id)
            if session is None:
                raise SessionNotFoundError(
                    f"Analysis session '{session_id}' was not found."
                )
            datasets = self.storage.get_session_datasets(session.id, user_id)
            if not datasets:
                raise SessionNotFoundError(
                    f"Analysis session '{session_id}' was not found."
                )
            return session, datasets

        dataset = self.storage.get_dataset(normalized_id, user_id)
        if dataset is None:
            get_active_dataset = getattr(
                self.storage,
                "get_active_dataset",
                None,
            )
            active_dataset = (
                get_active_dataset(user_id)
                if callable(get_active_dataset)
                else None
            )
            dataset = next(
                (
                    candidate
                    for candidate in [active_dataset]
                    if candidate is not None
                    and (candidate.session_id or candidate.id) == normalized_id
                ),
                None,
            )
        if dataset is None:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            )
        return (
            AnalysisSessionRecord(
                id=dataset.session_id or dataset.id,
                user_id=dataset.user_id,
                description=dataset.description,
                status=dataset.status,
                rag_status=dataset.rag_status,
                error_message=dataset.error_message,
            ),
            [dataset],
        )

    def _load_dataset(self, session_id: str, user_id: str) -> DatasetRecord:
        _, datasets = self._load_workspace(session_id, user_id)
        return datasets[0]

    def _generate_dashboard_from_bytes(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> DashboardResponse:
        dataset_info = self._inspect_file(dataset.file_name, content)
        with self._temporary_agent_input(dataset, content) as agent_input:
            try:
                from app.agents.single.business_intelligence_agent import (
                    business_intelligence_agent,
                )

                return business_intelligence_agent.generate_dashboard(agent_input)
            except Exception:
                logger.exception(
                    "Business intelligence agent failed session_id=%s operation=dashboard. Returning fallback dashboard.",
                    dataset.id,
                )
                return DashboardResponse.model_validate(
                    self._build_placeholder_dashboard(
                        dataset=dataset,
                        dataset_info=dataset_info,
                    )
                )

    def _save_dashboard(
        self,
        dataset_id: str,
        dashboard_response: DashboardResponse,
    ) -> None:
        self.storage.save_dashboard(
            dataset_id=dataset_id,
            status=dashboard_response.status,
            response=dashboard_response.model_dump(mode="json"),
        )

    def _persist_workspace_execution(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
        execution: PipelineExecution,
    ) -> DashboardResponse:
        response = execution.response
        update_session = getattr(self.storage, "update_session_status", None)
        if callable(update_session):
            update_session(session.id, rag_status="indexing")
        else:
            for dataset in datasets:
                self.storage.update_dataset_status(
                    dataset.id,
                    rag_status="indexing",
                )

        if not hasattr(self._rag_service, "index_documents"):
            rag_status = (
                "ready"
                if response.status != "failed"
                and self._try_index_rag(datasets[0], contents[0])
                else "failed"
            )
            self._save_dashboard(session.id, response)
            for dataset in datasets:
                self.storage.update_dataset_status(
                    dataset.id,
                    status="failed" if response.status == "failed" else "ready",
                    rag_status=rag_status,
                    error_message=None,
                )
            return response

        retrieval_documents = list(execution.retrieval_documents or [])
        if self.settings.bi_pipeline_mode == "single" and response.status != "failed":
            retrieval_documents = self._single_workspace_documents(
                session,
                datasets,
                contents,
            )

        if response.status == "failed":
            indexing_result = {
                "status": "failed",
                "document_count": len(retrieval_documents),
                "indexed_count": 0,
                "failed_count": len(retrieval_documents),
                "message": "Retrieval indexing was skipped because analysis failed.",
            }
        else:
            indexing_result = self._rag_service.index_documents(
                session_id=session.id,
                dataset_id=session.id,
                retrieval_documents=retrieval_documents,
            )

        rag_status = (
            "ready" if indexing_result.get("status") == "success" else "failed"
        )
        if rag_status == "failed" and response.status != "failed":
            response = response.model_copy(
                update={
                    "status": "partial",
                    "warnings": [
                        *response.warnings,
                        ApiMessage(
                            code="RAG_INDEXING_FAILED",
                            message=(
                                "The dashboard was generated, but workspace "
                                "retrieval indexing could not be completed."
                            ),
                            component="rag",
                            recoverable=True,
                        ),
                    ],
                }
            )

        workflow = dict(execution.workflow or {})
        workflow.update(
            {
                "session_id": session.id,
                "dataset_ids": [dataset.id for dataset in datasets],
                "workflow_status": response.status,
                "retrieval_indexing_result": indexing_result,
            }
        )
        self._save_workflow_dashboard(session.id, response, workflow)

        if callable(update_session):
            update_session(
                session.id,
                status="failed" if response.status == "failed" else "ready",
                rag_status=rag_status,
                error_message=(
                    "Business intelligence pipeline failed."
                    if response.status == "failed"
                    else "Retrieval indexing failed."
                    if rag_status == "failed"
                    else None
                ),
            )
        else:
            for dataset in datasets:
                self.storage.update_dataset_status(
                    dataset.id,
                    status="failed" if response.status == "failed" else "ready",
                    rag_status=rag_status,
                    error_message=(
                        "Business intelligence pipeline failed."
                        if response.status == "failed"
                        else None
                    ),
                )
        return response

    def _single_workspace_documents(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> list[dict[str, Any]]:
        try:
            from app.agents.single.business_intelligence_agent import (
                business_intelligence_agent,
            )

            documents: list[dict[str, Any]] = []
            for dataset, content in zip(datasets, contents, strict=True):
                with self._temporary_agent_input(dataset, content) as agent_input:
                    profile = business_intelligence_agent.profile_for_session(
                        agent_input
                    )
                    build_documents = getattr(
                        self._rag_service,
                        "build_dataset_documents",
                    )
                    documents.extend(
                        build_documents(
                            agent_input=agent_input,
                            profile=profile,
                            workspace_session_id=session.id,
                        )
                    )
            return documents
        except Exception:
            logger.exception(
                "Single-agent workspace retrieval preparation failed session_id=%s",
                session.id,
            )
            return []

    def _persist_pipeline_execution(
        self,
        session: DatasetRecord,
        content: bytes,
        execution: PipelineExecution,
    ) -> DashboardResponse:
        """Persist dashboard, RAG state, and dataset status exactly once."""
        response = execution.response
        rag_status = "failed"
        self.storage.update_dataset_status(session.id, rag_status="indexing")

        if self.settings.bi_pipeline_mode == "multi":
            workflow = dict(execution.workflow or {})
            retrieval_failed = (
                response.status == "failed"
                or "retrieval_preparation"
                in set(workflow.get("failed_agents") or [])
            )
            if retrieval_failed:
                indexing_result = {
                    "status": "failed",
                    "document_count": len(execution.retrieval_documents or []),
                    "indexed_count": 0,
                    "failed_count": len(execution.retrieval_documents or []),
                    "message": "Retrieval indexing was skipped because the workflow failed.",
                }
            else:
                indexing_result = self._rag_service.index_documents(
                    session_id=session.id,
                    dataset_id=session.id,
                    retrieval_documents=list(execution.retrieval_documents or []),
                )

            rag_status = (
                "ready"
                if indexing_result.get("status") == "success"
                else "failed"
            )
            if rag_status == "failed" and response.status != "failed":
                response = response.model_copy(
                    update={
                        "status": "partial",
                        "warnings": [
                            *response.warnings,
                            ApiMessage(
                                code="RAG_INDEXING_FAILED",
                                message=(
                                    "The dashboard was generated, but retrieval "
                                    "indexing could not be completed."
                                ),
                                component="rag",
                                recoverable=True,
                            ),
                        ],
                    }
                )

            workflow.update(
                {
                    "session_id": session.id,
                    "dataset_id": session.id,
                    "workflow_status": response.status,
                    "retrieval_indexing_result": indexing_result,
                }
            )
            self._save_workflow_dashboard(session.id, response, workflow)
        else:
            rag_status = "ready" if self._try_index_rag(session, content) else "failed"
            self._save_dashboard(session.id, response)

        self.storage.update_dataset_status(
            session.id,
            status="failed" if response.status == "failed" else "ready",
            rag_status=rag_status,
            error_message=(
                "Business intelligence pipeline failed."
                if response.status == "failed"
                else "Retrieval indexing failed."
                if rag_status == "failed"
                else None
            ),
        )
        return response

    def _save_workflow_dashboard(
        self,
        dataset_id: str,
        response: DashboardResponse,
        workflow: dict[str, Any],
    ) -> None:
        generic_cleaning_report = self._persistent_cleaning_report(
            workflow.get("generic_cleaning_report")
        )
        prepared_dataset = self._persistent_prepared_dataset(
            workflow.get("prepared_dataset")
        )
        persistent_workflow = dict(workflow)
        persistent_workflow["generic_cleaning_report"] = generic_cleaning_report
        persistent_workflow["prepared_dataset"] = prepared_dataset
        persistent_workflow.pop("retrieval_documents", None)

        self.storage.save_session_processing(
            dataset_id=dataset_id,
            workflow_status=response.status,
            generic_cleaning_report=generic_cleaning_report,
            prepared_dataset=prepared_dataset,
        )

        stored_response = response.model_dump(mode="json")
        stored_response["workflow"] = {
            key: value
            for key, value in persistent_workflow.items()
            if key != "dashboard_output"
        }
        stored_response["workflow"]["dashboard_output"] = stored_response.copy()
        stored_response["workflow"]["dashboard_output"].pop("workflow", None)
        self.storage.save_dashboard(
            dataset_id=dataset_id,
            status=response.status,
            response=stored_response,
        )

    @staticmethod
    def _persistent_cleaning_report(value: Any) -> dict[str, Any]:
        report = dict(value) if isinstance(value, dict) else {}
        report.pop("cleaned_file_path", None)
        return report

    @classmethod
    def _persistent_prepared_dataset(cls, value: Any) -> dict[str, Any]:
        prepared = dict(value) if isinstance(value, dict) else {}
        prepared.pop("prepared_file_path", None)
        prepared.pop("temporal_dataset_path", None)
        cleaning_report = prepared.get("cleaning_report")
        if isinstance(cleaning_report, dict):
            prepared["cleaning_report"] = cls._persistent_cleaning_report(
                cleaning_report
            )
        return prepared

    def _try_index_rag(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> bool:
        try:
            from app.agents.single.business_intelligence_agent import (
                business_intelligence_agent,
            )

            with self._temporary_agent_input(dataset, content) as agent_input:
                profile = business_intelligence_agent.profile_for_session(agent_input)
                self._rag_service.index_dataset(
                    agent_input=agent_input,
                    profile=profile,
                    force=True,
                )
            return True
        except Exception:
            logger.warning(
                "Recoverable RAG indexing failure session_id=%s operation=index_rag",
                dataset.id,
                exc_info=True,
            )
            return False

    def _chat_with_agent(
        self,
        dataset: DatasetRecord,
        content: bytes,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        try:
            from app.agents.single.business_intelligence_agent import (
                business_intelligence_agent,
            )

            with self._temporary_agent_input(dataset, content) as agent_input:
                response = business_intelligence_agent.chat(
                    agent_input=agent_input,
                    query=query,
                    history=history,
                )
                source_ids = business_intelligence_agent.source_ids_for_session(
                    dataset.id
                )
            return response, source_ids

        except Exception:
            logger.exception(
                "Business intelligence agent failed session_id=%s operation=chat",
                dataset.id,
            )
            return (
                "**Answer:** I cannot answer from the dataset profile because "
                "the AI business intelligence agent is currently unavailable.\n\n"
                f"**Grounding:** Dataset `{dataset.file_name}`; user asked "
                f"`{query}`.",
                [],
            )

    @contextmanager
    def _temporary_agent_input(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> Iterator[BusinessIntelligenceAgentInput]:
        """Provide a temporary source file to legacy single-agent callers."""
        with self._temporary_agent_workspace(dataset, content) as (
            agent_input,
            _,
        ):
            yield agent_input

    @contextmanager
    def _temporary_agent_workspace(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> Iterator[tuple[BusinessIntelligenceAgentInput, Path]]:
        suffix = Path(dataset.file_name).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="bi_dataset_") as directory:
            workspace_root = Path(directory)
            path = workspace_root / dataset.file_name
            if path.suffix.lower() != suffix:
                path = workspace_root / f"dataset{suffix}"
            path.write_bytes(content)
            workspace = workspace_root / "processing"
            workspace.mkdir()
            yield (
                BusinessIntelligenceAgentInput(
                    sessionId=dataset.id,
                    datasetId=dataset.id,
                    filePath=str(path),
                    fileName=dataset.file_name,
                    description=dataset.description,
                ),
                workspace,
            )

    def _inspect_file(
        self,
        file_name: str,
        content: bytes,
    ) -> dict[str, Any]:
        suffix = Path(file_name).suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(io.BytesIO(content), low_memory=False)
            elif suffix == ".xlsx":
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise InvalidUploadError("Only CSV and XLSX files are supported.")
        except UnicodeDecodeError as error:
            raise InvalidUploadError(
                "The CSV file must use UTF-8 encoding."
            ) from error
        except Exception as error:
            raise InvalidUploadError(
                "The uploaded file could not be parsed."
            ) from error

        row_count = int(len(df))
        column_count = int(len(df.columns))
        missing_value_count = int(df.isna().sum().sum())
        duplicate_row_count = int(df.duplicated().sum())
        total_cells = row_count * column_count
        completeness_percent = (
            round(((total_cells - missing_value_count) / total_cells) * 100, 2)
            if total_cells
            else 100.0
        )
        measures = [
            str(column)
            for column in df.select_dtypes(include="number").columns
        ]
        dimensions = [
            str(column)
            for column in df.columns
            if str(column) not in measures
        ]

        return {
            "rowCount": row_count,
            "columnCount": column_count,
            "measures": measures,
            "dimensions": dimensions,
            "missingValueCount": missing_value_count,
            "duplicateRowCount": duplicate_row_count,
            "completenessPercent": completeness_percent,
        }

    def _read_preview_page(
        self,
        file_name: str,
        content: bytes,
        page: int,
        page_size: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        start_row = (page - 1) * page_size
        suffix = Path(file_name).suffix.lower()

        try:
            if suffix == ".csv":
                skiprows = (
                    (lambda row_index: row_index != 0 and row_index <= start_row)
                    if start_row
                    else None
                )
                dataframe = pd.read_csv(
                    io.BytesIO(content),
                    low_memory=False,
                    skiprows=skiprows,
                    nrows=page_size,
                )
            elif suffix == ".xlsx":
                dataframe = pd.read_excel(
                    io.BytesIO(content),
                    skiprows=range(1, start_row + 1) if start_row else None,
                    nrows=page_size,
                )
            else:
                raise InvalidUploadError("Only CSV and XLSX files are supported.")
        except UnicodeDecodeError as error:
            raise InvalidUploadError(
                "The CSV file must use UTF-8 encoding."
            ) from error
        except InvalidUploadError:
            raise
        except Exception as error:
            raise InvalidUploadError(
                "The uploaded dataset preview could not be read."
            ) from error

        columns = self._unique_column_names(
            [str(column) for column in dataframe.columns]
        )
        dataframe.columns = columns
        rows = [
            {
                column: self._json_preview_value(value)
                for column, value in row.items()
            }
            for row in dataframe.to_dict(orient="records")
        ]
        return columns, rows

    @staticmethod
    def _unique_column_names(columns: list[str]) -> list[str]:
        used: dict[str, int] = {}
        result: list[str] = []
        for index, raw_name in enumerate(columns, start=1):
            base_name = raw_name.strip() or f"Column {index}"
            occurrence = used.get(base_name, 0)
            used[base_name] = occurrence + 1
            result.append(
                base_name if occurrence == 0 else f"{base_name} ({occurrence + 1})"
            )
        return result

    @staticmethod
    def _json_preview_value(value: Any) -> str | int | float | bool | None:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)):
            return value
        return str(value)

    def _build_placeholder_dashboard(
        self,
        dataset: DatasetRecord,
        dataset_info: dict[str, Any],
    ) -> dict[str, Any]:
        generated_at = self._current_timestamp()
        row_count = int(dataset_info["rowCount"])
        column_count = int(dataset_info["columnCount"])
        completeness = float(dataset_info["completenessPercent"])
        file_size_kb = round(dataset.file_size / 1024, 1)

        return {
            "status": "partial",
            "sessionId": dataset.id,
            "dashboard": {
                "title": f"Business Intelligence Dashboard - {dataset.file_name}",
                "executiveSummary": (
                    f"The uploaded dataset contains {row_count:,} rows and "
                    f"{column_count:,} columns. The AI analysis pipeline was "
                    "unavailable, so this dashboard shows basic dataset metrics."
                ),
                "kpis": [
                    self._kpi("dataset_rows", "Dataset Rows", row_count),
                    self._kpi("dataset_columns", "Dataset Columns", column_count),
                    self._kpi("data_completeness", "Data Completeness", completeness, "%"),
                    self._kpi("file_size", "File Size", file_size_kb, " KB"),
                ],
                "timeline": None,
                "supportingCharts": [
                    {
                        "id": "dataset_structure",
                        "type": "bar",
                        "title": "Dataset Structure",
                        "subtitle": "Rows and columns detected during upload",
                        "valueFormat": "number",
                        "categories": ["Rows", "Columns"],
                        "series": [
                            {
                                "id": "dataset_structure_values",
                                "name": "Count",
                                "data": [row_count, column_count],
                            }
                        ],
                        "layout": {"columnSpan": 1, "rowSpan": 1},
                    },
                    {
                        "id": "data_quality",
                        "type": "donut",
                        "title": "Data Completeness",
                        "subtitle": "Complete and missing values",
                        "valueFormat": "percentage",
                        "segments": [
                            {
                                "id": "complete_values",
                                "label": "Complete",
                                "value": completeness,
                            },
                            {
                                "id": "missing_values",
                                "label": "Missing",
                                "value": round(100 - completeness, 2),
                            },
                        ],
                        "layout": {"columnSpan": 1, "rowSpan": 1},
                    },
                ],
                "analysis": {
                    "businessSummary": (
                        "The file was uploaded successfully and basic structure "
                        "checks were completed."
                    ),
                    "keyFindings": [
                        f"The uploaded file is named {dataset.file_name}.",
                        f"The dataset contains {row_count:,} rows.",
                        f"The dataset contains {column_count:,} columns.",
                        f"{dataset_info['missingValueCount']:,} missing values were detected.",
                    ],
                },
                "insights": {
                    "criticalAnomalies": [],
                    "warnings": [],
                    "limitations": [
                        {
                            "id": "ai_pipeline_unavailable",
                            "title": "AI pipeline unavailable",
                            "description": (
                                "Advanced KPIs, trends, forecasts, anomalies and "
                                "recommendations could not be generated by the AI agent."
                            ),
                            "severity": "info",
                            "sourceIds": [],
                        }
                    ],
                    "opportunities": [],
                },
                "recommendedActions": [
                    {
                        "id": "retry_ai_analysis",
                        "title": "Retry AI analysis",
                        "description": (
                            "Retry the business intelligence agent when the AI "
                            "pipeline is available."
                        ),
                        "priority": "medium",
                        "sourceIds": [],
                    },
                    {
                        "id": "review_dataset_quality",
                        "title": "Review dataset quality",
                        "description": (
                            "Resolve the detected missing values and duplicate records "
                            "before rerunning advanced analysis."
                        ),
                        "priority": "medium",
                        "sourceIds": ["dataset_summary"],
                    },
                    {
                        "id": "confirm_business_context",
                        "title": "Confirm business context",
                        "description": (
                            "Confirm the intended KPI definitions, reporting period, "
                            "and business priorities before the next analysis run."
                        ),
                        "priority": "low",
                        "sourceIds": ["dataset_summary"],
                    }
                ],
                "datasetSummary": {
                    "fileName": dataset.file_name,
                    "rowCount": row_count,
                    "columnCount": column_count,
                    "timeField": None,
                    "period": None,
                    "measures": dataset_info["measures"],
                    "dimensions": dataset_info["dimensions"],
                    "quality": {
                        "completenessPercent": completeness,
                        "missingValueCount": dataset_info["missingValueCount"],
                        "duplicateRowCount": dataset_info["duplicateRowCount"],
                    },
                    "generatedAt": generated_at,
                },
                "sections": [
                    {"id": "kpis", "title": "Key Performance Indicators", "order": 1, "visible": True},
                    {"id": "timeline", "title": "Performance Over Time", "order": 2, "visible": False},
                    {"id": "supportingCharts", "title": "Supporting Analysis", "order": 3, "visible": True},
                    {"id": "details", "title": "Insights and Recommendations", "order": 4, "visible": True},
                ],
                "layout": {
                    "kpis": {"columns": 4, "maxRows": 2},
                    "timeline": {"columnSpan": 12},
                    "supportingCharts": {"columns": 2, "maxRows": 2},
                    "details": {"columns": 2, "maxRows": 2},
                },
            },
            "warnings": [
                {
                    "code": "AI_PIPELINE_UNAVAILABLE",
                    "message": (
                        "The AI agent could not generate the dashboard, so this "
                        "fallback contains basic dataset information only."
                    ),
                    "component": "business_intelligence_agent",
                    "recoverable": True,
                }
            ],
            "errors": [],
        }

    @staticmethod
    def _kpi(
        item_id: str,
        title: str,
        value: int | float,
        suffix: str = "",
    ) -> dict[str, Any]:
        display = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        return {
            "id": item_id,
            "title": title,
            "value": f"{display}{suffix}",
            "rawValue": value,
            "indicator": {
                "kind": "note",
                "text": "Detected during upload",
            },
        }

    @staticmethod
    def _validate_upload_metadata(
        file_name: str,
        mime_type: str,
        extension: str,
    ) -> None:
        if not file_name or file_name in {".csv", ".xlsx"}:
            raise InvalidUploadError("The uploaded file must have a valid name.")
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidUploadError("Only CSV and XLSX files are supported.")
        if mime_type and mime_type not in ALLOWED_MIME_TYPES:
            raise InvalidUploadError("The uploaded file type is not supported.")

    @staticmethod
    def _sanitize_file_name(file_name: str) -> str:
        safe_name = Path(file_name or "uploaded-file").name.strip()
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix.lower()
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
        if not stem:
            stem = "uploaded-file"
        return f"{stem[:120]}{suffix}"

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        try:
            return str(UUID(session_id))
        except ValueError as error:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            ) from error

    @staticmethod
    def _chat_message(
        message: MessageRecord,
        grounded: bool | None = None,
    ) -> dict[str, Any]:
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "grounded": (
                bool(grounded)
                if grounded is not None
                else message.role == "assistant"
            ),
            "createdAt": message.created_at,
        }

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error).strip()
        if not text:
            return "Processing failed."
        return text[:300]

    @staticmethod
    def _is_duplicate_dataset_error(error: Exception) -> bool:
        code = str(getattr(error, "code", ""))
        message = str(error).lower()
        return code == "23505" or (
            "duplicate" in message and "user" in message and "dataset" in message
        )

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


business_intelligence_service = BusinessIntelligenceService()
