from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import UUID, uuid4

import pandas as pd
from fastapi import UploadFile

from app.agents.multi.chat_agent import GroundedChatDraft, chat_agent
from app.core.config import Settings, get_settings
from app.guardrails.chat_grounding import chat_grounding_guardrail
from app.schemas.business_intelligence import (
    ApiMessage,
    BusinessIntelligenceAgentInput,
    ChatResponse,
    DashboardResponse,
)
from app.orchestration.business_intelligence_graph import (
    business_intelligence_graph,
)
from app.rag.models import RetrievedDocument
from app.rag.rag_service import DEFAULT_RETRIEVAL_LIMIT, rag_service
from app.services.supabase_service import (
    DatasetRecord,
    MessageRecord,
    SupabaseService,
    supabase_service,
)


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
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


class BusinessIntelligenceService:
    def __init__(
        self,
        storage: SupabaseService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.storage = storage or supabase_service
        self.settings = settings or get_settings()

    async def create_analysis(
        self,
        file: UploadFile,
        description: str | None = None,
    ) -> dict[str, Any]:
        original_name = Path(file.filename or "").name
        file_name = self._sanitize_file_name(original_name)
        mime_type = (file.content_type or "").strip()
        extension = Path(file_name).suffix.lower()
        self._validate_upload_metadata(file_name, mime_type, extension)

        content = await file.read()
        if not content:
            raise InvalidUploadError("The uploaded file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise InvalidUploadError("The uploaded file is too large.")

        session_id = str(uuid4())
        storage_path = f"{session_id}/{file_name}"
        file_hash = hashlib.sha256(content).hexdigest()

        self.storage.upload_file(
            storage_path=storage_path,
            content=content,
            mime_type=mime_type,
        )

        dataset_inserted = False
        try:
            dataset = self.storage.create_dataset(
                dataset_id=session_id,
                file_name=file_name,
                storage_path=storage_path,
                mime_type=mime_type,
                file_size=len(content),
                file_hash=file_hash,
                description=description,
            )
            dataset_inserted = True

            dashboard_response = await self._run_selected_pipeline(dataset, content)
            self._save_dashboard(dataset.id, dashboard_response)

            if self.settings.bi_pipeline_mode == "single":
                self._try_index_rag(
                    dataset=dataset,
                    content=content,
                )

            self.storage.update_dataset_status(
                dataset.id,
                status=(
                    "failed" if dashboard_response.status == "failed" else "ready"
                ),
                error_message=(
                    "Business intelligence pipeline failed."
                    if dashboard_response.status == "failed"
                    else None
                ),
            )

        except Exception as error:
            safe_error = self._safe_error(error)
            logger.exception(
                "Dataset processing failed session_id=%s operation=upload storage_path=%s",
                session_id,
                storage_path,
            )
            if dataset_inserted:
                self.storage.update_dataset_status(
                    session_id,
                    status="failed",
                    error_message=safe_error,
                )
            else:
                self.storage.delete_file(storage_path)
            raise

        return {
            "status": "success",
            "sessionId": session_id,
            "fileName": file_name,
            "message": "File uploaded and analysis session created successfully.",
        }

    async def get_dashboard(self, session_id: str) -> DashboardResponse:
        dataset = self._load_dataset(session_id)
        dashboard = self.storage.get_dashboard(dataset.id)

        if dashboard is not None:
            try:
                return DashboardResponse.model_validate(dashboard.response)
            except Exception:
                logger.exception(
                    "Saved dashboard validation failed session_id=%s operation=get_dashboard",
                    dataset.id,
                )

        return await self._run_selected_pipeline(dataset)

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
        if self.settings.bi_pipeline_mode == "multi":
            return await self._run_multi_agent_pipeline(session, content)
        return await self._run_single_agent_pipeline(session, content)

    async def _run_single_agent_pipeline(
        self,
        session: DatasetRecord,
        content: bytes | None = None,
    ) -> DashboardResponse:
        return self._generate_dashboard_from_bytes(
            dataset=session,
            content=(
                content
                if content is not None
                else self.storage.download_file(session.storage_path)
            ),
        )

    async def _run_multi_agent_pipeline(
        self,
        session: DatasetRecord,
        content: bytes | None = None,
    ) -> DashboardResponse:
        """Run the compiled workflow and return only its canonical dashboard."""
        session_id = session.id
        content = (
            content
            if content is not None
            else self.storage.download_file(session.storage_path)
        )

        with self._temporary_agent_input(session, content) as agent_input:
            initial_state = {
                "session_id": session_id,
                "dataset_id": session.id,
                "business_description": session.description,
                "uploaded_file_path": agent_input.filePath,
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
                response = DashboardResponse.model_validate(dashboard_output)
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
                return failed_response

        logger.info("Multi-agent pipeline completed session_id=%s", session_id)
        logger.info(
            "Multi-agent final workflow status session_id=%s status=%s",
            session_id,
            response.status,
        )
        return response

    def chat(self, session_id: str, query: str) -> ChatResponse:
        dataset = self._load_dataset(session_id)
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")

        self.storage.save_message(
            dataset_id=dataset.id,
            role="user",
            content=cleaned_query,
            sources=[],
        )
        logger.info("Chat retrieval started session_id=%s", dataset.id)
        try:
            documents = rag_service.retrieve_for_session(
                session_id=dataset.id,
                query=cleaned_query,
                limit=DEFAULT_RETRIEVAL_LIMIT,
            )
            logger.info(
                "Chat documents retrieved session_id=%s count=%s",
                dataset.id,
                len(documents),
            )
        except Exception:
            logger.exception(
                "Chat retrieval or generation failed session_id=%s operation=retrieval",
                dataset.id,
            )
            return self._save_chat_response(
                dataset.id,
                "The analysis assistant could not answer this question at the moment.",
                [],
            )

        try:
            draft = asyncio.run(
                chat_agent.run(
                    session_id=dataset.id,
                    query=cleaned_query,
                    retrieved_documents=documents,
                )
            )
            guarded_draft = chat_grounding_guardrail.validate(
                query=cleaned_query,
                retrieved_documents=documents,
                draft=draft,
            )
            logger.info("Chat response guarded session_id=%s", dataset.id)
        except Exception:
            logger.exception(
                "Chat agent or grounding failed session_id=%s",
                dataset.id,
            )
            guarded_draft = GroundedChatDraft(
                answer="The analysis assistant could not answer this question at the moment.",
                source_ids=[],
                insufficient_context=True,
            )

        return self._save_chat_response(
            dataset.id,
            guarded_draft.answer,
            guarded_draft.source_ids,
        )

    def _save_chat_response(
        self,
        dataset_id: str,
        response_text: str,
        source_ids: list[str],
    ) -> ChatResponse:
        self.storage.save_message(
            dataset_id=dataset_id,
            role="assistant",
            content=response_text,
            sources=source_ids,
        )
        return ChatResponse(response=response_text)

    def get_chat_history(self, session_id: str) -> dict[str, Any]:
        dataset = self._load_dataset(session_id)
        return {
            "sessionId": dataset.id,
            "messages": [
                self._chat_message(message)
                for message in self.storage.get_recent_messages(dataset.id, limit=50)
            ],
        }

    def _load_dataset(self, session_id: str) -> DatasetRecord:
        dataset_id = self._validate_session_id(session_id)
        dataset = self.storage.get_dataset(dataset_id)
        if dataset is None:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            )
        return dataset

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

    def _try_index_rag(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> None:
        try:
            self.storage.update_dataset_status(dataset.id, rag_status="indexing")

            from app.agents.single.business_intelligence_agent import (
                business_intelligence_agent,
            )
            from app.rag.rag_service import rag_service

            with self._temporary_agent_input(dataset, content) as agent_input:
                profile = business_intelligence_agent.profile_for_session(agent_input)
                rag_service.index_dataset(
                    agent_input=agent_input,
                    profile=profile,
                    force=True,
                )

            self.storage.update_dataset_status(dataset.id, rag_status="ready")
        except Exception as error:
            logger.warning(
                "Recoverable RAG indexing failure session_id=%s operation=index_rag",
                dataset.id,
                exc_info=True,
            )
            self.storage.update_dataset_status(
                dataset.id,
                rag_status="failed",
                error_message=self._safe_error(error),
            )

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
        suffix = Path(dataset.file_name).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="bi_dataset_") as directory:
            path = Path(directory) / dataset.file_name
            if path.suffix.lower() != suffix:
                path = Path(directory) / f"dataset{suffix}"
            path.write_bytes(content)
            yield BusinessIntelligenceAgentInput(
                sessionId=dataset.id,
                filePath=str(path),
                fileName=dataset.file_name,
                description=dataset.description,
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
    def _current_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


business_intelligence_service = BusinessIntelligenceService()
