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

    async def create_analysis(
        self,
        file: UploadFile,
        user_id: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("An authenticated user is required.")
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

        dataset_info = self._inspect_file(file_name, content)
        if self.storage.get_active_dataset(user_id) is not None:
            raise DatasetAlreadyExistsError(
                "You already have an active dataset. Use Start Over before uploading another file."
            )

        session_id = str(uuid4())
        storage_path = f"{user_id}/{session_id}/{file_name}"
        file_hash = hashlib.sha256(content).hexdigest()

        dataset_inserted = False
        file_uploaded = False
        try:
            dataset = self.storage.create_dataset(
                dataset_id=session_id,
                user_id=user_id,
                file_name=file_name,
                storage_path=storage_path,
                mime_type=mime_type,
                file_size=len(content),
                file_hash=file_hash,
                description=description,
                row_count=int(dataset_info["rowCount"]),
                column_count=int(dataset_info["columnCount"]),
            )
            dataset_inserted = True

            self.storage.upload_file(
                storage_path=storage_path,
                content=content,
                mime_type=mime_type,
            )
            file_uploaded = True

            await self._run_selected_pipeline(dataset, content)

        except Exception as error:
            if not dataset_inserted and self._is_duplicate_dataset_error(error):
                raise DatasetAlreadyExistsError(
                    "You already have an active dataset. Use Start Over before uploading another file."
                ) from error

            safe_error = self._safe_error(error)
            logger.exception(
                "Dataset processing failed session_id=%s operation=upload storage_path=%s",
                session_id,
                storage_path,
            )
            if dataset_inserted:
                if file_uploaded:
                    self.storage.update_dataset_status(
                        session_id,
                        status="failed",
                        error_message=safe_error,
                    )
                else:
                    self.storage.delete_dataset(session_id, user_id)
            raise

        return {
            "status": "success",
            "sessionId": session_id,
            "fileName": file_name,
            "message": "File uploaded and analysis session created successfully.",
        }

    def get_active_dataset(self, user_id: str) -> DatasetRecord:
        if not user_id:
            raise SessionNotFoundError("No active dataset was found.")
        dataset = self.storage.get_active_dataset(user_id)
        if dataset is None:
            raise SessionNotFoundError("No active dataset was found.")
        return dataset

    def get_active_dataset_details(self, user_id: str) -> dict[str, Any]:
        dataset = self.get_active_dataset(user_id)
        row_count = dataset.row_count
        column_count = dataset.column_count
        if row_count is None or column_count is None:
            dataset_info = self._inspect_file(
                dataset.file_name,
                self.storage.download_file(dataset.storage_path),
            )
            row_count = int(dataset_info["rowCount"])
            column_count = int(dataset_info["columnCount"])

        return {
            "sessionId": dataset.id,
            "fileName": dataset.file_name,
            "fileSize": dataset.file_size,
            "uploadedAt": dataset.created_at or "",
            "rowCount": row_count,
            "columnCount": column_count,
            "analysisStatus": dataset.status,
            "ragStatus": dataset.rag_status,
            "originalPrompt": dataset.description,
        }

    def get_dataset_preview(
        self,
        user_id: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        dataset = self.get_active_dataset(user_id)
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
        dataset = self.get_active_dataset(user_id)
        self.storage.delete_file(dataset.storage_path)
        self.storage.delete_dataset(dataset.id, user_id)

    async def get_dashboard(self, session_id: str, user_id: str) -> DashboardResponse:
        dataset = self._load_dataset(session_id, user_id)
        dashboard = self.storage.get_dashboard(dataset.id)

        if dashboard is not None:
            try:
                return self._with_dashboard_model_metadata(
                    DashboardResponse.model_validate(dashboard.response)
                )
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
    ) -> PipelineExecution:
        """Run the compiled workflow and return its dashboard and durable artifacts."""
        session_id = session.id
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
        if self.settings.bi_pipeline_mode == "multi":
            return self._chat_with_multi_agent_pipeline(
                session_id=session_id,
                query=query,
                user_id=user_id,
            )

        dataset = self._load_dataset(session_id, user_id)
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
        dataset = self._load_dataset(session_id, user_id)
        history = self._chat_history(dataset.id)
        result = self._chat_service.answer(dataset.id, query, history=history)

        self.storage.save_message(
            dataset_id=dataset.id,
            role="user",
            content=result.query,
            sources=[],
        )
        return self._save_chat_response(
            dataset.id,
            result.draft.answer,
            result.draft.source_ids,
        )

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
        dataset = self._load_dataset(session_id, user_id)
        return {
            "sessionId": dataset.id,
            "messages": [
                self._chat_message(message)
                for message in self.storage.get_recent_messages(dataset.id, limit=50)
            ],
        }

    def _load_dataset(self, session_id: str, user_id: str) -> DatasetRecord:
        dataset_id = self._validate_session_id(session_id)
        if not user_id:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            )
        dataset = self.storage.get_dataset(dataset_id, user_id)
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
