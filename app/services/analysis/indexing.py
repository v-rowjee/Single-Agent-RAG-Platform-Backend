"""Persistence and retrieval-indexing coordination for analysis executions."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.multi.retrieval_preparation import dashboard_retrieval_documents
from app.agents.single import business_intelligence as single_agent_module
from app.core.config import Settings
from app.schemas.api import ApiMessage, DashboardResponse
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import BackgroundTaskScheduler, PipelineExecution
from app.services.persistence.analysis import (
    AnalysisRepository,
    AnalysisSessionRecord,
    DatasetRecord,
)

logger = logging.getLogger(__name__)


class AnalysisExecutionPersistenceService:
    def __init__(
        self,
        *,
        analysis: AnalysisRepository | Any,
        indexing: Any,
        retriever: Any,
        settings: Settings,
        files: DatasetFileService,
        dashboards: DashboardAssembler,
        single_agent: Any | None = None,
    ) -> None:
        self.analysis = analysis
        self.indexing = indexing
        self.retriever = retriever
        self.settings = settings
        self.files = files
        self.dashboards = dashboards
        self.single_agent = single_agent

    def persist(
        self,
        *,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
        execution: PipelineExecution,
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> DashboardResponse:
        response = execution.response
        update_session = getattr(self.analysis, "update_session_status", None)
        if callable(update_session):
            update_session(session.id, rag_status="indexing")
        else:
            for dataset in datasets:
                self.analysis.update_dataset_status(dataset.id, rag_status="indexing")

        if not hasattr(self.indexing, "index_documents"):
            rag_status = (
                "ready"
                if response.status != "failed"
                and self.try_index_rag(datasets[0], contents[0])
                else "failed"
            )
            self.save_dashboard(session.id, response)
            for dataset in datasets:
                self.analysis.update_dataset_status(
                    dataset.id,
                    status="failed" if response.status == "failed" else "ready",
                    rag_status=rag_status,
                    error_message=None,
                )
            return response

        documents = list(execution.retrieval_documents or [])
        if self.settings.bi_pipeline_mode == "single" and response.status != "failed":
            documents = self.single_workspace_documents(session, datasets, contents)

        if background_tasks is not None and response.status != "failed":
            workflow = self._workflow(
                session,
                datasets,
                execution,
                response,
                {
                    "status": "indexing",
                    "document_count": len(documents),
                    "indexed_count": 0,
                    "failed_count": 0,
                    "message": "Retrieval indexing is running in the background.",
                },
            )
            self.save_workflow_dashboard(session.id, response, workflow)
            self._update_statuses(
                session,
                datasets,
                status="ready",
                rag_status="indexing",
                error_message=None,
            )
            background_tasks.add_task(
                self.complete_workspace_indexing,
                session=session,
                datasets=datasets,
                retrieval_documents=documents,
                response=response,
                workflow=workflow,
            )
            return response

        if response.status == "failed":
            result = {
                "status": "failed",
                "document_count": len(documents),
                "indexed_count": 0,
                "failed_count": len(documents),
                "message": "Retrieval indexing was skipped because analysis failed.",
            }
        else:
            result = self.indexing.index_documents(
                session_id=session.id,
                dataset_id=session.id,
                retrieval_documents=documents,
            )
        rag_status = "ready" if result.get("status") == "success" else "failed"
        response = self._with_indexing_warning(response, rag_status)
        workflow = self._workflow(session, datasets, execution, response, result)
        self.save_workflow_dashboard(session.id, response, workflow)
        self._update_statuses(
            session,
            datasets,
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

    def complete_workspace_indexing(
        self,
        *,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        retrieval_documents: list[dict[str, Any]],
        response: DashboardResponse,
        workflow: dict[str, Any],
    ) -> None:
        try:
            result = self.indexing.index_documents(
                session_id=session.id,
                dataset_id=session.id,
                retrieval_documents=retrieval_documents,
            )
        except Exception as error:
            logger.exception(
                "Background retrieval indexing crashed session_id=%s",
                session.id,
            )
            result = {
                "status": "failed",
                "document_count": len(retrieval_documents),
                "indexed_count": 0,
                "failed_count": len(retrieval_documents),
                "message": self.safe_error(error),
            }
        rag_status = "ready" if result.get("status") == "success" else "failed"
        completed_response = self._with_indexing_warning(response, rag_status)
        completed_workflow = dict(workflow)
        completed_workflow.update(
            {
                "workflow_status": completed_response.status,
                "retrieval_indexing_result": result,
            }
        )
        try:
            self.save_workflow_dashboard(
                session.id,
                completed_response,
                completed_workflow,
            )
        except Exception:
            logger.exception(
                "Unable to persist background indexing result session_id=%s",
                session.id,
            )
        try:
            self._update_statuses(
                session,
                datasets,
                status="ready",
                rag_status=rag_status,
                error_message=(
                    "Retrieval indexing failed." if rag_status == "failed" else None
                ),
            )
        except Exception:
            logger.exception(
                "Unable to persist background indexing status session_id=%s",
                session.id,
            )

    def rebuild_dashboard_retrieval(
        self,
        *,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        background_tasks: BackgroundTaskScheduler,
    ) -> None:
        dashboard_record = self.analysis.get_dashboard(session.id)
        if dashboard_record is None:
            raise ValueError("Generate the dashboard before rebuilding retrieval.")
        response = DashboardResponse.model_validate(dashboard_record.response)
        documents = [
            item.model_dump(mode="json")
            for item in dashboard_retrieval_documents(
                response.model_dump(mode="json")
            )
        ]
        if not documents:
            raise ValueError("The saved dashboard contains no retrieval evidence.")
        documents = self.dashboards.workspace_retrieval_documents(
            session,
            datasets,
            documents,
        )
        workflow = {
            "session_id": session.id,
            "dataset_ids": [dataset.id for dataset in datasets],
            "workflow_status": response.status,
            "retrieval_indexing_result": {
                "status": "indexing",
                "document_count": len(documents),
                "indexed_count": 0,
                "failed_count": 0,
                "message": "Dashboard retrieval indexing is running in the background.",
            },
        }
        self.save_workflow_dashboard(session.id, response, workflow)
        self._update_statuses(
            session,
            datasets,
            status="ready",
            rag_status="indexing",
            error_message=None,
        )
        background_tasks.add_task(
            self.complete_workspace_indexing,
            session=session,
            datasets=datasets,
            retrieval_documents=documents,
            response=response,
            workflow=workflow,
        )

    def save_dashboard(
        self,
        session_id: str,
        response: DashboardResponse,
    ) -> None:
        self.analysis.save_dashboard(
            dataset_id=session_id,
            status=response.status,
            response=response.model_dump(mode="json"),
        )

    def save_workflow_dashboard(
        self,
        session_id: str,
        response: DashboardResponse,
        workflow: dict[str, Any],
    ) -> None:
        cleaning = self.persistent_cleaning_report(
            workflow.get("generic_cleaning_report")
        )
        prepared = self.persistent_prepared_dataset(
            workflow.get("prepared_dataset")
        )
        persistent_workflow = dict(workflow)
        persistent_workflow["generic_cleaning_report"] = cleaning
        persistent_workflow["prepared_dataset"] = prepared
        persistent_workflow.pop("retrieval_documents", None)
        self.analysis.save_session_processing(
            dataset_id=session_id,
            workflow_status=response.status,
            generic_cleaning_report=cleaning,
            prepared_dataset=prepared,
        )
        stored = response.model_dump(mode="json")
        stored["workflow"] = {
            key: value
            for key, value in persistent_workflow.items()
            if key != "dashboard_output"
        }
        stored["workflow"]["dashboard_output"] = stored.copy()
        stored["workflow"]["dashboard_output"].pop("workflow", None)
        self.analysis.save_dashboard(
            dataset_id=session_id,
            status=response.status,
            response=stored,
        )

    def single_workspace_documents(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> list[dict[str, Any]]:
        try:
            documents: list[dict[str, Any]] = []
            for dataset, content in zip(datasets, contents, strict=True):
                with self.files.temporary_agent_input(dataset, content) as agent_input:
                    agent = (
                        self.single_agent
                        or single_agent_module.business_intelligence_agent
                    )
                    profile = agent.profile_for_session(agent_input)
                    documents.extend(
                        self.retriever.build_dataset_documents(
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

    def try_index_rag(self, dataset: DatasetRecord, content: bytes) -> bool:
        try:
            with self.files.temporary_agent_input(dataset, content) as agent_input:
                agent = (
                    self.single_agent
                    or single_agent_module.business_intelligence_agent
                )
                profile = agent.profile_for_session(agent_input)
                self.indexing.index_dataset(
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

    @staticmethod
    def persistent_cleaning_report(value: Any) -> dict[str, Any]:
        report = dict(value) if isinstance(value, dict) else {}
        report.pop("cleaned_file_path", None)
        return report

    @classmethod
    def persistent_prepared_dataset(cls, value: Any) -> dict[str, Any]:
        prepared = dict(value) if isinstance(value, dict) else {}
        prepared.pop("prepared_file_path", None)
        prepared.pop("temporal_dataset_path", None)
        cleaning = prepared.get("cleaning_report")
        if isinstance(cleaning, dict):
            prepared["cleaning_report"] = cls.persistent_cleaning_report(cleaning)
        return prepared

    @staticmethod
    def safe_error(error: Exception) -> str:
        text = str(error).strip()
        return (text or "Processing failed.")[:300]

    def _update_statuses(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        *,
        status: str,
        rag_status: str,
        error_message: str | None,
    ) -> None:
        update_session = getattr(self.analysis, "update_session_status", None)
        if callable(update_session):
            update_session(
                session.id,
                status=status,
                rag_status=rag_status,
                error_message=error_message,
            )
            return
        for dataset in datasets:
            self.analysis.update_dataset_status(
                dataset.id,
                status=status,
                rag_status=rag_status,
                error_message=error_message,
            )

    @staticmethod
    def _with_indexing_warning(
        response: DashboardResponse,
        rag_status: str,
    ) -> DashboardResponse:
        if rag_status != "failed" or response.status == "failed":
            return response
        return response.model_copy(
            update={
                "status": "partial",
                "warnings": [
                    *response.warnings,
                    ApiMessage(
                        code="RAG_INDEXING_FAILED",
                        message=(
                            "The dashboard was generated, but workspace retrieval "
                            "indexing could not be completed."
                        ),
                        component="rag",
                        recoverable=True,
                    ),
                ],
            }
        )

    @staticmethod
    def _workflow(
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        execution: PipelineExecution,
        response: DashboardResponse,
        indexing_result: dict[str, Any],
    ) -> dict[str, Any]:
        workflow = dict(execution.workflow or {})
        workflow.update(
            {
                "session_id": session.id,
                "dataset_ids": [dataset.id for dataset in datasets],
                "workflow_status": response.status,
                "retrieval_indexing_result": indexing_result,
            }
        )
        return workflow
