"""Business-intelligence application facade implementation.

Detailed parsing, persistence, pipeline, indexing, and chat behavior lives in
the focused services under app.services.analysis.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
)
from app.orchestration.graphs.analysis_graph import analysis_graph
from app.orchestration.graphs.chat_graph import ChatGraph
from app.rag.indexing.indexing_service import IndexingService, indexing_service
from app.rag.retrieval.retriever import retriever
from app.rag.vector_store import VectorStore, vector_store
from app.schemas.api import ChatResponse, DashboardResponse, UploadCandidate
from app.services.analysis.chat import BusinessIntelligenceChatService
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.indexing import AnalysisExecutionPersistenceService
from app.services.analysis.models import BackgroundTaskScheduler, PipelineExecution
from app.services.analysis.pipelines import AnalysisPipelineRunner
from app.services.analysis.uploads import DatasetUploadService
from app.services.analysis.workspaces import WorkspaceService
from app.services.persistence.analysis import (
    AnalysisRepository,
    DatasetRecord,
    analysis_repository,
)
from app.services.persistence.messages import MessageRepository, message_repository
from app.services.persistence.supabase import (
    SupabaseGateway,
    supabase_gateway,
)

logger = logging.getLogger(__name__)


class BusinessIntelligenceService:
    def __init__(
        self,
        storage: SupabaseGateway | Any | None = None,
        settings: Settings | None = None,
        *,
        analysis_repo: AnalysisRepository | Any | None = None,
        message_repo: MessageRepository | Any | None = None,
        vector_store_adapter: VectorStore | Any | None = None,
        rag: Any | None = None,
        indexing: IndexingService | Any | None = None,
        multi_chat_agent: Any | None = None,
        chat_service: ChatGraph | None = None,
        file_service: DatasetFileService | None = None,
        upload_service: DatasetUploadService | None = None,
        workspace_service: WorkspaceService | None = None,
        pipeline_runner: AnalysisPipelineRunner | None = None,
        execution_persistence: AnalysisExecutionPersistenceService | None = None,
        dashboard_assembler: DashboardAssembler | None = None,
        bi_chat_service: BusinessIntelligenceChatService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or supabase_gateway

        # Temporary compatibility for combined test/integration doubles. Remove
        # once every caller injects the repository interfaces explicitly.
        self.analysis = analysis_repo or (
            storage
            if storage is not None
            and (
                hasattr(storage, "create_dataset")
                or hasattr(storage, "get_dataset")
                or hasattr(storage, "save_session_processing")
            )
            else analysis_repository
        )
        self.messages = message_repo or (
            storage
            if storage is not None
            and hasattr(storage, "save_message")
            and hasattr(storage, "get_recent_messages")
            else message_repository
        )
        self.vector_store = vector_store_adapter or vector_store
        self._retriever = rag or retriever
        self._indexing_service = indexing or (
            rag
            if rag is not None
            and (
                hasattr(rag, "index_documents")
                or hasattr(rag, "index_dataset")
            )
            else indexing_service
        )
        calculation_retriever = (
            rag
            if rag is not None
            and hasattr(rag, "route_query")
            and hasattr(rag, "calculate_evidence")
            else retriever
        )

        self._file_service = file_service or DatasetFileService()
        self._dashboard_assembler = dashboard_assembler or DashboardAssembler(
            settings=self.settings,
            files=self._file_service,
        )
        self._workspace_service = workspace_service or WorkspaceService(
            analysis=self.analysis,
            storage=self.storage,
            files=self._file_service,
        )
        self._upload_service = upload_service or DatasetUploadService(
            analysis=self.analysis,
            storage=self.storage,
            files=self._file_service,
        )
        self._pipeline_runner = pipeline_runner or AnalysisPipelineRunner(
            settings=self.settings,
            files=self._file_service,
            dashboards=self._dashboard_assembler,
            storage=self.storage,
            graph=analysis_graph,
        )
        self._execution_persistence = (
            execution_persistence
            or AnalysisExecutionPersistenceService(
                analysis=self.analysis,
                indexing=self._indexing_service,
                retriever=self._retriever,
                settings=self.settings,
                files=self._file_service,
                dashboards=self._dashboard_assembler,
            )
        )
        chat_graph = chat_service or ChatGraph(
            rag=self._retriever,
            agent=multi_chat_agent,
        )
        self._chat_service = bi_chat_service or BusinessIntelligenceChatService(
            workspaces=self._workspace_service,
            messages=self.messages,
            storage=self.storage,
            retriever=calculation_retriever,
            chat_graph=chat_graph,
            settings=self.settings,
            files=self._file_service,
        )
        self._legacy_contract_sessions: set[str] = set()

    def uses_legacy_contract(self, session_id: str) -> bool:
        return session_id in self._legacy_contract_sessions

    async def create_analysis(
        self,
        files: list[UploadCandidate] | UploadCandidate | Any,
        user_id: str,
        description: str | None = None,
        legacy_contract: bool = False,
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> dict[str, Any]:
        workspace = await self._upload_service.create_workspace(
            files=files,
            user_id=user_id,
            description=description,
        )
        try:
            await self._run_workspace_pipeline(
                workspace.session,
                workspace.datasets,
                workspace.contents,
                background_tasks=background_tasks,
            )
        except Exception:
            logger.exception(
                "Workspace processing failed session_id=%s operation=upload",
                workspace.session.id,
            )
            self._upload_service.rollback_new_workspace(
                session_id=workspace.session.id,
                user_id=user_id,
                datasets=workspace.datasets,
                uploaded_paths=workspace.uploaded_paths,
                session_created=workspace.session_created,
            )
            raise
        if legacy_contract:
            self._legacy_contract_sessions.add(workspace.session.id)
        result = {
            "status": "success",
            "sessionId": workspace.session.id,
            "datasetIds": [dataset.id for dataset in workspace.datasets],
            "message": "Files uploaded and analysis workspace created successfully.",
        }
        if len(workspace.datasets) == 1:
            result["fileName"] = workspace.datasets[0].file_name
        return result

    async def add_datasets(
        self,
        files: list[UploadCandidate] | UploadCandidate | Any,
        user_id: str,
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> dict[str, Any]:
        session, existing = self._workspace_service.active_workspace(user_id)
        if session.requires_reset:
            raise InvalidUploadError(
                "This legacy workspace must be reset before datasets can be added."
            )
        self._workspace_service.ensure_mutable(session)
        existing_contents = [
            self.storage.download_file(dataset.storage_path)
            for dataset in existing
        ]
        added = await self._upload_service.add_to_workspace(
            files=files,
            session=session,
            existing_datasets=existing,
        )
        datasets = [*existing, *added.datasets]
        contents = [*existing_contents, *added.contents]
        self._workspace_service.prepare_reanalysis(session, datasets)
        try:
            await self._run_workspace_pipeline(
                session,
                datasets,
                contents,
                background_tasks=background_tasks,
            )
        except Exception as error:
            self._workspace_service.mark_reanalysis_failed(
                session,
                datasets,
                error,
            )
            raise
        return {
            "status": "success",
            "sessionId": session.id,
            "datasetIds": [dataset.id for dataset in added.datasets],
            "message": "Datasets added and workspace analysis rebuilt successfully.",
        }

    async def remove_dataset(
        self,
        dataset_id: str,
        user_id: str,
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> None:
        session, datasets = self._workspace_service.active_workspace(user_id)
        if len(datasets) == 1 and datasets[0].id == dataset_id:
            self.reset_active_dataset(user_id)
            return
        session, remaining, _ = self._workspace_service.remove_dataset_source(
            dataset_id,
            user_id,
        )
        contents = [
            self.storage.download_file(dataset.storage_path)
            for dataset in remaining
        ]
        self._workspace_service.prepare_reanalysis(session, remaining)
        try:
            await self._run_workspace_pipeline(
                session,
                remaining,
                contents,
                background_tasks=background_tasks,
            )
        except Exception as error:
            self._workspace_service.mark_reanalysis_failed(
                session,
                remaining,
                error,
            )
            raise

    def get_active_dataset(self, user_id: str) -> DatasetRecord:
        return self._workspace_service.get_active_dataset(user_id)

    def get_active_dataset_details(self, user_id: str) -> dict[str, Any]:
        session, _ = self._workspace_service.active_workspace(user_id)
        return self._workspace_service.get_active_dataset_details(
            user_id,
            legacy_contract=session.id in self._legacy_contract_sessions,
        )

    def get_dataset_preview(
        self,
        user_id: str,
        dataset_id: str | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        return self._workspace_service.get_dataset_preview(
            user_id,
            dataset_id,
            page,
            page_size,
        )

    def reset_active_dataset(self, user_id: str) -> None:
        session_id = self._workspace_service.reset_active_dataset(user_id)
        self._legacy_contract_sessions.discard(session_id)

    async def get_dashboard(
        self,
        session_id: str,
        user_id: str,
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> DashboardResponse:
        session, datasets = self._workspace_service.load_workspace(
            session_id,
            user_id,
        )
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before its dashboard is available."
            )
        dashboard = self.analysis.get_dashboard(session.id)
        if dashboard is not None:
            try:
                return self._dashboard_assembler.with_model_metadata(
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
        return await self._run_workspace_pipeline(
            session,
            datasets,
            contents,
            background_tasks=background_tasks,
        )

    def rebuild_dashboard_retrieval(
        self,
        session_id: str,
        user_id: str,
        background_tasks: BackgroundTaskScheduler,
    ) -> None:
        session, datasets = self._workspace_service.load_workspace(
            session_id,
            user_id,
        )
        self._execution_persistence.rebuild_dashboard_retrieval(
            session=session,
            datasets=datasets,
            background_tasks=background_tasks,
        )

    def chat(self, session_id: str, query: str, user_id: str) -> ChatResponse:
        return self._chat_service.chat(session_id, query, user_id)

    def get_chat_history(self, session_id: str, user_id: str) -> dict[str, Any]:
        return self._chat_service.get_chat_history(session_id, user_id)

    async def _run_workspace_pipeline(
        self,
        session: Any,
        datasets: list[DatasetRecord],
        contents: list[bytes],
        background_tasks: BackgroundTaskScheduler | None = None,
    ) -> DashboardResponse:
        if len(datasets) != len(contents) or not datasets:
            raise ValueError("Every workspace dataset requires source content.")
        workspace_dataset, workspace_content = (
            self._pipeline_runner.workspace_analysis_input(
                session,
                datasets,
                contents,
            )
        )
        if self.settings.bi_pipeline_mode == "multi":
            execution = await self._call_multi_pipeline_compatibly(
                workspace_dataset,
                workspace_content,
                session.id,
                datasets,
            )
        else:
            execution = await self._run_single_agent_pipeline(
                workspace_dataset,
                workspace_content,
            )
        if isinstance(execution, DashboardResponse):
            execution = PipelineExecution(response=execution)
        combined = self._dashboard_assembler.combine_workspace_executions(
            session,
            datasets,
            [execution],
            contents,
        )
        return self._execution_persistence.persist(
            session=session,
            datasets=datasets,
            contents=contents,
            execution=combined,
            background_tasks=background_tasks,
        )

    async def _call_multi_pipeline_compatibly(
        self,
        dataset: DatasetRecord,
        content: bytes,
        session_id: str,
        datasets: list[DatasetRecord],
    ) -> PipelineExecution | DashboardResponse:
        # Existing extensions and tests replace this seam with older call
        # signatures. Signature filtering avoids using TypeError as control flow.
        runner = self._run_multi_agent_pipeline
        parameters = inspect.signature(runner).parameters
        kwargs: dict[str, Any] = {}
        if "workspace_session_id" in parameters:
            kwargs["workspace_session_id"] = session_id
        if "workspace_datasets" in parameters:
            kwargs["workspace_datasets"] = datasets
        return await runner(dataset, content, **kwargs)

    async def _run_single_agent_pipeline(
        self,
        dataset: DatasetRecord,
        content: bytes | None = None,
    ) -> PipelineExecution:
        return await self._pipeline_runner.run_single_agent(dataset, content)

    async def _run_multi_agent_pipeline(
        self,
        dataset: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
        workspace_datasets: list[DatasetRecord] | None = None,
    ) -> PipelineExecution:
        return await self._pipeline_runner.run_multi_agent(
            dataset,
            content,
            workspace_session_id,
            workspace_datasets,
            graph=analysis_graph,
        )

    # Narrow compatibility delegates retained for callers identified before the
    # refactor. New code should depend on the focused services directly.
    def _inspect_file(self, file_name: str, content: bytes) -> dict[str, Any]:
        return self._file_service.inspect_file(file_name, content).api_dict()

    def _build_placeholder_dashboard(
        self,
        dataset: DatasetRecord,
        dataset_info: dict[str, Any],
    ) -> dict[str, Any]:
        inspection = self._file_service.inspect_file(
            dataset.file_name,
            self.storage.download_file(dataset.storage_path),
        ) if not dataset_info else None
        if inspection is None:
            from app.services.analysis.models import DatasetInspection

            inspection = DatasetInspection(
                row_count=int(dataset_info["rowCount"]),
                column_count=int(dataset_info["columnCount"]),
                measures=list(dataset_info["measures"]),
                dimensions=list(dataset_info["dimensions"]),
                missing_value_count=int(dataset_info["missingValueCount"]),
                duplicate_row_count=int(dataset_info["duplicateRowCount"]),
                completeness_percent=float(dataset_info["completenessPercent"]),
            )
        return self._dashboard_assembler.build_placeholder_dashboard(
            dataset,
            inspection,
        )

    def _select_chat_datasets(
        self,
        query: str,
        datasets: list[DatasetRecord],
    ) -> tuple[DatasetRecord | None, list[DatasetRecord]]:
        return self._chat_service.select_chat_datasets(query, datasets)

    def _workspace_calculation_response(
        self,
        query: str,
        datasets: list[DatasetRecord],
    ) -> str | None:
        return self._chat_service.workspace_calculation_response(query, datasets)

    def _save_workflow_dashboard(
        self,
        session_id: str,
        response: DashboardResponse,
        workflow: dict[str, Any],
    ) -> None:
        self._execution_persistence.save_workflow_dashboard(
            session_id,
            response,
            workflow,
        )

    def _try_index_rag(self, dataset: DatasetRecord, content: bytes) -> bool:
        return self._execution_persistence.try_index_rag(dataset, content)
