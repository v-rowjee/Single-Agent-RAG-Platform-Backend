"""Workspace loading, ownership, metadata, and lifecycle operations."""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from app.core.exceptions import InvalidUploadError, SessionNotFoundError
from app.services.analysis.files import DatasetFileService
from app.services.persistence.analysis import (
    AnalysisRepository,
    AnalysisSessionRecord,
    DatasetRecord,
)
from app.services.persistence.supabase import SupabaseGateway


class WorkspaceService:
    def __init__(
        self,
        *,
        analysis: AnalysisRepository | Any,
        storage: SupabaseGateway | Any,
        files: DatasetFileService,
    ) -> None:
        self.analysis = analysis
        self.storage = storage
        self.files = files

    def active_workspace(
        self,
        user_id: str,
    ) -> tuple[AnalysisSessionRecord, list[DatasetRecord]]:
        if not user_id:
            raise SessionNotFoundError("No active dataset was found.")
        get_active_session = getattr(self.analysis, "get_active_session", None)
        if callable(get_active_session):
            session = get_active_session(user_id)
            if session is None:
                raise SessionNotFoundError("No active dataset was found.")
            datasets = self.analysis.get_session_datasets(session.id, user_id)
            if not datasets:
                raise SessionNotFoundError("No active dataset was found.")
            return session, datasets
        dataset = self.analysis.get_active_dataset(user_id)
        if dataset is None:
            raise SessionNotFoundError("No active dataset was found.")
        return self._legacy_session(dataset), [dataset]

    def load_workspace(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[AnalysisSessionRecord, list[DatasetRecord]]:
        normalized_id = self.validate_session_id(session_id)
        if not user_id:
            raise self._not_found(session_id)
        get_session = getattr(self.analysis, "get_session", None)
        if callable(get_session):
            session = get_session(normalized_id, user_id)
            if session is None:
                raise self._not_found(session_id)
            datasets = self.analysis.get_session_datasets(session.id, user_id)
            if not datasets:
                raise self._not_found(session_id)
            return session, datasets

        dataset = self.analysis.get_dataset(normalized_id, user_id)
        if dataset is None:
            get_active = getattr(self.storage, "get_active_dataset", None)
            active = get_active(user_id) if callable(get_active) else None
            if active is not None and (active.session_id or active.id) == normalized_id:
                dataset = active
        if dataset is None:
            raise self._not_found(session_id)
        return self._legacy_session(dataset), [dataset]

    def get_active_dataset(self, user_id: str) -> DatasetRecord:
        return self.active_workspace(user_id)[1][0]

    def get_active_dataset_details(
        self,
        user_id: str,
        *,
        legacy_contract: bool = False,
    ) -> dict[str, Any]:
        session, datasets = self.active_workspace(user_id)
        metadata: list[dict[str, Any]] = []
        for dataset in datasets:
            row_count = dataset.row_count
            column_count = dataset.column_count
            if row_count is None or column_count is None:
                inspection = self.files.inspect_file(
                    dataset.file_name,
                    self.storage.download_file(dataset.storage_path),
                )
                row_count = inspection.row_count
                column_count = inspection.column_count
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
        if legacy_contract and len(metadata) == 1:
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
        session, datasets = self.active_workspace(user_id)
        dataset = datasets[0] if dataset_id is None else next(
            (item for item in datasets if item.id == dataset_id),
            None,
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
        total_rows = dataset.row_count
        if total_rows is None:
            total_rows = self.files.inspect_file(
                dataset.file_name,
                content,
            ).row_count
        columns, rows = self.files.read_preview_page(
            dataset.file_name,
            content,
            page,
            page_size,
        )
        return {
            "columns": columns,
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": math.ceil(total_rows / page_size) if total_rows else 0,
        }

    def reset_active_dataset(self, user_id: str) -> str:
        session, datasets = self.active_workspace(user_id)
        for dataset in datasets:
            self.storage.delete_file(dataset.storage_path)
        if callable(getattr(self.analysis, "delete_session", None)):
            self.analysis.delete_session(session.id, user_id)
        else:
            for dataset in datasets:
                self.analysis.delete_dataset(dataset.id, user_id)
        return session.id

    def remove_dataset_source(
        self,
        dataset_id: str,
        user_id: str,
    ) -> tuple[AnalysisSessionRecord, list[DatasetRecord], DatasetRecord]:
        session, datasets = self.active_workspace(user_id)
        if session.requires_reset:
            raise InvalidUploadError(
                "This legacy workspace must be reset before datasets can be removed."
            )
        self.ensure_mutable(session)
        target = next((item for item in datasets if item.id == dataset_id), None)
        if target is None:
            raise SessionNotFoundError(
                f"Dataset '{dataset_id}' was not found in the active workspace."
            )
        remaining = [item for item in datasets if item.id != target.id]
        self.analysis.delete_dataset(target.id, user_id)
        try:
            self.storage.delete_file(target.storage_path)
        except Exception:
            # Metadata deletion is authoritative; callers can still reanalyse the
            # remaining valid source files.
            import logging

            logging.getLogger(__name__).exception(
                "Dataset metadata was removed but its stored object could not be deleted path=%s",
                target.storage_path,
            )
        return session, remaining, target

    @staticmethod
    def ensure_mutable(session: AnalysisSessionRecord) -> None:
        if session.status == "processing" or session.rag_status == "indexing":
            raise InvalidUploadError(
                "Workspace analysis or retrieval indexing is already in progress. "
                "Try again when it has finished."
            )

    def prepare_reanalysis(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
    ) -> None:
        clear = getattr(self.analysis, "clear_session_analysis", None)
        if callable(clear):
            clear(session.id)
        update_session = getattr(self.analysis, "update_session_status", None)
        if callable(update_session):
            update_session(
                session.id,
                status="processing",
                rag_status="pending",
                error_message=None,
            )
        update_dataset = getattr(self.analysis, "update_dataset_status", None)
        if callable(update_dataset):
            for dataset in datasets:
                update_dataset(
                    dataset.id,
                    status="processing",
                    rag_status="pending",
                    error_message=None,
                )

    def mark_reanalysis_failed(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        error: Exception,
    ) -> None:
        message = self.safe_error(error)
        update_session = getattr(self.analysis, "update_session_status", None)
        if callable(update_session):
            update_session(
                session.id,
                status="failed",
                rag_status="failed",
                error_message=message,
            )
        update_dataset = getattr(self.analysis, "update_dataset_status", None)
        if callable(update_dataset):
            for dataset in datasets:
                update_dataset(
                    dataset.id,
                    status="failed",
                    rag_status="failed",
                    error_message=message,
                )

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        try:
            return str(UUID(session_id))
        except ValueError as error:
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            ) from error

    @staticmethod
    def safe_error(error: Exception) -> str:
        text = str(error).strip()
        return (text or "Processing failed.")[:300]

    @staticmethod
    def _legacy_session(dataset: DatasetRecord) -> AnalysisSessionRecord:
        return AnalysisSessionRecord(
            id=dataset.session_id or dataset.id,
            user_id=dataset.user_id,
            description=dataset.description,
            status=dataset.status,
            rag_status=dataset.rag_status,
            error_message=dataset.error_message,
        )

    @staticmethod
    def _not_found(session_id: str) -> SessionNotFoundError:
        return SessionNotFoundError(
            f"Analysis session '{session_id}' was not found."
        )
