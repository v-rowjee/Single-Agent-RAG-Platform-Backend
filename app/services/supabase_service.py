from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

DatasetStatus = Literal["processing", "ready", "failed"]
RagStatus = Literal["pending", "indexing", "ready", "failed"]
DashboardStatus = Literal["success", "partial", "failed"]
MessageRole = Literal["user", "assistant"]
JsonDict = dict[str, object]


class SupabaseUnavailableError(Exception):
    """Raised when the backend cannot use Supabase."""


@dataclass(frozen=True)
class AnalysisSessionRecord:
    id: str
    user_id: str
    description: str | None
    status: str
    rag_status: str
    error_message: str | None
    requires_reset: bool = False
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    user_id: str
    file_name: str
    storage_path: str
    mime_type: str
    file_size: int
    file_hash: str
    description: str | None
    status: str
    rag_status: str
    error_message: str | None
    session_id: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class DashboardRecord:
    id: str
    dataset_id: str
    status: str
    response: JsonDict
    generated_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class MessageRecord:
    id: str
    dataset_id: str
    role: str
    content: str
    sources: list[str]
    created_at: str


@dataclass(frozen=True)
class SessionProcessingRecord:
    dataset_id: str
    workflow_status: str
    generic_cleaning_report: JsonDict
    prepared_dataset: JsonDict
    created_at: str | None = None
    updated_at: str | None = None


class SupabaseService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: object | None = None

    @property
    def client(self):
        if self._client is None:
            if (
                not self.settings.supabase_url
                or not self.settings.supabase_service_role_key
            ):
                raise SupabaseUnavailableError(
                    "Supabase credentials are not configured."
                )

            from supabase import create_client

            self._client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_role_key,
            )
        return self._client

    def upload_file(
        self,
        storage_path: str,
        content: bytes,
        mime_type: str,
    ) -> None:
        bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
        file_options = {
            "content-type": mime_type,
            "upsert": "false",
        }
        try:
            bucket.upload(storage_path, content, file_options=file_options)
        except TypeError:
            bucket.upload(storage_path, content, file_options)

    def download_file(self, storage_path: str) -> bytes:
        bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
        payload = bucket.download(storage_path)
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        raise SupabaseUnavailableError("Storage download did not return bytes.")

    def delete_file(self, storage_path: str) -> None:
        bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
        bucket.remove([storage_path])

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
        session_id: str | None = None,
    ) -> DatasetRecord:
        payload: JsonDict = {
            "id": dataset_id,
            "user_id": user_id,
            "file_name": file_name,
            "storage_path": storage_path,
            "mime_type": mime_type,
            "file_size": file_size,
            "file_hash": file_hash,
            "description": description,
            "status": "processing",
            "rag_status": "pending",
            "error_message": None,
            "row_count": row_count,
            "column_count": column_count,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        table = self.client.table("datasets")
        try:
            response = table.insert(payload).execute()
        except Exception as error:
            if not self._is_missing_dataset_metadata_column(error):
                raise

            # Existing deployments may not yet have run the single-active-
            # dataset migration. The API can still serve them safely by
            # calculating these values from the uploaded file when needed.
            # Retain the fields in the primary insert so migrated databases
            # avoid that additional read.
            logger.warning(
                "datasets table is missing row_count/column_count; falling back "
                "to legacy storage. Apply scripts/migrate_single_active_dataset.sql."
            )
            legacy_payload = dict(payload)
            legacy_payload.pop("row_count", None)
            legacy_payload.pop("column_count", None)
            response = table.insert(legacy_payload).execute()
        rows = list(response.data or [])
        if not rows:
            raise SupabaseUnavailableError("Dataset insert returned no row.")
        return self._dataset(rows[0])

    def create_session(
        self,
        session_id: str,
        user_id: str,
        description: str | None,
        *,
        requires_reset: bool = False,
    ) -> AnalysisSessionRecord:
        response = (
            self.client.table("analysis_sessions")
            .insert(
                {
                    "id": session_id,
                    "user_id": user_id,
                    "description": description,
                    "status": "processing",
                    "rag_status": "pending",
                    "error_message": None,
                    "requires_reset": requires_reset,
                }
            )
            .execute()
        )
        rows = list(response.data or [])
        if not rows:
            raise SupabaseUnavailableError("Analysis-session insert returned no row.")
        return self._session(rows[0])

    def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> AnalysisSessionRecord | None:
        response = (
            self.client.table("analysis_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._session(rows[0]) if rows else None

    def get_active_session(self, user_id: str) -> AnalysisSessionRecord | None:
        response = (
            self.client.table("analysis_sessions")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._session(rows[0]) if rows else None

    def get_session_datasets(
        self,
        session_id: str,
        user_id: str,
    ) -> list[DatasetRecord]:
        response = (
            self.client.table("datasets")
            .select("*")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )
        return [self._dataset(row) for row in list(response.data or [])]

    def delete_session(self, session_id: str, user_id: str) -> None:
        self.client.table("analysis_sessions").delete().eq("id", session_id).eq(
            "user_id",
            user_id,
        ).execute()

    def update_session_status(
        self,
        session_id: str,
        status: DatasetStatus | None = None,
        rag_status: RagStatus | None = None,
        error_message: str | None = None,
    ) -> None:
        payload: JsonDict = {}
        if status is not None:
            payload["status"] = status
        if rag_status is not None:
            payload["rag_status"] = rag_status
        if error_message is not None or status != "failed":
            payload["error_message"] = error_message
        if payload:
            self.client.table("analysis_sessions").update(payload).eq(
                "id",
                session_id,
            ).execute()

    def get_dataset(self, dataset_id: str, user_id: str) -> DatasetRecord | None:
        response = (
            self.client.table("datasets")
            .select("*")
            .eq("id", dataset_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._dataset(rows[0]) if rows else None

    def get_active_dataset(self, user_id: str) -> DatasetRecord | None:
        session = self.get_active_session(user_id)
        if session is not None:
            datasets = self.get_session_datasets(session.id, user_id)
            return datasets[0] if datasets else None
        response = (
            self.client.table("datasets")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._dataset(rows[0]) if rows else None

    def delete_dataset(self, dataset_id: str, user_id: str) -> None:
        self.client.table("datasets").delete().eq("id", dataset_id).eq(
            "user_id",
            user_id,
        ).execute()

    def update_dataset_status(
        self,
        dataset_id: str,
        status: DatasetStatus | None = None,
        rag_status: RagStatus | None = None,
        error_message: str | None = None,
    ) -> None:
        payload: JsonDict = {}
        if status is not None:
            payload["status"] = status
        if rag_status is not None:
            payload["rag_status"] = rag_status
        if error_message is not None or status != "failed":
            payload["error_message"] = error_message
        if not payload:
            return
        self.client.table("datasets").update(payload).eq("id", dataset_id).execute()

    def save_dashboard(
        self,
        dataset_id: str,
        status: DashboardStatus,
        response: JsonDict,
    ) -> DashboardRecord:
        payload: JsonDict = {
            "session_id": dataset_id,
            "status": status,
            "response": response,
            "generated_at": self._now(),
        }
        result = (
            self.client.table("dashboards")
            .upsert(payload, on_conflict="session_id")
            .execute()
        )
        rows = list(result.data or [])
        if not rows:
            fetched = self.get_dashboard(dataset_id)
            if fetched is None:
                raise SupabaseUnavailableError("Dashboard upsert returned no row.")
            return fetched
        return self._dashboard(rows[0])

    def get_dashboard(self, dataset_id: str) -> DashboardRecord | None:
        response = (
            self.client.table("dashboards")
            .select("*")
            .eq("session_id", dataset_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._dashboard(rows[0]) if rows else None

    def save_message(
        self,
        dataset_id: str,
        role: MessageRole,
        content: str,
        sources: list[str] | None = None,
    ) -> MessageRecord:
        payload: JsonDict = {
            "session_id": dataset_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        }
        response = self.client.table("messages").insert(payload).execute()
        rows = list(response.data or [])
        if not rows:
            raise SupabaseUnavailableError("Message insert returned no row.")
        return self._message(rows[0])

    def get_recent_messages(
        self,
        dataset_id: str,
        limit: int = 6,
    ) -> list[MessageRecord]:
        response = (
            self.client.table("messages")
            .select("*")
            .eq("session_id", dataset_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(response.data or [])
        return [self._message(row) for row in reversed(rows)]

    def save_session_processing(
        self,
        dataset_id: str,
        workflow_status: str,
        generic_cleaning_report: JsonDict,
        prepared_dataset: JsonDict,
    ) -> SessionProcessingRecord:
        payload: JsonDict = {
            "session_id": dataset_id,
            "workflow_status": workflow_status,
            "generic_cleaning_report": generic_cleaning_report,
            "prepared_dataset": prepared_dataset,
        }
        result = (
            self.client.table("session_processing")
            .upsert(payload, on_conflict="session_id")
            .execute()
        )
        rows = list(result.data or [])
        if not rows:
            fetched = self.get_session_processing(dataset_id)
            if fetched is None:
                raise SupabaseUnavailableError(
                    "Session-processing upsert returned no row."
                )
            return fetched
        return self._session_processing(rows[0])

    def get_session_processing(
        self,
        dataset_id: str,
    ) -> SessionProcessingRecord | None:
        response = (
            self.client.table("session_processing")
            .select("*")
            .eq("session_id", dataset_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._session_processing(rows[0]) if rows else None

    def delete_document_chunks(self, dataset_id: str) -> None:
        self.client.table("document_chunks").delete().eq(
            "dataset_id",
            dataset_id,
        ).execute()

    def insert_document_chunks(
        self,
        chunks: list[JsonDict],
        batch_size: int = 50,
    ) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if batch:
                self.client.table("document_chunks").insert(batch).execute()

    def replace_document_chunks(
        self,
        dataset_id: str,
        chunks: list[JsonDict],
    ) -> int:
        """Replace a dataset index in one PostgreSQL transaction."""
        try:
            response = self.client.rpc(
                "replace_document_chunks",
                {
                    "p_dataset_id": dataset_id,
                    "p_chunks": chunks,
                },
            ).execute()
        except Exception as exc:
            if not self._is_missing_atomic_replace_rpc(exc):
                raise
            logger.warning(
                "Atomic RAG replacement RPC is not installed; using the "
                "non-destructive compatibility path. Apply "
                "scripts/migrate_atomic_rag_index.sql."
            )
            return self._replace_document_chunks_compat(dataset_id, chunks)

        value = response.data
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, dict):
            value = value.get("replace_document_chunks")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SupabaseUnavailableError(
                "Atomic document replacement returned an invalid row count."
            ) from exc

    def replace_session_document_chunks(
        self,
        session_id: str,
        chunks: list[JsonDict],
    ) -> int:
        response = self.client.rpc(
            "replace_session_document_chunks",
            {
                "p_session_id": session_id,
                "p_chunks": chunks,
            },
        ).execute()
        value = response.data
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, dict):
            value = value.get("replace_session_document_chunks")
        return int(value)

    def _replace_document_chunks_compat(
        self,
        dataset_id: str,
        chunks: list[JsonDict],
    ) -> int:
        """Upsert first and remove stale rows last so failures retain an index."""
        response = (
            self.client.table("document_chunks")
            .select("id,source_id,chunk_index")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        existing = list(response.data or [])
        current_keys = {
            (str(chunk["source_id"]), int(chunk.get("chunk_index") or 0))
            for chunk in chunks
        }

        table = self.client.table("document_chunks")
        for start in range(0, len(chunks), 50):
            batch = chunks[start : start + 50]
            if batch:
                table.upsert(
                    batch,
                    on_conflict="dataset_id,source_id,chunk_index",
                ).execute()

        stale_ids = [
            str(row["id"])
            for row in existing
            if (
                str(row.get("source_id") or ""),
                int(row.get("chunk_index") or 0),
            )
            not in current_keys
        ]
        for start in range(0, len(stale_ids), 100):
            batch = stale_ids[start : start + 100]
            if batch:
                self.client.table("document_chunks").delete().in_(
                    "id",
                    batch,
                ).execute()
        return len(chunks)

    @staticmethod
    def _is_missing_atomic_replace_rpc(error: Exception) -> bool:
        text = str(error).casefold()
        return (
            "pgrst202" in text
            or (
                "replace_document_chunks" in text
                and any(term in text for term in ("not find", "not found", "schema cache"))
            )
        )

    def match_document_chunks(
        self,
        dataset_id: str,
        query_embedding: list[float],
        match_count: int = 12,
        match_threshold: float = 0.2,
    ) -> list[JsonDict]:
        response = self.client.rpc(
            "match_document_chunks",
            {
                "p_dataset_id": dataset_id,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_match_threshold": match_threshold,
            },
        ).execute()
        return [dict(row) for row in list(response.data or [])]

    def match_session_document_chunks(
        self,
        session_id: str,
        query_embedding: list[float],
        match_count: int = 12,
        match_threshold: float = 0.2,
    ) -> list[JsonDict]:
        response = self.client.rpc(
            "match_session_document_chunks",
            {
                "p_session_id": session_id,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_match_threshold": match_threshold,
            },
        ).execute()
        return [dict(row) for row in list(response.data or [])]

    @staticmethod
    def _dataset(row: JsonDict) -> DatasetRecord:
        return DatasetRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            file_name=str(row["file_name"]),
            storage_path=str(row["storage_path"]),
            mime_type=str(row["mime_type"]),
            file_size=int(row["file_size"]),
            file_hash=str(row["file_hash"]),
            description=(
                str(row["description"]) if row.get("description") is not None else None
            ),
            status=str(row["status"]),
            rag_status=str(row["rag_status"]),
            error_message=(
                str(row["error_message"])
                if row.get("error_message") is not None
                else None
            ),
            session_id=(
                str(row["session_id"])
                if row.get("session_id") is not None
                else None
            ),
            row_count=(
                int(row["row_count"])
                if row.get("row_count") is not None
                else None
            ),
            column_count=(
                int(row["column_count"])
                if row.get("column_count") is not None
                else None
            ),
            created_at=(
                str(row["created_at"]) if row.get("created_at") is not None else None
            ),
            updated_at=(
                str(row["updated_at"]) if row.get("updated_at") is not None else None
            ),
        )

    @staticmethod
    def _session(row: JsonDict) -> AnalysisSessionRecord:
        return AnalysisSessionRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            description=(
                str(row["description"]) if row.get("description") is not None else None
            ),
            status=str(row["status"]),
            rag_status=str(row["rag_status"]),
            error_message=(
                str(row["error_message"])
                if row.get("error_message") is not None
                else None
            ),
            requires_reset=bool(row.get("requires_reset", False)),
            created_at=(
                str(row["created_at"]) if row.get("created_at") is not None else None
            ),
            updated_at=(
                str(row["updated_at"]) if row.get("updated_at") is not None else None
            ),
        )

    @staticmethod
    def _dashboard(row: JsonDict) -> DashboardRecord:
        response = row.get("response")
        return DashboardRecord(
            id=str(row["id"]),
            dataset_id=str(row.get("session_id") or row.get("dataset_id")),
            status=str(row["status"]),
            response=dict(response) if isinstance(response, dict) else {},
            generated_at=(
                str(row["generated_at"])
                if row.get("generated_at") is not None
                else None
            ),
            updated_at=(
                str(row["updated_at"]) if row.get("updated_at") is not None else None
            ),
        )

    @staticmethod
    def _message(row: JsonDict) -> MessageRecord:
        sources = row.get("sources")
        return MessageRecord(
            id=str(row["id"]),
            dataset_id=str(row.get("session_id") or row.get("dataset_id")),
            role=str(row["role"]),
            content=str(row["content"]),
            sources=[
                str(source)
                for source in sources
                if isinstance(source, str | int | float)
            ]
            if isinstance(sources, list)
            else [],
            created_at=str(row.get("created_at") or SupabaseService._now()),
        )

    @staticmethod
    def _session_processing(row: JsonDict) -> SessionProcessingRecord:
        generic_cleaning_report = row.get("generic_cleaning_report")
        prepared_dataset = row.get("prepared_dataset")
        return SessionProcessingRecord(
            dataset_id=str(row.get("session_id") or row.get("dataset_id")),
            workflow_status=str(row["workflow_status"]),
            generic_cleaning_report=(
                dict(generic_cleaning_report)
                if isinstance(generic_cleaning_report, dict)
                else {}
            ),
            prepared_dataset=(
                dict(prepared_dataset)
                if isinstance(prepared_dataset, dict)
                else {}
            ),
            created_at=(
                str(row["created_at"])
                if row.get("created_at") is not None
                else None
            ),
            updated_at=(
                str(row["updated_at"])
                if row.get("updated_at") is not None
                else None
            ),
        )

    @staticmethod
    def _is_missing_dataset_metadata_column(error: Exception) -> bool:
        message = str(error).lower()
        mentions_metadata_column = (
            "row_count" in message or "column_count" in message
        )
        return mentions_metadata_column and (
            "does not exist" in message
            or "could not find" in message
            or "schema cache" in message
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


supabase_service = SupabaseService()
