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
class DatasetRecord:
    id: str
    file_name: str
    storage_path: str
    mime_type: str
    file_size: int
    file_hash: str
    description: str | None
    status: str
    rag_status: str
    error_message: str | None
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
        file_name: str,
        storage_path: str,
        mime_type: str,
        file_size: int,
        file_hash: str,
        description: str | None,
    ) -> DatasetRecord:
        payload: JsonDict = {
            "id": dataset_id,
            "file_name": file_name,
            "storage_path": storage_path,
            "mime_type": mime_type,
            "file_size": file_size,
            "file_hash": file_hash,
            "description": description,
            "status": "processing",
            "rag_status": "pending",
            "error_message": None,
        }
        response = self.client.table("datasets").insert(payload).execute()
        rows = list(response.data or [])
        if not rows:
            raise SupabaseUnavailableError("Dataset insert returned no row.")
        return self._dataset(rows[0])

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        response = (
            self.client.table("datasets")
            .select("*")
            .eq("id", dataset_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return self._dataset(rows[0]) if rows else None

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
            "dataset_id": dataset_id,
            "status": status,
            "response": response,
            "generated_at": self._now(),
        }
        result = (
            self.client.table("dashboards")
            .upsert(payload, on_conflict="dataset_id")
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
            .eq("dataset_id", dataset_id)
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
            "dataset_id": dataset_id,
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
            .eq("dataset_id", dataset_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(response.data or [])
        return [self._message(row) for row in reversed(rows)]

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

    @staticmethod
    def _dataset(row: JsonDict) -> DatasetRecord:
        return DatasetRecord(
            id=str(row["id"]),
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
            dataset_id=str(row["dataset_id"]),
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
            dataset_id=str(row["dataset_id"]),
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
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


supabase_service = SupabaseService()
