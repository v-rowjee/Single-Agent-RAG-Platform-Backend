"""Supabase-backed vector store for transactional indexing and retrieval."""

from __future__ import annotations

import logging
from typing import Protocol

from app.services.persistence.supabase import SupabaseGateway, SupabaseUnavailableError, supabase_gateway

logger = logging.getLogger(__name__)
JsonDict = dict[str, object]

class VectorStore(Protocol):
    def replace_session_document_chunks(self, session_id: str, chunks: list[dict]) -> None: ...
    def match_session_document_chunks(self, session_id: str, query_embedding: list[float], match_count: int, match_threshold: float) -> list[dict]: ...

class SupabaseVectorStore:
    def __init__(self, gateway: SupabaseGateway | None = None) -> None:
        self.gateway = gateway or supabase_gateway

    @property
    def client(self):
        return self.gateway.client

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
                    "non-destructive compatibility path."
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

vector_store: VectorStore = SupabaseVectorStore()
