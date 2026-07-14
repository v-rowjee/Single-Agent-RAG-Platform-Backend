"""Deterministic indexing for retrieval documents produced by the BI graph."""
from __future__ import annotations

import logging
from typing import Any

from app.rag.embedding_service import get_embedding_service
from app.services.supabase_service import SupabaseService, supabase_service


logger = logging.getLogger(__name__)


class RetrievalIndexingService:
    def __init__(self, storage: SupabaseService | None = None) -> None:
        self.storage = storage or supabase_service

    def index_documents(
        self,
        session_id: str,
        dataset_id: str,
        retrieval_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        document_count = len(retrieval_documents)
        try:
            if not session_id or not dataset_id:
                raise ValueError("session_id and dataset_id are required.")
            if any(
                not isinstance(document, dict)
                or not str(document.get("id") or "").strip()
                or not str(document.get("content") or "").strip()
                for document in retrieval_documents
            ):
                raise ValueError("Retrieval documents require id and content.")

            self.storage.update_dataset_status(dataset_id, rag_status="indexing")
            logger.info(
                "Retrieval replacement started session_id=%s document_count=%s",
                session_id,
                document_count,
            )
            # document_chunks is scoped by dataset_id, which is the persisted
            # session identifier in this service.  This is an exact session
            # delete; it never clears the collection.
            self.storage.delete_document_chunks(dataset_id)
            logger.info(
                "Existing session documents deleted session_id=%s",
                session_id,
            )

            if not retrieval_documents:
                self.storage.update_dataset_status(dataset_id, rag_status="ready")
                return {
                    "status": "success",
                    "document_count": 0,
                    "indexed_count": 0,
                    "failed_count": 0,
                }

            embeddings = get_embedding_service().embed_documents(
                [str(document["content"]) for document in retrieval_documents]
            )
            if len(embeddings) != document_count:
                raise ValueError("Document and embedding counts do not match.")

            rows: list[dict[str, object]] = []
            for index, (document, embedding) in enumerate(
                zip(retrieval_documents, embeddings)
            ):
                metadata = dict(document.get("metadata") or {})
                metadata.update(
                    {
                        "session_id": session_id,
                        "dataset_id": dataset_id,
                        "source_ids": list(document.get("source_ids") or []),
                        "title": str(document.get("title") or document["id"]),
                    }
                )
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "source_id": str(document["id"]),
                        "document_type": str(document.get("document_type") or "bi"),
                        "chunk_index": index,
                        "content": str(document["content"]),
                        "metadata": metadata,
                        "embedding": [float(value) for value in embedding],
                    }
                )

            self.storage.insert_document_chunks(rows, batch_size=50)
            self.storage.update_dataset_status(dataset_id, rag_status="ready")
            logger.info(
                "New documents indexed session_id=%s indexed_count=%s",
                session_id,
                len(rows),
            )
            return {
                "status": "success",
                "document_count": document_count,
                "indexed_count": len(rows),
                "failed_count": 0,
            }
        except Exception:
            logger.exception(
                "Retrieval replacement failed session_id=%s document_count=%s",
                session_id,
                document_count,
            )
            try:
                if dataset_id:
                    self.storage.update_dataset_status(dataset_id, rag_status="failed")
            except Exception:
                logger.exception(
                    "Failed to mark retrieval indexing failed session_id=%s",
                    session_id,
                )
            return {
                "status": "failed",
                "document_count": document_count,
                "indexed_count": 0,
                "failed_count": document_count,
                "message": "Retrieval indexing failed.",
            }


retrieval_indexing_service = RetrievalIndexingService()
