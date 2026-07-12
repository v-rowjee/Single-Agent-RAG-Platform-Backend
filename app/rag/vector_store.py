from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from app.rag.config import RAG_STORAGE_ROOT
from app.rag.models import RagDocument, RetrievedDocument


logger = logging.getLogger(__name__)


class RagVectorStore:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.safe_session_id = self._safe_session_id(session_id)
        self.collection_name = f"rag_{self.safe_session_id[:48]}"
        self.storage_path = self._session_path(self.safe_session_id)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._client: Any | None = None

    def index_documents(
        self,
        documents: list[RagDocument],
        embeddings: list[list[float]],
        file_signature: dict[str, str | int | float],
    ) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document list.")
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts do not match.")

        vector_size = len(embeddings[0])
        if vector_size <= 0:
            raise ValueError("Embedding vectors are empty.")

        self._recreate_collection(vector_size)

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, self._point_key(document, index))),
                vector=embeddings[index],
                payload={
                    "page_content": document.page_content,
                    **self._payload_metadata(document.metadata),
                },
            )
            for index, document in enumerate(documents)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        self._write_index_meta(
            {
                "session_id": self.session_id,
                "collection_name": self.collection_name,
                "document_count": len(documents),
                "vector_size": vector_size,
                "file_signature": file_signature,
            }
        )
        logger.info(
            "RAG index created session_id=%s collection=%s chunks=%s vector_size=%s",
            self.session_id,
            self.collection_name,
            len(documents),
            vector_size,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int,
        min_score: float = 0.0,
    ) -> list[RetrievedDocument]:
        if not query_vector or not self.index_exists():
            return []

        hits = self._search_points(query_vector=query_vector, limit=limit)
        output: list[RetrievedDocument] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", {}) or {})
            if payload.get("session_id") != self.session_id:
                continue
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if score < min_score:
                continue
            content = str(payload.pop("page_content", "")).strip()
            if not content:
                continue
            output.append(
                RetrievedDocument(
                    page_content=content,
                    metadata=self._clean_loaded_metadata(payload),
                    score=score,
                )
            )
        return output

    def index_exists(self) -> bool:
        if not self._collection_exists():
            return False
        try:
            count = self.client.count(
                collection_name=self.collection_name,
                exact=False,
            )
            return int(getattr(count, "count", 0)) > 0
        except Exception:
            return self.index_meta_path.exists()

    def delete_session_index(self) -> None:
        if self._collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
        self.index_meta_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug(
                    "Failed to close Qdrant client session_id=%s",
                    self.session_id,
                    exc_info=True,
                )
            finally:
                self._client = None

    def read_index_meta(self) -> dict[str, Any] | None:
        if not self.index_meta_path.exists():
            return None
        try:
            return json.loads(self.index_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    @property
    def index_meta_path(self) -> Path:
        return self.storage_path / "index_meta.json"

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=str(self.storage_path))
        return self._client

    def _recreate_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            collections = self.client.get_collections().collections
            return any(item.name == self.collection_name for item in collections)
        except Exception:
            return False

    def _search_points(self, query_vector: list[float], limit: int) -> list[Any]:
        try:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            )
        except AttributeError:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(response, "points", []))

    def _write_index_meta(self, payload: dict[str, Any]) -> None:
        self.index_meta_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("_")
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe[:80]}_{digest}" if safe else digest

    @staticmethod
    def _session_path(safe_session_id: str) -> Path:
        root = RAG_STORAGE_ROOT.resolve()
        path = (RAG_STORAGE_ROOT / safe_session_id).resolve()
        if root != path and root not in path.parents:
            raise ValueError("Invalid RAG storage path.")
        return path

    @staticmethod
    def _payload_metadata(
        metadata: dict[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and isinstance(value, str | int | float | bool)
        }

    @staticmethod
    def _clean_loaded_metadata(payload: dict[str, Any]) -> dict[str, str | int | float | bool]:
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, str | int | float | bool)
        }

    @staticmethod
    def _point_key(document: RagDocument, index: int) -> str:
        source_id = str(document.metadata.get("source_id", f"document_{index}"))
        chunk_index = str(document.metadata.get("chunk_index", "0"))
        session_id = str(document.metadata.get("session_id", "session"))
        return f"{session_id}:{source_id}:{chunk_index}:{index}"
