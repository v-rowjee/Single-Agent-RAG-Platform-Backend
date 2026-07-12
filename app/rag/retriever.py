from __future__ import annotations

import logging

from app.rag.config import MIN_RETRIEVAL_SCORE, VECTOR_SEARCH_LIMIT
from app.rag.embedding_service import get_embedding_service
from app.rag.models import RetrievedDocument
from app.rag.vector_store import RagVectorStore


logger = logging.getLogger(__name__)


class RagRetriever:
    def __init__(self, vector_store: RagVectorStore) -> None:
        self.vector_store = vector_store
        self.embedding_service = get_embedding_service()

    def retrieve(
        self,
        query: str,
        limit: int = VECTOR_SEARCH_LIMIT,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ) -> list[RetrievedDocument]:
        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            min_score=min_score,
        )
        deduped = self._dedupe(results)
        logger.info(
            "RAG retrieval session_id=%s query=%r candidates=%s",
            self.vector_store.session_id,
            query[:120],
            len(deduped),
        )
        return deduped

    @staticmethod
    def _dedupe(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        output: list[RetrievedDocument] = []
        seen: set[str] = set()
        for document in documents:
            key = document.page_content.strip()
            if key in seen:
                continue
            seen.add(key)
            output.append(document)
        return output
