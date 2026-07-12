from __future__ import annotations

import logging
import threading
from typing import Any

from app.rag.config import RERANK_LIMIT, RERANKER_MODEL_NAME
from app.rag.models import RerankedDocument, RetrievedDocument


logger = logging.getLogger(__name__)


class FastEmbedReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        limit: int = RERANK_LIMIT,
    ) -> list[RerankedDocument]:
        if not documents:
            return []
        try:
            model = self._model_instance()
            contents = [document.page_content for document in documents]
            raw_results = list(model.rerank(query=query, documents=contents))
            ranked = self._merge_results(documents, raw_results)
            ranked.sort(
                key=lambda item: (
                    item.reranker_score if item.reranker_score is not None else item.score
                ),
                reverse=True,
            )
            logger.info("RAG reranking candidates=%s kept=%s", len(documents), min(limit, len(ranked)))
            return ranked[:limit]
        except Exception:
            logger.exception("RAG reranking failed; using vector-search order.")
            return [
                RerankedDocument(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    score=document.score,
                    reranker_score=None,
                )
                for document in documents[:limit]
            ]

    def _model_instance(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    @staticmethod
    def _merge_results(
        documents: list[RetrievedDocument],
        raw_results: list[Any],
    ) -> list[RerankedDocument]:
        output: list[RerankedDocument] = []
        used: set[int] = set()
        for position, result in enumerate(raw_results):
            if isinstance(result, int | float):
                index = position
                score = float(result)
            else:
                index = getattr(result, "index", None)
                score = getattr(result, "score", None)
            if index is None and isinstance(result, dict):
                index = result.get("index")
            if not isinstance(index, int):
                index = position
            if index < 0 or index >= len(documents):
                continue
            used.add(index)
            if score is None and isinstance(result, dict):
                score = result.get("score")
            document = documents[index]
            output.append(
                RerankedDocument(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    score=document.score,
                    reranker_score=float(score) if score is not None else None,
                )
            )
        for index, document in enumerate(documents):
            if index not in used:
                output.append(
                    RerankedDocument(
                        page_content=document.page_content,
                        metadata=document.metadata,
                        score=document.score,
                        reranker_score=None,
                    )
                )
        return output


_reranker: FastEmbedReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> FastEmbedReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = FastEmbedReranker()
    return _reranker
