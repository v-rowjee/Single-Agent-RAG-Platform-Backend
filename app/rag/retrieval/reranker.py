from __future__ import annotations

import logging
import threading
from typing import Any

from app.core.config import get_rag_config
from app.rag.models import RerankedDocument, RetrievedDocument

logger = logging.getLogger(__name__)
_RERANKING_POLICY = get_rag_config().reranking


class SentenceTransformerReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _RERANKING_POLICY.model
        self._model: Any | None = None
        self._lock = threading.Lock()

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        limit: int = _RERANKING_POLICY.limit,
    ) -> list[RerankedDocument]:
        if not documents:
            return []
        try:
            model = self._model_instance()

            pairs = [(query, document.page_content) for document in documents]

            raw_scores = model.predict(
                pairs,
                batch_size=_RERANKING_POLICY.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            scores = self._to_float_scores(raw_scores)

            if len(scores) != len(documents):
                raise ValueError("Document and reranker score counts do not match.")

            ranked = [
                RerankedDocument(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    score=document.score,
                    reranker_score=reranker_score,
                )
                for document, reranker_score in zip(documents, scores)
            ]

            ranked.sort(
                key=lambda item: (
                    item.reranker_score
                    if item.reranker_score is not None
                    else item.score
                ),
                reverse=True,
            )

            logger.info(
                "RAG reranking model=%s candidates=%s kept=%s",
                self.model_name,
                len(documents),
                min(limit, len(ranked)),
            )

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

    def _model_instance(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    from torch import nn

                    logger.info(
                        "Loading reranker model=%s",
                        self.model_name,
                    )

                    self._model = CrossEncoder(
                        self.model_name,
                        max_length=_RERANKING_POLICY.max_length,
                        activation_fn=nn.Sigmoid(),
                    )

        return self._model

    @staticmethod
    def _to_float_scores(raw_scores: Any) -> list[float]:
        values = (
            raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        )

        return [
            float(value[0] if isinstance(value, list) else value) for value in values
        ]


_reranker: SentenceTransformerReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> SentenceTransformerReranker:
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = SentenceTransformerReranker()

    return _reranker
