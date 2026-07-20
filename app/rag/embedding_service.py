from __future__ import annotations

import threading
from collections.abc import Iterable

from app.core.config import get_rag_config


_EMBEDDING_POLICY = get_rag_config().embedding
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
}


class SentenceTransformerEmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _EMBEDDING_POLICY.model
        self._model: object | None = None
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        prefix = _QUERY_PREFIXES.get(self.model_name, "")
        vectors = self._encode([f"{prefix}{text}"])
        return vectors[0] if vectors else []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encode_options: dict[str, object] = {
            "batch_size": _EMBEDDING_POLICY.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        return self._normalise_vectors(
            self._model_instance().encode(texts, **encode_options)
        )

    def _model_instance(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalise_vectors(vectors: Iterable[object]) -> list[list[float]]:
        output: list[list[float]] = []
        for vector in vectors:
            if hasattr(vector, "tolist"):
                values = vector.tolist()
            else:
                values = list(vector)  # type: ignore[arg-type]
            output.append([float(value) for value in values])
        return output


_embedding_service: SentenceTransformerEmbeddingService | None = None
_embedding_service_lock = threading.Lock()


def get_embedding_service() -> SentenceTransformerEmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        with _embedding_service_lock:
            if _embedding_service is None:
                _embedding_service = SentenceTransformerEmbeddingService()
    return _embedding_service
