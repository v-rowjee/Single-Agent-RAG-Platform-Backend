from __future__ import annotations

import sys
from types import SimpleNamespace

from app.rag.embedding_service import SentenceTransformerEmbeddingService


class FakeVector:
    def __init__(self, value: int) -> None:
        self.value = value

    def tolist(self) -> list[int]:
        return [self.value] * 384


class FakeSentenceTransformer:
    instances: list["FakeSentenceTransformer"] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.instances.append(self)

    def encode(self, texts: list[str], **options: object) -> list[FakeVector]:
        self.calls.append((texts, options))
        return [FakeVector(index + 1) for index, _ in enumerate(texts)]


def _install_fake_sentence_transformers(monkeypatch) -> None:
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )


def test_embedding_service_lazily_encodes_documents_and_queries(monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    service = SentenceTransformerEmbeddingService(
        model_name="BAAI/bge-small-en-v1.5"
    )

    document_vectors = service.embed_documents(["Revenue is 120.", "Profit is 30."])
    query_vector = service.embed_query("What is revenue?")

    assert len(FakeSentenceTransformer.instances) == 1
    model = FakeSentenceTransformer.instances[0]
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert model.calls == [
        (
            ["Revenue is 120.", "Profit is 30."],
            {
                "batch_size": 8,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        ),
        (
            [
                "Represent this sentence for searching relevant passages: "
                "What is revenue?"
            ],
            {
                "batch_size": 8,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        ),
    ]
    assert len(document_vectors) == 2
    assert len(document_vectors[0]) == 384
    assert document_vectors[0][0] == 1.0
    assert document_vectors[1][0] == 2.0
    assert len(query_vector) == 384
    assert query_vector[0] == 1.0


def test_embedding_service_does_not_load_model_for_empty_input(monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    service = SentenceTransformerEmbeddingService()

    assert service.embed_documents([]) == []
    assert service.embed_query("") == []
    assert FakeSentenceTransformer.instances == []
