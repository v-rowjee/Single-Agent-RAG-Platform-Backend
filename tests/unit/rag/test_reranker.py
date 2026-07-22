from __future__ import annotations

import sys
from types import SimpleNamespace

from app.rag.models import RetrievedDocument
from app.rag.retrieval.reranker import SentenceTransformerReranker


class FakeScores:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeCrossEncoder:
    instances: list["FakeCrossEncoder"] = []
    scores = [0.2, 0.9, -0.4]
    error: Exception | None = None

    def __init__(self, model_name: str, **options: object) -> None:
        self.model_name = model_name
        self.options = options
        self.calls: list[
            tuple[list[tuple[str, str]], dict[str, object]]
        ] = []
        self.instances.append(self)

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **options: object,
    ) -> FakeScores:
        self.calls.append((pairs, options))
        if self.error is not None:
            raise self.error
        return FakeScores(self.scores[: len(pairs)])


def _install_fake_cross_encoder(monkeypatch) -> None:
    FakeCrossEncoder.instances.clear()
    FakeCrossEncoder.error = None
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )


def _documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            page_content="Revenue is 120.",
            metadata={"source_id": "revenue"},
            score=0.8,
        ),
        RetrievedDocument(
            page_content="Profit is 30.",
            metadata={"source_id": "profit"},
            score=0.7,
        ),
        RetrievedDocument(
            page_content="The office is in London.",
            metadata={"source_id": "office"},
            score=0.6,
        ),
    ]


def test_bge_reranker_lazily_scores_and_orders_documents(monkeypatch) -> None:
    _install_fake_cross_encoder(monkeypatch)
    reranker = SentenceTransformerReranker()

    ranked = reranker.rerank("What is profit?", _documents(), limit=2)

    assert len(FakeCrossEncoder.instances) == 1
    model = FakeCrossEncoder.instances[0]
    assert model.model_name == "BAAI/bge-reranker-v2-m3"
    assert model.options["max_length"] == 384
    assert "activation_fn" in model.options
    assert model.calls == [
        (
            [
                ("What is profit?", "Revenue is 120."),
                ("What is profit?", "Profit is 30."),
                ("What is profit?", "The office is in London."),
            ],
            {
                    "batch_size": 8,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        )
    ]
    assert [item.metadata["source_id"] for item in ranked] == [
        "profit",
        "revenue",
    ]
    assert [item.reranker_score for item in ranked] == [0.9, 0.2]


def test_bge_reranker_preserves_vector_order_on_failure(monkeypatch) -> None:
    _install_fake_cross_encoder(monkeypatch)
    FakeCrossEncoder.error = RuntimeError("reranker unavailable")
    reranker = SentenceTransformerReranker()

    ranked = reranker.rerank("What is revenue?", _documents(), limit=2)

    assert [item.metadata["source_id"] for item in ranked] == [
        "revenue",
        "profit",
    ]
    assert all(item.reranker_score is None for item in ranked)


def test_bge_reranker_does_not_load_model_without_documents(monkeypatch) -> None:
    _install_fake_cross_encoder(monkeypatch)
    reranker = SentenceTransformerReranker()

    assert reranker.rerank("What is revenue?", []) == []
    assert FakeCrossEncoder.instances == []
