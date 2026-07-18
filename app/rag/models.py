from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QueryType = Literal[
    "calculation",
    "forecast",
    "lookup",
    "comparison",
    "explanation",
    "mixed",
]


@dataclass(slots=True)
class RagDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    page_content: str
    metadata: dict[str, Any]
    score: float


@dataclass(slots=True)
class RerankedDocument(RetrievedDocument):
    reranker_score: float | None = None


@dataclass(slots=True)
class CalculatedEvidence:
    text: str
    direct_answer: str | None = None


@dataclass(slots=True)
class IndexStatus:
    session_id: str
    collection_name: str
    document_count: int
    chunk_count: int
    vector_size: int
