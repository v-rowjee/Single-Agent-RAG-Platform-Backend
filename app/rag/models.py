from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DocumentType = Literal[
    "dataset_overview",
    "column_profile",
    "numeric_summary",
    "category_analysis",
    "timeline_analysis",
    "row_batch",
]

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
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    page_content: str
    metadata: dict[str, str | int | float | bool]
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
