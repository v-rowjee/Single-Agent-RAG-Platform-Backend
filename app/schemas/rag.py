from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RagDocumentType = Literal[
    "dataset_overview",
    "column_profile",
    "numeric_summary",
    "category_analysis",
    "timeline_analysis",
    "row_batch",
]


class RagSourceMetadata(BaseModel):
    session_id: str
    file_name: str
    document_type: RagDocumentType
    source_id: str
    measure: str | None = None
    dimension: str | None = None
    date_field: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    chunk_index: int | None = None


class RagSource(BaseModel):
    page_content: str = Field(min_length=1)
    metadata: RagSourceMetadata
    score: float | None = None
    reranker_score: float | None = None
