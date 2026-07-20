from __future__ import annotations

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_rag_config
from app.rag.document_builder import DatasetDocumentBuilder
from app.rag.embedding_service import get_embedding_service
from app.rag.models import CalculatedEvidence, IndexStatus, QueryType, RagDocument, RerankedDocument, RetrievedDocument
from app.rag.reranker import get_reranker
from app.schemas.business_intelligence import BusinessIntelligenceAgentInput
from app.services.supabase_service import SupabaseService, supabase_service


logger = logging.getLogger(__name__)
_RAG_CONFIG = get_rag_config()

DERIVED_REVENUE_COLUMN = "__calculated_revenue__"
REVENUE_COLUMN_TERMS = (
    "revenue",
    "turnover",
    "sales amount",
    "sales_amount",
    "net sales",
    "net_sales",
    "sales value",
    "sales_value",
)


class RagService:
    def __init__(self, storage: SupabaseService | None = None) -> None:
        self._builder = DatasetDocumentBuilder()
        self._completed_signatures: dict[str, dict[str, str | int | float]] = {}
        self._index_locks: dict[str, threading.Lock] = {}
        self._service_lock = threading.Lock()
        self.storage = storage or supabase_service

    def index_dataset(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
        force: bool = False,
    ) -> IndexStatus:
        signature = self._file_signature(agent_input.filePath)

        if not force and self._completed_signatures.get(agent_input.sessionId) == signature:
            logger.info(
                "RAG index cache hit session_id=%s",
                agent_input.sessionId,
            )
            return IndexStatus(
                session_id=agent_input.sessionId,
                collection_name="document_chunks",
                document_count=0,
                chunk_count=0,
                vector_size=_RAG_CONFIG.embedding.dimensions,
            )

        lock = self._lock_for_session(agent_input.sessionId)
        with lock:
            if not force and self._completed_signatures.get(agent_input.sessionId) == signature:
                logger.info(
                    "RAG index cache hit session_id=%s",
                    agent_input.sessionId,
                )
                return IndexStatus(
                    session_id=agent_input.sessionId,
                    collection_name="document_chunks",
                    document_count=0,
                    chunk_count=0,
                    vector_size=_RAG_CONFIG.embedding.dimensions,
                )

            retrieval_documents = self.build_dataset_documents(
                agent_input=agent_input,
                profile=profile,
                workspace_session_id=agent_input.sessionId,
            )
            result = self.index_documents(
                session_id=agent_input.sessionId,
                dataset_id=agent_input.datasetId or agent_input.sessionId,
                retrieval_documents=retrieval_documents,
            )
            if result["status"] != "success":
                raise RuntimeError(str(result.get("message") or "RAG indexing failed."))

            self._completed_signatures[agent_input.sessionId] = signature
            return IndexStatus(
                session_id=agent_input.sessionId,
                collection_name="document_chunks",
                document_count=len(retrieval_documents),
                chunk_count=int(
                    result.get("chunk_count") or len(retrieval_documents)
                ),
                vector_size=int(result.get("vector_size") or 0),
            )

    def build_dataset_documents(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
        workspace_session_id: str,
    ) -> list[dict[str, Any]]:
        dataset_id = agent_input.datasetId or agent_input.sessionId
        documents = self._builder.build(
            df=load_dataframe(agent_input.filePath),
            profile=profile,
            session_id=workspace_session_id,
            file_name=agent_input.fileName,
        )
        output: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = dict(document.metadata)
            source_id = str(
                metadata.get("source_id") or f"document_{index}"
            )
            metadata.update(
                {
                    "session_id": workspace_session_id,
                    "dataset_id": dataset_id,
                    "file_name": agent_input.fileName,
                }
            )
            output.append(
                {
                    "id": f"{dataset_id}:{source_id}",
                    "content": document.page_content,
                    "document_type": str(
                        metadata.get("document_type") or "dataset_overview"
                    ),
                    "metadata": metadata,
                }
            )
        return output

    def index_documents(
        self,
        session_id: str,
        dataset_id: str,
        retrieval_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replace one dataset's retrieval index from either pipeline."""
        document_count = len(retrieval_documents)
        try:
            if not session_id or not dataset_id:
                raise ValueError("session_id and dataset_id are required.")
            if any(
                not isinstance(document, dict)
                or not str(document.get("id") or "").strip()
                or not str(document.get("content") or "").strip()
                for document in retrieval_documents
            ):
                raise ValueError("Retrieval documents require id and content.")

            chunked_documents = self._chunk_retrieval_documents(retrieval_documents)
            chunk_count = len(chunked_documents)
            logger.info(
                "Retrieval replacement started session_id=%s document_count=%s chunk_count=%s",
                session_id,
                document_count,
                chunk_count,
            )

            if not chunked_documents:
                replace = getattr(
                    self.storage,
                    "replace_session_document_chunks",
                    self.storage.replace_document_chunks,
                )
                replace(session_id, [])
                return {
                    "status": "success",
                    "document_count": 0,
                    "chunk_count": 0,
                    "indexed_count": 0,
                    "failed_count": 0,
                    "vector_size": 0,
                }

            embeddings = get_embedding_service().embed_documents(
                [str(document["content"]) for document in chunked_documents]
            )
            if len(embeddings) != chunk_count:
                raise ValueError("Document and embedding counts do not match.")
            expected_dimensions = _RAG_CONFIG.embedding.dimensions
            if any(len(embedding) != expected_dimensions for embedding in embeddings):
                raise ValueError(
                    "The embedding model must return "
                    f"{expected_dimensions}-dimensional vectors."
                )

            rows: list[dict[str, object]] = []
            for index, (document, embedding) in enumerate(
                zip(chunked_documents, embeddings)
            ):
                metadata = dict(document.get("metadata") or {})
                source_id = str(document["id"])
                source_ids = document.get("source_ids")
                if not isinstance(source_ids, list):
                    source_ids = metadata.get("source_ids")
                source_dataset_id = str(
                    metadata.get("dataset_id") or dataset_id
                )
                metadata.update(
                    {
                        "session_id": session_id,
                        "dataset_id": source_dataset_id,
                        "source_ids": list(source_ids or []),
                        "title": str(
                            document.get("title")
                            or metadata.get("title")
                            or source_id
                        ),
                    }
                )
                chunk_index = int(metadata.get("chunk_index", 0))
                metadata["chunk_index"] = chunk_index
                rows.append(
                    {
                        "session_id": session_id,
                        "dataset_id": source_dataset_id,
                        "source_id": source_id,
                        "document_type": str(
                            document.get("document_type")
                            or metadata.get("document_type")
                            or "dataset_overview"
                        ),
                        "chunk_index": chunk_index,
                        "content": str(document["content"]),
                        "metadata": metadata,
                        "embedding": [float(value) for value in embedding],
                    }
                )

            replace = getattr(
                self.storage,
                "replace_session_document_chunks",
                self.storage.replace_document_chunks,
            )
            replaced_count = replace(session_id, rows)
            if replaced_count != len(rows):
                raise ValueError(
                    "Atomic retrieval replacement returned an unexpected row count."
                )
            logger.info(
                "New documents indexed session_id=%s indexed_count=%s",
                session_id,
                len(rows),
            )
            return {
                "status": "success",
                "document_count": document_count,
                "chunk_count": chunk_count,
                "indexed_count": len(rows),
                "failed_count": 0,
                "vector_size": len(embeddings[0]) if embeddings else 0,
            }
        except Exception as exc:
            logger.exception(
                "Retrieval replacement failed session_id=%s document_count=%s",
                session_id,
                document_count,
            )
            return {
                "status": "failed",
                "document_count": document_count,
                "chunk_count": 0,
                "indexed_count": 0,
                "failed_count": document_count,
                "vector_size": 0,
                "message": str(exc),
            }

    def _chunk_retrieval_documents(
        self,
        retrieval_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for document in retrieval_documents:
            source_id = str(document["id"])
            document_type = str(
                document.get("document_type")
                or (document.get("metadata") or {}).get("document_type")
                or "dataset_overview"
            )
            metadata = {
                **dict(document.get("metadata") or {}),
                "source_id": source_id,
                "document_type": document_type,
            }
            for chunk in self._builder.chunk_documents(
                [
                    RagDocument(
                        page_content=str(document["content"]),
                        metadata=metadata,
                    )
                ]
            ):
                output.append(
                    {
                        **document,
                        "id": source_id,
                        "content": chunk.page_content,
                        "document_type": document_type,
                        "metadata": dict(chunk.metadata),
                    }
                )
        return output

    def ensure_index(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
    ) -> bool:
        try:
            self.index_dataset(agent_input=agent_input, profile=profile, force=False)
            return True
        except Exception:
            logger.exception(
                "Recoverable RAG indexing failure session_id=%s",
                agent_input.sessionId,
            )
            return False

    def retrieve(
        self,
        session_id: str,
        query: str,
        limit: int = _RAG_CONFIG.retrieval.vector_search_limit,
    ) -> list[RetrievedDocument]:
        """Retrieve evidence through one session-scoped vector-search path."""
        if not session_id:
            raise ValueError("A session ID is required for retrieval.")

        try:
            query_vector = get_embedding_service().embed_query(query)
            expected_dimensions = _RAG_CONFIG.embedding.dimensions
            if len(query_vector) != expected_dimensions:
                raise ValueError(
                    "The embedding model must return "
                    f"{expected_dimensions}-dimensional vectors."
                )
            logger.info(
                "RAG session filter applied session_id=%s dataset_id=%s",
                session_id,
                session_id,
            )
            match_session = getattr(
                self.storage,
                "match_session_document_chunks",
                None,
            )
            if callable(match_session):
                rows = match_session(
                    session_id=session_id,
                    query_embedding=[float(value) for value in query_vector],
                    match_count=limit,
                    match_threshold=_RAG_CONFIG.retrieval.match_threshold,
                )
            else:
                rows = self.storage.match_document_chunks(
                    dataset_id=session_id,
                    query_embedding=[float(value) for value in query_vector],
                    match_count=limit,
                    match_threshold=_RAG_CONFIG.retrieval.match_threshold,
                )
            documents = self._dedupe_retrieved(
                [
                    RetrievedDocument(
                        page_content=str(row.get("content", "")).strip(),
                        metadata=self._result_metadata(row),
                        score=float(row.get("similarity") or 0.0),
                    )
                    for row in rows
                    if str(row.get("content", "")).strip()
                ]
            )
            session_documents = [
                document
                for document in documents
                if self._matches_session(document, session_id)
            ][:limit]
            logger.info(
                "RAG retrieval session_id=%s query=%r candidates=%s matches=%s",
                session_id,
                query[:120],
                len(documents),
                [
                    (
                        str(document.metadata.get("source_id", "unknown_source")),
                        round(document.score, 3),
                    )
                    for document in session_documents
                ],
            )
            return session_documents
        except Exception:
            logger.exception("RAG retrieval failed session_id=%s", session_id)
            return []

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RerankedDocument]:
        if not documents:
            return []
        return get_reranker().rerank(
            query=query,
            documents=documents,
            limit=_RAG_CONFIG.reranking.limit,
        )

    def build_context(
        self,
        documents: list[RerankedDocument] | list[RetrievedDocument],
        calculated_evidence: str | None = None,
    ) -> str:
        sections: list[str] = []
        used_chars = 0
        if calculated_evidence:
            evidence = calculated_evidence.strip()
            sections.append(evidence)
            used_chars += len(evidence)

        for document in documents[:_RAG_CONFIG.reranking.limit]:
            metadata = document.metadata
            source_id = str(metadata.get("source_id", "unknown_source"))
            header = [
                f"[SOURCE {source_id}]",
                f"Type: {metadata.get('document_type', 'unknown')}",
            ]
            for key, label in (
                ("measure", "Measure"),
                ("dimension", "Dimension"),
                ("date_field", "Date field"),
            ):
                if metadata.get(key):
                    header.append(f"{label}: {metadata[key]}")
            header.append("Content:")
            remaining = (
                _RAG_CONFIG.retrieval.max_context_chars
                - used_chars
                - sum(len(item) for item in header)
                - 32
            )
            if remaining <= 0:
                break
            content = document.page_content.strip()
            if len(content) > remaining:
                content = content[: max(0, remaining - 20)].rstrip() + "\n[truncated]"
            item = "\n".join([*header, content])
            sections.append(item)
            used_chars += len(item)
            if used_chars >= _RAG_CONFIG.retrieval.max_context_chars:
                break
        context = "\n\n".join(sections).strip()
        logger.debug(
            "RAG context built sources=%s chars=%s calculated=%s",
            [str(document.metadata.get("source_id", "unknown_source")) for document in documents],
            len(context),
            bool(calculated_evidence),
        )
        return context

    def route_query(self, query: str, profile: dict[str, Any]) -> QueryType:
        lowered = query.casefold()
        summary = profile.get("summary", {})
        columns = [
            str(column).casefold()
            for column in [*summary.get("measures", []), *summary.get("dimensions", [])]
        ]
        mentions_column = any(column and column in lowered for column in columns)
        has_calc = any(
            word in lowered
            for word in (
                "sum",
                "total",
                "average",
                "avg",
                "count",
                "highest",
                "lowest",
                "maximum",
                "minimum",
                "top",
                "bottom",
                "best",
                "performed best",
                )
        )
        has_forecast = any(word in lowered for word in ("predict", "forecast", "project"))
        has_compare = any(word in lowered for word in ("compare", "versus", " vs ", "difference"))
        has_explain = any(word in lowered for word in ("why", "explain", "describe", "insight", "trend"))

        if has_forecast:
            return "mixed" if has_explain else "forecast"
        if has_calc and has_explain:
            return "mixed"
        if has_calc or (mentions_column and any(word in lowered for word in ("how many", "how much"))):
            return "calculation"
        if has_compare:
            return "comparison"
        if has_explain:
            return "explanation"
        return "lookup"

    def calculate_evidence(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
        query_type: str,
        profile: dict[str, Any],
    ) -> CalculatedEvidence | None:
        if query_type not in {"calculation", "forecast", "comparison", "mixed"}:
            return None
        try:
            analytics = DeterministicAnalytics(agent_input=agent_input, profile=profile)
            return analytics.calculate(query=query)
        except Exception:
            logger.exception(
                "Deterministic analytics failed session_id=%s",
                agent_input.sessionId,
            )
            return None

    def _lock_for_session(self, session_id: str) -> threading.Lock:
        with self._service_lock:
            if session_id not in self._index_locks:
                self._index_locks[session_id] = threading.Lock()
            return self._index_locks[session_id]

    @staticmethod
    def _file_signature(file_path: str) -> dict[str, str | int | float]:
        path = Path(file_path).resolve()
        stat = path.stat()
        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def _dedupe_retrieved(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        output: list[RetrievedDocument] = []
        seen: set[str] = set()
        for document in documents:
            key = f"{document.metadata.get('source_id')}:{document.page_content}"
            if key in seen:
                continue
            seen.add(key)
            output.append(document)
        return output

    @staticmethod
    def _result_metadata(row: dict[str, object]) -> dict[str, Any]:
        raw_metadata = row.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata["source_id"] = str(row.get("source_id", metadata.get("source_id", "")))
        metadata["document_type"] = str(
            row.get("document_type", metadata.get("document_type", "dataset_overview"))
        )
        if row.get("dataset_id") is not None:
            metadata.setdefault("dataset_id", str(row["dataset_id"]))
        if row.get("session_id") is not None:
            metadata.setdefault("session_id", str(row["session_id"]))
        return {str(key): value for key, value in metadata.items()}

    @staticmethod
    def _matches_session(document: RetrievedDocument, session_id: str) -> bool:
        metadata = document.metadata
        indexed_session_id = str(metadata.get("session_id") or "").strip()
        return not indexed_session_id or indexed_session_id == session_id


class DeterministicAnalytics:
    def __init__(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
    ) -> None:
        self.agent_input = agent_input
        self.profile = profile
        self.df = load_dataframe(agent_input.filePath)
        self.summary = profile.get("summary", {})
        self.date_field = self._date_field()
        self.measures = self._measures()
        self.dimensions = self._dimensions()
        self._revenue_source_columns: tuple[str, str] | None = None
        self._revenue_measure = self._find_revenue_measure()
        if self._revenue_measure is None:
            source_columns = self._find_derived_revenue_columns()
            if source_columns is not None:
                price_column, quantity_column = source_columns
                price = pd.to_numeric(self.df[price_column], errors="coerce")
                quantity = pd.to_numeric(self.df[quantity_column], errors="coerce")
                derived = price * quantity
                if derived.notna().any():
                    self.df[DERIVED_REVENUE_COLUMN] = derived
                    self.measures.append(DERIVED_REVENUE_COLUMN)
                    self._revenue_measure = DERIVED_REVENUE_COLUMN
                    self._revenue_source_columns = source_columns

    def calculate(self, query: str) -> CalculatedEvidence | None:
        lowered = query.casefold()
        if any(word in lowered for word in ("forecast", "predict", "project")):
            return self._forecast(query)

        operation = self._operation(lowered)
        if not operation:
            return None

        measure = self._query_measure(query)
        if operation == "count" and measure is None:
            return self._count_rows(query)
        if measure is None and operation in {"top", "bottom", "group_by"}:
            measure = self._default_performance_measure()
        if measure is None:
            return None

        dimension = self._query_column(query, self.dimensions)
        filters = self._filters(query, exclude=set())
        working = self._apply_filters(self.df, filters)
        if working.empty:
            return None

        if operation in {"sum", "average", "minimum", "maximum"}:
            numeric = pd.to_numeric(working[measure], errors="coerce").dropna()
            if numeric.empty:
                return None
            value = {
                "sum": numeric.sum(),
                "average": numeric.mean(),
                "minimum": numeric.min(),
                "maximum": numeric.max(),
            }[operation]
            filter_text = self._filter_text(filters)
            label = operation.title()
            measure_label = self._measure_label(measure)
            text = (
                f"Calculated evidence:\n"
                f"{label} {measure_label}{filter_text}: {self._number(value)}.\n"
                f"Source fields: {self._source_fields([*self._measure_source_fields(measure), *filters.keys()])}."
            )
            direct = (
                f"**Answer:** {label} `{measure_label}`{filter_text} is **{self._number(value)}**.\n\n"
                f"**Grounding:** Calculated from {self._measure_grounding(measure)}"
                f"{self._fields_suffix(filters)} in dataset `{self.agent_input.fileName}`."
            )
            return CalculatedEvidence(text=text, direct_answer=direct)

        if operation in {"top", "bottom", "group_by"}:
            if dimension is None:
                dimension = self._best_dimension(query)
            if dimension is None:
                return None
            filters = self._filters(query, exclude={dimension})
            working = self._apply_filters(self.df, filters)
            if working.empty:
                return None
            numeric = pd.to_numeric(working[measure], errors="coerce")
            grouped_df = pd.DataFrame({dimension: working[dimension], measure: numeric}).dropna()
            if grouped_df.empty:
                return None
            grouped = grouped_df.groupby(dimension)[measure].sum().sort_values(ascending=False)
            if grouped.empty:
                return None
            if operation == "bottom":
                selected = grouped.sort_values(ascending=True).head(1)
                item_label = "Bottom"
            elif operation == "top":
                selected = grouped.head(1)
                item_label = "Top"
            else:
                selected = grouped.head(10)
                item_label = "Grouped"
            values = "; ".join(f"{label}: {self._number(value)}" for label, value in selected.items())
            filter_text = self._filter_text(filters)
            measure_label = self._measure_label(measure)
            text = (
                f"Calculated evidence:\n"
                f"{item_label} {measure_label} by {dimension}{filter_text}: {values}.\n"
                f"Source fields: {self._source_fields([*self._measure_source_fields(measure), dimension, *filters.keys()])}."
            )
            direct = (
                f"**Answer:** {item_label} `{measure_label}` by `{dimension}`{filter_text}: "
                f"**{values}**.\n\n"
                f"**Grounding:** Calculated from {self._measure_grounding(measure)} grouped by `{dimension}`"
                f"{self._fields_suffix(filters)} in dataset `{self.agent_input.fileName}`."
            )
            return CalculatedEvidence(text=text, direct_answer=direct)

        return None

    def _count_rows(self, query: str) -> CalculatedEvidence | None:
        filters = self._filters(query, exclude=set())
        working = self._apply_filters(self.df, filters)
        if working.empty and filters:
            return None
        value = len(working)
        filter_text = self._filter_text(filters)
        text = (
            f"Calculated evidence:\n"
            f"Count of rows{filter_text}: {value:,}.\n"
            f"Source fields: {self._source_fields(list(filters.keys())) if filters else 'row count'}."
        )
        direct = (
            f"**Answer:** The row count{filter_text} is **{value:,}**.\n\n"
            f"**Grounding:** Calculated from dataset `{self.agent_input.fileName}`"
            f"{self._fields_suffix(filters)}."
        )
        return CalculatedEvidence(text=text, direct_answer=direct)

    def _forecast(self, query: str) -> CalculatedEvidence | None:
        measure = self._query_measure(query) or self._revenue_measure or self._default_performance_measure()
        time_column = self._forecast_time_column()
        if measure is None or time_column is None:
            return None

        filters = self._filters(query, exclude={time_column})
        working = self._apply_filters(self.df, filters)
        if working.empty:
            return None

        periods = self._annual_periods(working[time_column])
        values = pd.to_numeric(working[measure], errors="coerce")
        series = pd.DataFrame({"period": periods, "value": values}).dropna()
        if series.empty:
            return None
        grouped = series.groupby("period")["value"].sum().sort_index()
        if len(grouped) < 4:
            return None

        years = [int(year) for year in grouped.index]
        last_year = max(years)
        match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        target_year = int(match.group(1)) if match else last_year + 1
        if target_year <= last_year:
            return None

        x = np.arange(len(grouped), dtype=float)
        slope, intercept = np.polyfit(x, grouped.astype(float).to_numpy(), 1)
        prediction = float(slope * (len(grouped) - 1 + target_year - last_year) + intercept)
        measure_label = self._measure_label(measure)
        filter_text = self._filter_text(filters)
        value_text = (
            self._currency(prediction)
            if self._is_revenue_measure(measure)
            else self._number(prediction)
        )
        target_text = f"in {target_year}" if match else f"for the next year ({target_year})"
        source_fields = self._source_fields(
            [time_column, *self._measure_source_fields(measure), *filters.keys()]
        )
        text = (
            "Calculated evidence:\n"
            f"Forecasted total {measure_label}{filter_text} {target_text}: {value_text}.\n"
            f"Method: linear trend on annual totals from {min(years)} to {last_year}.\n"
            f"Source fields: {source_fields}."
        )
        direct = (
            f"**Answer:** The forecasted total `{measure_label}`{filter_text} {target_text} "
            f"is **{value_text}**, using a linear trend on annual totals.\n\n"
            f"**Grounding:** Calculated from {self._measure_grounding(measure)} using "
            f"`{time_column}` from {min(years)} to {last_year}."
        )
        return CalculatedEvidence(text=text, direct_answer=direct)

    def _operation(self, lowered: str) -> str | None:
        if any(word in lowered for word in ("how many", "count", "number of rows", "row count")):
            return "count"
        if any(word in lowered for word in ("highest", "top", "largest", "most", "best")):
            return "top"
        if any(word in lowered for word in ("lowest", "bottom", "smallest", "least")):
            return "bottom"
        if any(word in lowered for word in (" by ", "group by", "breakdown")):
            return "group_by"
        if any(word in lowered for word in ("average", "avg", "mean")):
            return "average"
        if any(word in lowered for word in ("minimum", "min")):
            return "minimum"
        if any(word in lowered for word in ("maximum", "max")):
            return "maximum"
        if any(word in lowered for word in ("sum", "total")):
            return "sum"
        return None

    def _filters(self, query: str, exclude: set[str]) -> dict[str, str | int]:
        filters: dict[str, str | int] = {}
        lowered = query.casefold()
        for dimension in self.dimensions[:16]:
            if dimension in exclude or dimension not in self.df.columns:
                continue
            unique_values = self.df[dimension].dropna().astype(str).unique()
            if len(unique_values) > 250:
                continue
            for value in sorted(unique_values, key=len, reverse=True):
                if value and re.search(rf"\b{re.escape(value.casefold())}\b", lowered):
                    filters[dimension] = value
                    break

        year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        if year_matches:
            year_column = self._column("Year") or self._year_like_column()
            if year_column and year_column not in exclude:
                filters[year_column] = int(year_matches[-1])
        return filters

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: dict[str, str | int],
    ) -> pd.DataFrame:
        working = df
        for column, value in filters.items():
            if column not in working.columns:
                return working.iloc[0:0]
            if isinstance(value, int):
                numeric = pd.to_numeric(working[column], errors="coerce")
                working = working[numeric == value]
            else:
                working = working[
                    working[column].astype(str).str.casefold() == value.casefold()
                ]
        return working

    def _query_column(self, query: str, columns: list[str]) -> str | None:
        lowered = query.casefold()
        for column in sorted(columns, key=len, reverse=True):
            if re.search(rf"\b{re.escape(column.casefold())}\b", lowered):
                return column
            title = column.replace("_", " ").replace("-", " ")
            if title.casefold() in lowered:
                return column
            if re.search(rf"\b{re.escape(title.casefold())}s\b", lowered):
                return column
            if "product" in lowered and "product" in title.casefold():
                return column
        if len(columns) == 1:
            return columns[0]
        return None

    def _query_measure(self, query: str) -> str | None:
        measure = self._query_column(
            query,
            [item for item in self.measures if item != DERIVED_REVENUE_COLUMN],
        )
        if measure is not None:
            return measure
        if any(term in query.casefold() for term in ("revenue", "turnover")):
            return self._revenue_measure
        return None

    def _default_performance_measure(self) -> str | None:
        if self._revenue_measure is not None:
            return self._revenue_measure
        preferred_terms = ("sales", "profit", "amount", "value")
        for term in preferred_terms:
            for measure in self.measures:
                if term in measure.casefold():
                    return measure
        return next(
            (measure for measure in self.measures if measure != DERIVED_REVENUE_COLUMN),
            None,
        )

    def _best_dimension(self, query: str = "") -> str | None:
        candidates = [
            dimension
            for dimension in self.dimensions
            if dimension in self.df.columns and 2 <= self.df[dimension].nunique(dropna=True) <= 50
        ]
        lowered = query.casefold()
        for keyword in ("product", "category", "customer", "region"):
            if keyword in lowered:
                matched = next(
                    (dimension for dimension in candidates if keyword in dimension.casefold()),
                    None,
                )
                if matched is not None:
                    return matched
        return candidates[0] if candidates else None

    def _query_category(self, column: str | None, query: str) -> str | None:
        if not column or column not in self.df.columns:
            return None
        lowered = query.casefold()
        values = sorted(
            (str(value) for value in self.df[column].dropna().unique()),
            key=len,
            reverse=True,
        )
        for value in values:
            if re.search(rf"\b{re.escape(value.casefold())}\b", lowered):
                return value
        return None

    def _measures(self) -> list[str]:
        configured = [str(item) for item in self.summary.get("measures", []) if str(item) in self.df.columns]
        if configured:
            return configured
        return [
            str(column)
            for column in self.df.select_dtypes(include="number").columns
            if not self._identifier(str(column))
        ]

    def _dimensions(self) -> list[str]:
        configured = [str(item) for item in self.summary.get("dimensions", []) if str(item) in self.df.columns]
        if configured:
            return configured
        return [
            str(column)
            for column in self.df.columns
            if str(column) not in self.measures and str(column) != self.date_field
        ]

    def _date_field(self) -> str | None:
        configured = self.summary.get("timeField")
        if isinstance(configured, str) and configured in self.df.columns:
            return configured
        return self._year_like_column()

    def _find_revenue_measure(self) -> str | None:
        candidates = [*self.measures]
        candidates.extend(
            str(column)
            for column in self.df.select_dtypes(include="number").columns
            if str(column) not in candidates
        )
        for column in candidates:
            name = self._normalised_name(column)
            if any(term in name for term in REVENUE_COLUMN_TERMS):
                return column
        return None

    def _find_derived_revenue_columns(self) -> tuple[str, str] | None:
        numeric_columns = [
            str(column)
            for column in self.df.columns
            if pd.to_numeric(self.df[column], errors="coerce").notna().any()
        ]
        price_column = self._best_named_column(
            numeric_columns,
            ("unit price", "price", "unit cost"),
        )
        quantity_column = self._best_named_column(
            numeric_columns,
            ("sales volume", "quantity", "qty", "units sold", "unit volume", "volume"),
        )
        if price_column is None or quantity_column is None or price_column == quantity_column:
            return None
        return price_column, quantity_column

    @classmethod
    def _best_named_column(
        cls,
        columns: list[str],
        terms: tuple[str, ...],
    ) -> str | None:
        for term in terms:
            for column in columns:
                if term in cls._normalised_name(column):
                    return column
        return None

    def _measure_label(self, measure: str) -> str:
        if measure == DERIVED_REVENUE_COLUMN:
            return "Revenue"
        return measure.replace("_", " ")

    def _measure_source_fields(self, measure: str) -> list[str]:
        if measure == DERIVED_REVENUE_COLUMN and self._revenue_source_columns:
            return list(self._revenue_source_columns)
        return [measure]

    def _measure_grounding(self, measure: str) -> str:
        if measure == DERIVED_REVENUE_COLUMN and self._revenue_source_columns:
            price_column, quantity_column = self._revenue_source_columns
            return f"`Revenue` derived as `{price_column}` × `{quantity_column}`"
        return f"`{measure}`"

    def _is_revenue_measure(self, measure: str) -> bool:
        return measure == self._revenue_measure or any(
            term in self._normalised_name(measure) for term in REVENUE_COLUMN_TERMS
        )

    def _forecast_time_column(self) -> str | None:
        if self.date_field and self.date_field in self.df.columns:
            return self.date_field
        return self._year_like_column()

    @staticmethod
    def _annual_periods(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_values = numeric.dropna()
        if not numeric_values.empty and numeric_values.between(1800, 3000).all():
            return numeric.round().astype("Int64")
        dates = pd.to_datetime(values, errors="coerce")
        return dates.dt.year.astype("Int64")

    @staticmethod
    def _normalised_name(value: str) -> str:
        return re.sub(r"[_-]+", " ", value).casefold()

    def _column(self, name: str) -> str | None:
        expected = name.casefold()
        for column in self.df.columns:
            if str(column).casefold() == expected:
                return str(column)
        return None

    def _year_like_column(self) -> str | None:
        for column in self.df.columns:
            if "year" in str(column).casefold():
                return str(column)
        return None

    def _identifier(self, column: str) -> bool:
        name = column.lower()
        looks_like_id = (
            name == "id"
            or name.endswith("_id")
            or any(word in name for word in ("code", "reference", "number"))
        )
        return looks_like_id and (
            len(self.df) == 0 or self.df[column].nunique(dropna=True) / len(self.df) >= 0.5
        )

    @staticmethod
    def _filter_text(filters: dict[str, str | int]) -> str:
        if not filters:
            return ""
        return " for " + ", ".join(f"{key}={value}" for key, value in filters.items())

    @staticmethod
    def _fields_suffix(filters: dict[str, str | int]) -> str:
        if not filters:
            return ""
        fields = ", ".join(f"`{field}`" for field in filters)
        return f" with filters from {fields}"

    @staticmethod
    def _source_fields(fields: list[str]) -> str:
        values = [field for field in fields if field]
        return ", ".join(dict.fromkeys(values)) if values else "row count"

    @staticmethod
    def _number(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:,.2f}" if math.isfinite(number) else str(value)

    @classmethod
    def _currency(cls, value: float) -> str:
        return f"${value:,.2f}"


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Only CSV, Excel and JSON files are supported.")


def compact_profile_for_chat(profile: dict[str, Any]) -> str:
    summary = profile.get("summary", {})
    compact = {
        "fileName": summary.get("fileName"),
        "rowCount": summary.get("rowCount"),
        "columnCount": summary.get("columnCount"),
        "timeField": summary.get("timeField"),
        "period": summary.get("period"),
        "measures": summary.get("measures", [])[:8],
        "dimensions": summary.get("dimensions", [])[:10],
        "quality": summary.get("quality"),
        "metrics": profile.get("metrics", [])[:5],
        "bar": profile.get("bar"),
        "donut": profile.get("donut"),
        "timeline": profile.get("timeline"),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"), default=str)


rag_service = RagService()
