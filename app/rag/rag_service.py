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

from app.rag.config import MAX_CONTEXT_CHARS, RERANK_LIMIT, VECTOR_SEARCH_LIMIT
from app.rag.document_builder import DatasetDocumentBuilder
from app.rag.embedding_service import get_embedding_service
from app.rag.models import CalculatedEvidence, IndexStatus, QueryType, RerankedDocument, RetrievedDocument
from app.rag.reranker import get_reranker
from app.schemas.business_intelligence import BusinessIntelligenceAgentInput
from app.services.supabase_service import SupabaseService, supabase_service


logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_LIMIT = 6


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
                vector_size=384,
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
                    vector_size=384,
                )

            df = load_dataframe(agent_input.filePath)
            documents = self._builder.build(
                df=df,
                profile=profile,
                session_id=agent_input.sessionId,
                file_name=agent_input.fileName,
            )
            logger.info(
                "Retrieval replacement started session_id=%s document_count=%s",
                agent_input.sessionId,
                len(documents),
            )
            # document_chunks is keyed by dataset_id, which is the single-agent
            # session identifier.  Delete only this session before replacements.
            self.storage.delete_document_chunks(agent_input.sessionId)
            logger.info(
                "Existing session documents deleted session_id=%s",
                agent_input.sessionId,
            )
            embeddings = get_embedding_service().embed_documents(
                [document.page_content for document in documents]
            )
            if documents and len(documents) != len(embeddings):
                raise ValueError("Document and embedding counts do not match.")

            rows: list[dict[str, object]] = []
            for index, document in enumerate(documents):
                metadata = dict(document.metadata)
                source_id = str(metadata.get("source_id", f"document_{index}"))
                document_type = str(metadata.get("document_type", "dataset_overview"))
                chunk_index = int(metadata.get("chunk_index", index))
                metadata["chunk_index"] = chunk_index
                rows.append(
                    {
                        "dataset_id": agent_input.sessionId,
                        "source_id": source_id,
                        "document_type": document_type,
                        "chunk_index": chunk_index,
                        "content": document.page_content,
                        "metadata": metadata,
                        "embedding": [float(value) for value in embeddings[index]],
                    }
                )
            self.storage.insert_document_chunks(rows, batch_size=50)
            logger.info(
                "New documents indexed session_id=%s indexed_count=%s",
                agent_input.sessionId,
                len(rows),
            )
            self._completed_signatures[agent_input.sessionId] = signature
            vector_size = len(embeddings[0]) if embeddings else 0
            return IndexStatus(
                session_id=agent_input.sessionId,
                collection_name="document_chunks",
                document_count=len(documents),
                chunk_count=len(documents),
                vector_size=vector_size,
            )

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

    def index_exists(self, session_id: str) -> bool:
        return session_id in self._completed_signatures

    def delete_session_index(self, session_id: str) -> None:
        self.storage.delete_document_chunks(session_id)
        self._completed_signatures.pop(session_id, None)

    def close(self) -> None:
        return None

    def retrieve(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
    ) -> list[RetrievedDocument]:
        try:
            query_vector = get_embedding_service().embed_query(query)
            rows = self.storage.match_document_chunks(
                dataset_id=agent_input.sessionId,
                query_embedding=[float(value) for value in query_vector],
                match_count=VECTOR_SEARCH_LIMIT,
                match_threshold=0.2,
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
            logger.info(
                "RAG retrieval session_id=%s query=%r candidates=%s",
                agent_input.sessionId,
                query[:120],
                len(documents),
            )
            return documents
        except Exception:
            logger.exception(
                "RAG retrieval failed session_id=%s",
                agent_input.sessionId,
            )
            return []

    def retrieve_for_session(
        self,
        session_id: str,
        query: str,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> list[RetrievedDocument]:
        """Retrieve multi-agent evidence through the database-scoped vector query."""
        if not session_id:
            raise ValueError("A session ID is required for retrieval.")

        query_vector = get_embedding_service().embed_query(query)
        logger.info(
            "RAG session filter applied session_id=%s dataset_id=%s",
            session_id,
            session_id,
        )
        rows = self.storage.match_document_chunks(
            dataset_id=session_id,
            query_embedding=[float(value) for value in query_vector],
            match_count=limit,
            match_threshold=0.2,
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
        return [
            document
            for document in documents
            if self._matches_session(document, session_id)
        ][:limit]

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RerankedDocument]:
        if not documents:
            return []
        return get_reranker().rerank(query=query, documents=documents, limit=RERANK_LIMIT)

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

        for document in documents[:RERANK_LIMIT]:
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
            remaining = MAX_CONTEXT_CHARS - used_chars - sum(len(item) for item in header) - 32
            if remaining <= 0:
                break
            content = document.page_content.strip()
            if len(content) > remaining:
                content = content[: max(0, remaining - 20)].rstrip() + "\n[truncated]"
            item = "\n".join([*header, content])
            sections.append(item)
            used_chars += len(item)
            if used_chars >= MAX_CONTEXT_CHARS:
                break
        return "\n\n".join(sections).strip()

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
        return {str(key): value for key, value in metadata.items()}

    @staticmethod
    def _matches_session(document: RetrievedDocument, session_id: str) -> bool:
        metadata = document.metadata
        indexed_session_id = str(metadata.get("session_id") or "").strip()
        indexed_dataset_id = str(metadata.get("dataset_id") or "").strip()
        return (
            (not indexed_session_id or indexed_session_id == session_id)
            and (not indexed_dataset_id or indexed_dataset_id == session_id)
        )


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
        self.measures = self._measures()
        self.dimensions = self._dimensions()
        self.date_field = self.summary.get("timeField")

    def calculate(self, query: str) -> CalculatedEvidence | None:
        lowered = query.casefold()
        if any(word in lowered for word in ("forecast", "predict", "project")) and "revenue" in lowered:
            return self._forecast_revenue(query)

        operation = self._operation(lowered)
        if not operation:
            return None

        measure = self._query_column(query, self.measures)
        if operation == "count" and measure is None:
            return self._count_rows(query)
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
            text = (
                f"Calculated evidence:\n"
                f"{label} {measure}{filter_text}: {self._number(value)}.\n"
                f"Source fields: {self._source_fields([measure, *filters.keys()])}."
            )
            direct = (
                f"**Answer:** {label} `{measure}`{filter_text} is **{self._number(value)}**.\n\n"
                f"**Grounding:** Calculated from `{measure}`"
                f"{self._fields_suffix(filters)} in dataset `{self.agent_input.fileName}`."
            )
            return CalculatedEvidence(text=text, direct_answer=direct)

        if operation in {"top", "bottom", "group_by"}:
            if dimension is None:
                dimension = self._best_dimension()
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
            text = (
                f"Calculated evidence:\n"
                f"{item_label} {measure} by {dimension}{filter_text}: {values}.\n"
                f"Source fields: {self._source_fields([measure, dimension, *filters.keys()])}."
            )
            direct = (
                f"**Answer:** {item_label} `{measure}` by `{dimension}`{filter_text}: "
                f"**{values}**.\n\n"
                f"**Grounding:** Calculated from `{measure}` grouped by `{dimension}`"
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

    def _forecast_revenue(self, query: str) -> CalculatedEvidence | None:
        year_column = self._column("Year")
        price_column = self._column("Price_USD")
        volume_column = self._column("Sales_Volume")
        if not year_column or not price_column or not volume_column:
            return None

        match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if not match:
            return None
        target_year = int(match.group(1))

        working = self.df[[year_column, price_column, volume_column]].copy()
        region_column = self._column("Region")
        region = self._query_category(region_column, query) if region_column else None
        if region_column and region:
            working[region_column] = self.df[region_column]
            working = working[
                working[region_column].astype(str).str.casefold() == region.casefold()
            ]

        working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
        working[price_column] = pd.to_numeric(working[price_column], errors="coerce")
        working[volume_column] = pd.to_numeric(working[volume_column], errors="coerce")
        working = working.dropna(subset=[year_column, price_column, volume_column])
        if working.empty:
            return None

        working["revenue"] = working[price_column] * working[volume_column]
        grouped = working.groupby(year_column)["revenue"].sum().sort_index()
        grouped = grouped[grouped.index.astype(int) == grouped.index]
        if len(grouped) < 4:
            return None

        years = [int(year) for year in grouped.index]
        last_year = max(years)
        if target_year <= last_year:
            return None

        values = grouped.astype(float).to_numpy()
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        steps_ahead = target_year - last_year
        prediction = float(slope * (len(values) - 1 + steps_ahead) + intercept)

        region_text = f" for {region}" if region else ""
        filter_text = f", filtered to Region={region}" if region else ""
        value_text = self._currency(prediction)
        text = (
            "Calculated evidence:\n"
            f"Forecasted total revenue{region_text} in {target_year}: {value_text}.\n"
            f"Method: linear trend on annual revenue from {min(years)} to {last_year}.\n"
            f"Source fields: {year_column}, {price_column}, {volume_column}"
            f"{', ' + region_column if region_column and region else ''}."
        )
        direct = (
            f"**Answer:** The forecasted total revenue{region_text} in "
            f"{target_year} is **{value_text}**, using a linear trend on annual revenue.\n\n"
            f"**Grounding:** Dataset `{self.agent_input.fileName}`; revenue = "
            f"`{price_column} * {volume_column}`{filter_text}, using "
            f"`{year_column}` {min(years)}-{last_year}."
        )
        return CalculatedEvidence(text=text, direct_answer=direct)

    def _operation(self, lowered: str) -> str | None:
        if any(word in lowered for word in ("how many", "count", "number of rows", "row count")):
            return "count"
        if any(word in lowered for word in ("highest", "top", "largest", "most")):
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
        if len(columns) == 1:
            return columns[0]
        return None

    def _best_dimension(self) -> str | None:
        candidates = [
            dimension
            for dimension in self.dimensions
            if dimension in self.df.columns and 2 <= self.df[dimension].nunique(dropna=True) <= 50
        ]
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
