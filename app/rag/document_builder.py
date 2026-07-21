from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_rag_config
from app.rag.models import RagDocument


_CHUNKING_POLICY = get_rag_config().chunking


class DatasetDocumentBuilder:
    def build(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        session_id: str,
        file_name: str,
    ) -> list[RagDocument]:
        working = df.copy()
        summary = profile.get("summary", {})
        date_field = summary.get("timeField")
        if isinstance(date_field, str) and date_field in working.columns:
            working[date_field] = pd.to_datetime(working[date_field], errors="coerce")

        measures = self._measures(working, summary)
        dimensions = self._dimensions(working, measures, date_field, summary)

        documents: list[RagDocument] = [
            self._dataset_overview(
                working,
                profile,
                session_id,
                file_name,
                measures,
                dimensions,
                date_field,
            )
        ]
        documents.extend(
            self._column_profiles(
                working,
                session_id,
                file_name,
                measures,
                dimensions,
                date_field,
            )
        )
        documents.extend(self._numeric_summaries(working, session_id, file_name, measures))
        documents.extend(
            self._category_analyses(
                working,
                session_id,
                file_name,
                measures,
                dimensions,
            )
        )
        documents.extend(
            self._timeline_analyses(
                working,
                profile,
                session_id,
                file_name,
                measures,
                date_field,
            )
        )
        documents.extend(
            self._row_batches(
                working,
                session_id,
                file_name,
                measures,
                dimensions,
                date_field,
            )
        )
        return self.chunk_documents(documents)

    def build_row_documents(
        self,
        df: pd.DataFrame,
        session_id: str,
        file_name: str,
        measures: list[str],
        dimensions: list[str],
        date_field: str | None,
    ) -> list[RagDocument]:
        """Build bounded raw-row evidence for either pipeline mode."""
        return self.chunk_documents(
            self._row_batches(
                df,
                session_id,
                file_name,
                measures,
                dimensions,
                date_field,
            )
        )

    def chunk_documents(self, documents: list[RagDocument]) -> list[RagDocument]:
        chunks: list[RagDocument] = []
        for document in documents:
            content = document.page_content.strip()
            if len(content) <= _CHUNKING_POLICY.size:
                chunks.append(document)
                continue

            for chunk_index, chunk in enumerate(self._split_content(content)):
                chunks.append(
                    RagDocument(
                        page_content=chunk,
                        metadata={**document.metadata, "chunk_index": chunk_index},
                    )
                )
        return chunks

    @staticmethod
    def _split_content(content: str) -> list[str]:
        """Split on readable boundaries while enforcing the configured limit."""
        size = _CHUNKING_POLICY.size
        overlap = _CHUNKING_POLICY.overlap
        output: list[str] = []
        start = 0
        content_length = len(content)

        while start < content_length:
            hard_end = min(start + size, content_length)
            end = hard_end
            if hard_end < content_length:
                minimum_break = start + max(1, size // 2)
                candidates = (
                    content.rfind("\n", minimum_break, hard_end + 1),
                    content.rfind(" ", minimum_break, hard_end + 1),
                )
                boundary = max(candidates)
                if boundary > start:
                    end = boundary

            chunk = content[start:end].strip()
            if chunk:
                output.append(chunk)
            if end >= content_length:
                break

            next_start = max(start + 1, end - overlap)
            while next_start < end and content[next_start].isspace():
                next_start += 1
            start = next_start

        return output

    def _dataset_overview(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        session_id: str,
        file_name: str,
        measures: list[str],
        dimensions: list[str],
        date_field: str | None,
    ) -> RagDocument:
        summary = profile.get("summary", {})
        quality = summary.get("quality", {})
        period = summary.get("period")
        lines = [
            f"Dataset overview for {file_name}.",
            f"Rows: {len(df):,}. Columns: {len(df.columns):,}.",
            f"Detected period: {period.get('label') if isinstance(period, dict) else 'Not detected'}.",
            f"Date field: {date_field or 'Not detected'}.",
            f"Measures: {', '.join(measures) if measures else 'None detected'}.",
            f"Dimensions: {', '.join(dimensions) if dimensions else 'None detected'}.",
            (
                "Data quality: "
                f"{quality.get('completenessPercent', 'unknown')}% complete; "
                f"{quality.get('missingValueCount', 'unknown')} missing values; "
                f"{quality.get('duplicateRowCount', 'unknown')} duplicate rows."
            ),
        ]
        return RagDocument(
            "\n".join(lines),
            self._metadata(
                session_id,
                file_name,
                "dataset_overview",
                "dataset_overview",
                date_field=date_field,
            ),
        )

    def _column_profiles(
        self,
        df: pd.DataFrame,
        session_id: str,
        file_name: str,
        measures: list[str],
        dimensions: list[str],
        date_field: str | None,
    ) -> list[RagDocument]:
        important = self._unique_keep_order([*measures, *dimensions, *( [date_field] if date_field else [] )])[:30]
        output: list[RagDocument] = []
        for column in important:
            if column not in df.columns:
                continue
            series = df[column]
            lines = [
                f"Column profile for {column}.",
                f"Inferred type: {self._inferred_type(series)}.",
                f"Missing count: {int(series.isna().sum()):,}.",
                f"Unique count: {int(series.nunique(dropna=True)):,}.",
            ]
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if column in measures and not numeric.empty:
                lines.extend(
                    [
                        f"Minimum: {self._number(numeric.min())}.",
                        f"Maximum: {self._number(numeric.max())}.",
                        f"Average: {self._number(numeric.mean())}.",
                        f"Median: {self._number(numeric.median())}.",
                    ]
                )
            elif column in dimensions:
                values = series.dropna().astype(str).value_counts().head(8)
                if not values.empty:
                    top_values = ", ".join(
                        f"{label} ({int(count)})" for label, count in values.items()
                    )
                    lines.append(f"Top values: {top_values}.")

            output.append(
                RagDocument(
                    "\n".join(lines),
                    self._metadata(
                        session_id,
                        file_name,
                        "column_profile",
                        f"column_{self._slug(column)}",
                        measure=column if column in measures else None,
                        dimension=column if column in dimensions else None,
                        date_field=column if column == date_field else None,
                    ),
                )
            )
        return output

    def _numeric_summaries(
        self,
        df: pd.DataFrame,
        session_id: str,
        file_name: str,
        measures: list[str],
    ) -> list[RagDocument]:
        output: list[RagDocument] = []
        for measure in measures[:12]:
            numeric = pd.to_numeric(df[measure], errors="coerce").dropna()
            if numeric.empty:
                continue
            lines = [
                f"Numeric summary for measure {measure}.",
                f"Count: {int(numeric.count()):,}.",
                f"Sum: {self._number(numeric.sum())}.",
                f"Average: {self._number(numeric.mean())}.",
                f"Median: {self._number(numeric.median())}.",
                f"Minimum: {self._number(numeric.min())}.",
                f"Maximum: {self._number(numeric.max())}.",
            ]
            output.append(
                RagDocument(
                    "\n".join(lines),
                    self._metadata(
                        session_id,
                        file_name,
                        "numeric_summary",
                        f"numeric_{self._slug(measure)}",
                        measure=measure,
                    ),
                )
            )
        return output

    def _category_analyses(
        self,
        df: pd.DataFrame,
        session_id: str,
        file_name: str,
        measures: list[str],
        dimensions: list[str],
    ) -> list[RagDocument]:
        output: list[RagDocument] = []
        for dimension in dimensions[:8]:
            if dimension not in df.columns:
                continue
            unique_count = int(df[dimension].nunique(dropna=True))
            if not 2 <= unique_count <= 50:
                continue
            for measure in measures[:6]:
                numeric = pd.to_numeric(df[measure], errors="coerce")
                working = pd.DataFrame({dimension: df[dimension], measure: numeric}).dropna()
                if working.empty:
                    continue
                aggregation = "mean" if self._average(measure) else "sum"
                grouped = (
                    working.groupby(dimension)[measure]
                    .agg(aggregation)
                    .sort_values(ascending=False)
                    .head(10)
                )
                if grouped.empty:
                    continue
                values = "\n".join(
                    f"- {label}: {self._number(value)}"
                    for label, value in grouped.items()
                )
                content = (
                    f"Category analysis for {measure} by {dimension}.\n"
                    f"Aggregation: {aggregation}.\n"
                    f"Top categories:\n{values}"
                )
                output.append(
                    RagDocument(
                        content,
                        self._metadata(
                            session_id,
                            file_name,
                            "category_analysis",
                            f"category_{self._slug(dimension)}_{self._slug(measure)}",
                            measure=measure,
                            dimension=dimension,
                        ),
                    )
                )
        return output

    def _timeline_analyses(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        session_id: str,
        file_name: str,
        measures: list[str],
        date_field: str | None,
    ) -> list[RagDocument]:
        output: list[RagDocument] = []
        timeline = profile.get("timeline")
        if isinstance(timeline, dict) and timeline.get("points"):
            measure = str(timeline.get("measure") or "")
            lines = [
                f"Timeline analysis for {measure}.",
                f"Date field: {date_field or 'Not detected'}.",
                f"Aggregation: {timeline.get('aggregation')}.",
                f"Granularity: {timeline.get('granularity')}.",
                "Historical points:",
            ]
            lines.extend(
                f"- {item.get('period')}: {item.get('value')}"
                for item in timeline.get("points", [])[:18]
                if isinstance(item, dict)
            )
            anomalies = timeline.get("anomalies") or []
            lines.append("Detected anomalies:")
            lines.extend(
                f"- {item.get('period')}: {item.get('value')} ({item.get('reason')})"
                for item in anomalies[:8]
                if isinstance(item, dict)
            )
            forecast = timeline.get("forecast") or []
            lines.append("Forecast points:")
            lines.extend(
                f"- {item.get('period')}: {item.get('value')}"
                for item in forecast[:6]
                if isinstance(item, dict)
            )
            output.append(
                RagDocument(
                    "\n".join(lines),
                    self._metadata(
                        session_id,
                        file_name,
                        "timeline_analysis",
                        f"timeline_{self._slug(measure or 'main')}",
                        measure=measure or None,
                        date_field=date_field,
                    ),
                )
            )

        if not date_field or date_field not in df.columns:
            return output

        dates = pd.to_datetime(df[date_field], errors="coerce")
        if dates.dropna().empty:
            return output

        for measure in measures[:3]:
            if any(doc.metadata.get("measure") == measure for doc in output):
                continue
            numeric = pd.to_numeric(df[measure], errors="coerce")
            working = pd.DataFrame({"date": dates, measure: numeric}).dropna()
            if working.empty:
                continue
            granularity, code = self._grain(working["date"])
            working["period"] = working["date"].dt.to_period(code)
            aggregation = "mean" if self._average(measure) else "sum"
            grouped = working.groupby("period")[measure].agg(aggregation).tail(18)
            points = "\n".join(
                f"- {period}: {self._number(value)}" for period, value in grouped.items()
            )
            output.append(
                RagDocument(
                    (
                        f"Timeline analysis for {measure}.\n"
                        f"Date field: {date_field}.\n"
                        f"Aggregation: {aggregation}.\n"
                        f"Granularity: {granularity}.\n"
                        f"Historical points:\n{points}"
                    ),
                    self._metadata(
                        session_id,
                        file_name,
                        "timeline_analysis",
                        f"timeline_{self._slug(measure)}",
                        measure=measure,
                        date_field=date_field,
                    ),
                )
            )
        return output

    def _row_batches(
        self,
        df: pd.DataFrame,
        session_id: str,
        file_name: str,
        measures: list[str],
        dimensions: list[str],
        date_field: str | None,
    ) -> list[RagDocument]:
        if df.empty:
            return []
        columns = self._unique_keep_order(
            [*( [date_field] if date_field else [] ), *dimensions, *measures, *[str(column) for column in df.columns]]
        )[:_CHUNKING_POLICY.max_columns_per_row_document]
        max_rows = (
            _CHUNKING_POLICY.max_row_batch_documents
            * _CHUNKING_POLICY.rows_per_batch_document
        )
        sample_size = min(len(df), max_rows)
        indices = (
            sorted(set(np.linspace(0, len(df) - 1, sample_size, dtype=int).tolist()))
            if sample_size
            else []
        )
        output: list[RagDocument] = []
        for batch_start in range(
            0,
            len(indices),
            _CHUNKING_POLICY.rows_per_batch_document,
        ):
            batch_indices = indices[
                batch_start : batch_start
                + _CHUNKING_POLICY.rows_per_batch_document
            ]
            if not batch_indices:
                continue
            lines = [
                (
                    "Sampled row batch. Rows are evenly sampled from the dataset; "
                    "this is not the complete dataset."
                )
            ]
            for index in batch_indices:
                row = df.iloc[index]
                values = "; ".join(
                    f"{column}={self._value(row[column])}"
                    for column in columns
                    if column in df.columns
                )
                lines.append(f"Row {index}: {values}")
            row_start = int(batch_indices[0])
            row_end = int(batch_indices[-1])
            output.append(
                RagDocument(
                    "\n".join(lines),
                    self._metadata(
                        session_id,
                        file_name,
                        "row_batch",
                        f"row_batch_{row_start}_{row_end}",
                        date_field=date_field,
                        row_start=row_start,
                        row_end=row_end,
                    ),
                )
            )
        return output

    @staticmethod
    def _metadata(
        session_id: str,
        file_name: str,
        document_type: str,
        source_id: str,
        measure: str | None = None,
        dimension: str | None = None,
        date_field: str | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> dict[str, str | int | float | bool]:
        values: dict[str, str | int | float | bool | None] = {
            "session_id": session_id,
            "file_name": file_name,
            "document_type": document_type,
            "source_id": source_id,
            "measure": measure,
            "dimension": dimension,
            "date_field": date_field,
            "row_start": row_start,
            "row_end": row_end,
        }
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _measures(df: pd.DataFrame, summary: dict[str, Any]) -> list[str]:
        configured = [str(item) for item in summary.get("measures", []) if str(item) in df.columns]
        if configured:
            return configured[:12]
        return [
            str(column)
            for column in df.select_dtypes(include="number").columns
            if not DatasetDocumentBuilder._identifier(df, str(column))
        ][:12]

    @staticmethod
    def _dimensions(
        df: pd.DataFrame,
        measures: list[str],
        date_field: str | None,
        summary: dict[str, Any],
    ) -> list[str]:
        configured = [str(item) for item in summary.get("dimensions", []) if str(item) in df.columns]
        if configured:
            return configured[:16]
        return [
            str(column)
            for column in df.columns
            if str(column) not in measures and str(column) != date_field
        ][:16]

    @staticmethod
    def _identifier(df: pd.DataFrame, column: str) -> bool:
        name = column.lower()
        looks_like_id = (
            name == "id"
            or name.endswith("_id")
            or any(word in name for word in ("code", "reference", "number"))
        )
        return looks_like_id and (
            len(df) == 0 or df[column].nunique(dropna=True) / len(df) >= 0.5
        )

    @staticmethod
    def _inferred_type(series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        name = str(series.name).casefold()
        if any(word in name for word in ("date", "time", "year", "month", "period")):
            parsed_dates = pd.to_datetime(series, errors="coerce")
            if len(series) and parsed_dates.notna().mean() >= 0.6:
                return "date"
        if series.nunique(dropna=True) <= max(50, len(series) * 0.2):
            return "categorical"
        return "text"

    @staticmethod
    def _average(name: str) -> bool:
        return any(
            word in name.lower()
            for word in ("price", "rate", "percent", "margin", "average", "avg", "score")
        )

    @staticmethod
    def _grain(dates: pd.Series) -> tuple[str, str]:
        days = int((dates.max() - dates.min()).days)
        if days > 730:
            return "year", "Y"
        if days > 120:
            return "month", "M"
        if days > 30:
            return "week", "W"
        return "day", "D"

    @staticmethod
    def _overlap_lines(lines: list[str]) -> list[str]:
        overlap: list[str] = []
        total = 0
        for line in reversed(lines):
            if total + len(line) > _CHUNKING_POLICY.overlap:
                break
            overlap.insert(0, line)
            total += len(line) + 1
        return overlap

    @staticmethod
    def _unique_keep_order(values: Iterable[str | None]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            key = str(value)
            if key not in seen:
                output.append(key)
                seen.add(key)
        return output

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
        return slug or "source"

    @staticmethod
    def _number(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(number):
            return f"{number:,.2f}"
        return str(value)

    @staticmethod
    def _value(value: object) -> str:
        if pd.isna(value):
            return "missing"
        text = str(value)
        return text[:120] + "..." if len(text) > 120 else text
