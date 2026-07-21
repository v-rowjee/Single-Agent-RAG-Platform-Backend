"""Deterministically prepare compact, authoritative retrieval documents."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.rag.document_builder import DatasetDocumentBuilder

MAX_RETRIEVAL_DOCUMENTS = 50
MAX_DOCUMENT_LENGTH = 1500


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalDocument(StrictModel):
    id: str = Field(min_length=1)
    document_type: Literal["dataset_summary", "kpi", "trend", "anomaly", "forecast", "insight", "recommendation", "limitation", "row_batch"]
    title: str = Field(min_length=1)
    content: str
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalPreparationOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    documents: list[RetrievalDocument] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def documents_as_dicts(self) -> list[dict[str, Any]]:
        return [document.model_dump(mode="json") for document in self.documents]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "item"


def _short(value: str) -> str:
    return value[:MAX_DOCUMENT_LENGTH]


def _document_id(prefix: str, source_id: Any) -> str:
    value = _slug(source_id)
    return value if value.startswith(f"{prefix}_") else f"{prefix}_{value}"


def _add(documents: list[RetrievalDocument], **values: Any) -> None:
    if len(documents) < MAX_RETRIEVAL_DOCUMENTS:
        values["content"] = _short(str(values.get("content") or ""))
        documents.append(RetrievalDocument(**values))


def dashboard_retrieval_documents(
    dashboard_output: dict[str, Any] | None,
) -> list[RetrievalDocument]:
    """Turn the saved dashboard into authoritative chat evidence.

    The dashboard can add fallback recommended actions after synthesis, so it is
    the final source of truth for the recommendations shown to the user.
    """
    payload = dashboard_output if isinstance(dashboard_output, dict) else {}
    dashboard = payload.get("dashboard", payload)
    if not isinstance(dashboard, dict):
        return []

    documents: list[RetrievalDocument] = []
    summary = str(dashboard.get("executiveSummary") or "").strip()
    if summary:
        _add(
            documents,
            id="dashboard_executive_summary",
            document_type="insight",
            title="Dashboard executive summary",
            content=summary,
            source_ids=["dashboard_executive_summary"],
            metadata={"source": "dashboard"},
        )
    for item in dashboard.get("kpis", []):
        if isinstance(item, dict) and item.get("id"):
            _add(
                documents,
                id=_document_id("kpi", item["id"]),
                document_type="kpi",
                title=str(item.get("title") or item["id"]),
                content=(
                    f"{item.get('title') or item['id']}: {item.get('value')}. "
                    f"{(item.get('indicator') or {}).get('text') or ''}"
                ).strip(),
                source_ids=[str(item["id"])],
                metadata={"source": "dashboard", "raw_value": item.get("rawValue")},
            )
    for item in dashboard.get("recommendedActions", []):
        if isinstance(item, dict) and item.get("id"):
            source_ids = [str(source_id) for source_id in item.get("sourceIds") or []]
            _add(
                documents,
                id=_document_id("recommendation", item["id"]),
                document_type="recommendation",
                title=str(item.get("title") or item["id"]),
                content=(
                    f"Recommended action ({item.get('priority') or 'medium'} priority): "
                    f"{item.get('title') or item['id']}. {item.get('description') or ''}"
                ).strip(),
                source_ids=source_ids,
                metadata={"source": "dashboard", "priority": item.get("priority")},
            )
    return documents


def _raw_row_documents(
    prepared: dict[str, Any],
) -> tuple[list[RetrievalDocument], list[str]]:
    prepared_path = Path(str(prepared.get("prepared_file_path") or ""))
    if not prepared_path.is_file():
        return [], ["Prepared rows were unavailable for retrieval indexing."]

    try:
        dataframe = pd.read_csv(prepared_path, low_memory=False)
        builder = DatasetDocumentBuilder()
        raw_documents = builder.build_row_documents(
            df=dataframe,
            session_id=str(prepared.get("session_id") or "prepared_dataset"),
            file_name=str(prepared.get("file_name") or prepared_path.name),
            measures=[str(value) for value in prepared.get("primary_measures") or []],
            dimensions=[
                str(value) for value in prepared.get("dimension_candidates") or []
            ],
            date_field=(
                str(prepared["date_column"])
                if prepared.get("date_column")
                else None
            ),
        )
    except Exception as exc:
        return [], [f"Prepared rows could not be indexed: {exc}"]

    output: list[RetrievalDocument] = []
    for index, document in enumerate(raw_documents):
        metadata = dict(document.metadata)
        source_id = str(metadata.get("source_id") or f"row_batch_{index}")
        row_start = metadata.get("row_start")
        row_end = metadata.get("row_end")
        title = (
            f"Prepared rows {row_start} to {row_end}"
            if row_start is not None and row_end is not None
            else f"Prepared row batch {index + 1}"
        )
        output.append(
            RetrievalDocument(
                id=source_id,
                document_type="row_batch",
                title=title,
                content=document.page_content,
                source_ids=[source_id],
                metadata={**metadata, "title": title},
            )
        )
    return output, []


class RetrievalPreparationAgent:
    async def run(self, prepared_dataset: dict[str, Any], kpi_trend_output: dict[str, Any] | None, anomaly_output: dict[str, Any] | None, forecasting_output: dict[str, Any] | None, synthesis_output: dict[str, Any], dashboard_output: dict[str, Any] | None = None) -> RetrievalPreparationOutput:
        prepared = prepared_dataset if isinstance(prepared_dataset, dict) else {}
        kpi, anomaly, forecast = kpi_trend_output or {}, anomaly_output or {}, forecasting_output or {}
        synthesis = synthesis_output if isinstance(synthesis_output, dict) else {}
        profile = prepared.get("dataset_profile") or {}
        currency = profile.get("currency") or prepared.get("currency")
        documents: list[RetrievalDocument] = []
        _add(documents, id="dataset_summary", document_type="dataset_summary", title="Dataset summary", content=f"Prepared dataset has {profile.get('row_count', 0)} rows and {profile.get('column_count', 0)} columns. Currency: {currency or 'not identified'}. Measures: {', '.join(prepared.get('primary_measures') or []) or 'not identified'}. Date column: {prepared.get('date_column') or 'not identified'}.", source_ids=["dataset_summary"], metadata={"row_count": profile.get("row_count"), "column_count": profile.get("column_count"), "currency": currency, "measures": prepared.get("primary_measures") or [], "date_column": prepared.get("date_column")})
        for item in kpi.get("kpis", []):
            if item.get("id"):
                _add(documents, id=_document_id("kpi", item["id"]), document_type="kpi", title=str(item.get("title") or item["id"]), content=f"{item.get('title') or item['id']}: {item.get('value')} ({item.get('aggregation')} of {item.get('measure')}; currency: {currency or 'not identified'}).", source_ids=[str(item["id"])], metadata={"aggregation": item.get("aggregation"), "measure": item.get("measure"), "value": item.get("value"), "raw_value": item.get("raw_value"), "currency": currency})
        for item in kpi.get("trends", []):
            if item.get("id"):
                points = item.get("points") or []
                first, last = (points[0], points[-1]) if points else ({}, {})
                _add(documents, id=_document_id("trend", item["id"]), document_type="trend", title=str(item.get("title") or item["id"]), content=f"{item.get('title') or item['id']} is a {item.get('granularity')} {item.get('aggregation')} trend for {item.get('measure')}. Period range: {first.get('period', 'not available')} to {last.get('period', 'not available')}. First value: {first.get('value', 'not available')}; last value: {last.get('value', 'not available')}.", source_ids=[str(item["id"])], metadata={"measure": item.get("measure"), "aggregation": item.get("aggregation"), "granularity": item.get("granularity"), "point_count": len(points), "start_period": first.get("period"), "end_period": last.get("period")})
        for item in anomaly.get("anomalies", []):
            if item.get("id"):
                _add(documents, id=_document_id("anomaly", item["id"]), document_type="anomaly", title=f"Anomaly: {item.get('metric') or item['id']}", content=str(item.get("evidence") or f"Observed value {item.get('observed_value')} in {item.get('period') or 'the analysed period'}.").strip(), source_ids=[str(item["id"])], metadata={"metric": item.get("metric"), "period": item.get("period"), "severity": item.get("severity"), "method": item.get("method"), "observed_value": item.get("observed_value"), "expected_value": item.get("expected_value"), "deviation_percentage": item.get("deviation_percentage"), "anomaly_score": item.get("anomaly_score")})
        if forecast.get("series_id") and forecast.get("forecast"):
            points = forecast.get("forecast") or []
            _add(documents, id=_document_id("forecast", forecast["series_id"]), document_type="forecast", title=str(forecast.get("title") or "Forecast"), content=f"Forecast for {forecast.get('measure') or 'the selected measure'} contains {len(points)} {forecast.get('granularity') or ''} points from {points[0].get('period') if points else 'not available'} to {points[-1].get('period') if points else 'not available'}. Currency: {currency or 'not identified'}. First forecast value: {points[0].get('value') if points else 'not available'}; last forecast value: {points[-1].get('value') if points else 'not available'}.", source_ids=[str(forecast["series_id"])], metadata={"measure": forecast.get("measure"), "granularity": forecast.get("granularity"), "horizon": forecast.get("horizon"), "point_count": len(points), "currency": currency})
        for item in synthesis.get("key_insights", []):
            if item.get("id"):
                source_ids = [str(ref.get("source_id")) for ref in item.get("evidence", []) if ref.get("source_id")]
                _add(documents, id=_document_id("insight", item["id"]), document_type="insight", title=str(item.get("title") or item["id"]), content=str(item.get("description") or ""), source_ids=source_ids, metadata={"importance": item.get("importance")})
        for item in synthesis.get("recommendations", []):
            if item.get("id"):
                source_ids = [str(ref.get("source_id")) for ref in item.get("evidence", []) if ref.get("source_id")]
                _add(documents, id=_document_id("recommendation", item["id"]), document_type="recommendation", title=str(item.get("title") or item["id"]), content=str(item.get("description") or ""), source_ids=source_ids, metadata={"priority": item.get("priority")})
        existing_ids = {document.id for document in documents}
        for document in dashboard_retrieval_documents(dashboard_output):
            if document.id not in existing_ids:
                _add(documents, **document.model_dump(mode="json"))
                existing_ids.add(document.id)
        limitations = list(dict.fromkeys([*(prepared.get("limitations") or []), *(kpi.get("limitations") or []), *(anomaly.get("limitations") or []), *(forecast.get("limitations") or []), *(synthesis.get("limitations") or [])]))
        if limitations:
            _add(documents, id="limitations", document_type="limitation", title="Analysis limitations", content=" ".join(str(value) for value in limitations), source_ids=["dataset_summary"], metadata={"count": len(limitations)})
        warnings = list(dict.fromkeys([*(prepared.get("warnings") or []), *(kpi.get("warnings") or []), *(anomaly.get("warnings") or []), *(forecast.get("warnings") or []), *(synthesis.get("warnings") or [])]))
        row_documents, row_warnings = _raw_row_documents(prepared)
        documents.extend(row_documents)
        warnings = list(dict.fromkeys([*warnings, *row_warnings]))
        return RetrievalPreparationOutput(status="complete" if documents else "partial", documents=documents, limitations=[str(value) for value in limitations], warnings=[str(value) for value in warnings])


retrieval_preparation_agent = RetrievalPreparationAgent()


async def retrieval_preparation_node(state: dict[str, Any]) -> dict[str, Any]:
    result = await retrieval_preparation_agent.run(state.get("prepared_dataset", {}), state.get("kpi_trend_output"), state.get("anomaly_output"), state.get("forecasting_output"), state.get("synthesis_output", {}), state.get("dashboard_output"))
    return {"retrieval_documents": result.documents_as_dicts(), "retrieval_output": result.model_dump(mode="json"), "completed_agents": ["retrieval_preparation"]}
