"""Deterministically prepare compact, authoritative retrieval documents."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_RETRIEVAL_DOCUMENTS = 50
MAX_DOCUMENT_LENGTH = 1500


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalDocument(StrictModel):
    id: str = Field(min_length=1)
    document_type: Literal["dataset_summary", "kpi", "trend", "anomaly", "forecast", "insight", "recommendation", "limitation"]
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


class RetrievalPreparationAgent:
    async def run(self, prepared_dataset: dict[str, Any], kpi_trend_output: dict[str, Any] | None, anomaly_output: dict[str, Any] | None, forecasting_output: dict[str, Any] | None, synthesis_output: dict[str, Any]) -> RetrievalPreparationOutput:
        prepared = prepared_dataset if isinstance(prepared_dataset, dict) else {}
        kpi, anomaly, forecast = kpi_trend_output or {}, anomaly_output or {}, forecasting_output or {}
        synthesis = synthesis_output if isinstance(synthesis_output, dict) else {}
        profile = prepared.get("dataset_profile") or {}
        documents: list[RetrievalDocument] = []
        _add(documents, id="dataset_summary", document_type="dataset_summary", title="Dataset summary", content=f"Prepared dataset has {profile.get('row_count', 0)} rows and {profile.get('column_count', 0)} columns. Measures: {', '.join(prepared.get('primary_measures') or []) or 'not identified'}. Date column: {prepared.get('date_column') or 'not identified'}.", source_ids=["dataset_summary"], metadata={"row_count": profile.get("row_count"), "column_count": profile.get("column_count"), "measures": prepared.get("primary_measures") or [], "date_column": prepared.get("date_column")})
        for item in kpi.get("kpis", []):
            if item.get("id"):
                _add(documents, id=_document_id("kpi", item["id"]), document_type="kpi", title=str(item.get("title") or item["id"]), content=f"{item.get('title') or item['id']}: {item.get('value')} ({item.get('aggregation')} of {item.get('measure')}).", source_ids=[str(item["id"])], metadata={"aggregation": item.get("aggregation"), "measure": item.get("measure"), "value": item.get("value"), "raw_value": item.get("raw_value")})
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
            _add(documents, id=_document_id("forecast", forecast["series_id"]), document_type="forecast", title=str(forecast.get("title") or "Forecast"), content=f"Forecast for {forecast.get('measure') or 'the selected measure'} contains {len(points)} {forecast.get('granularity') or ''} points from {points[0].get('period') if points else 'not available'} to {points[-1].get('period') if points else 'not available'}. First forecast value: {points[0].get('value') if points else 'not available'}; last forecast value: {points[-1].get('value') if points else 'not available'}.", source_ids=[str(forecast["series_id"])], metadata={"measure": forecast.get("measure"), "granularity": forecast.get("granularity"), "horizon": forecast.get("horizon"), "point_count": len(points)})
        for item in synthesis.get("key_insights", []):
            if item.get("id"):
                source_ids = [str(ref.get("source_id")) for ref in item.get("evidence", []) if ref.get("source_id")]
                _add(documents, id=_document_id("insight", item["id"]), document_type="insight", title=str(item.get("title") or item["id"]), content=str(item.get("description") or ""), source_ids=source_ids, metadata={"importance": item.get("importance")})
        for item in synthesis.get("recommendations", []):
            if item.get("id"):
                source_ids = [str(ref.get("source_id")) for ref in item.get("evidence", []) if ref.get("source_id")]
                _add(documents, id=_document_id("recommendation", item["id"]), document_type="recommendation", title=str(item.get("title") or item["id"]), content=str(item.get("description") or ""), source_ids=source_ids, metadata={"priority": item.get("priority")})
        limitations = list(dict.fromkeys([*(prepared.get("limitations") or []), *(kpi.get("limitations") or []), *(anomaly.get("limitations") or []), *(forecast.get("limitations") or []), *(synthesis.get("limitations") or [])]))
        if limitations:
            _add(documents, id="limitations", document_type="limitation", title="Analysis limitations", content=" ".join(str(value) for value in limitations), source_ids=["dataset_summary"], metadata={"count": len(limitations)})
        warnings = list(dict.fromkeys([*(prepared.get("warnings") or []), *(kpi.get("warnings") or []), *(anomaly.get("warnings") or []), *(forecast.get("warnings") or []), *(synthesis.get("warnings") or [])]))
        return RetrievalPreparationOutput(status="complete" if documents else "partial", documents=documents, limitations=[str(value) for value in limitations], warnings=[str(value) for value in warnings])


retrieval_preparation_agent = RetrievalPreparationAgent()


async def retrieval_preparation_node(state: dict[str, Any]) -> dict[str, Any]:
    result = await retrieval_preparation_agent.run(state.get("prepared_dataset", {}), state.get("kpi_trend_output"), state.get("anomaly_output"), state.get("forecasting_output"), state.get("synthesis_output", {}))
    return {"retrieval_documents": result.documents_as_dicts(), "retrieval_output": result.model_dump(mode="json"), "completed_agents": ["retrieval_preparation"]}
