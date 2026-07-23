"""Structured plans and outputs for specialist agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import StrictModel

MAX_KPIS = 8
MAX_TRENDS = 3
MAX_ANALYSES = 3
DEFAULT_HORIZON = 3
MAX_FORECAST_HORIZON = 24
MIN_RECOMMENDATIONS = 3
MAX_RECOMMENDATIONS = 5

class KPIDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    dimension: str | None = None
    dimension_value: str | int | float | bool | None = None

class TrendDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    date_column: str
    granularity: str
    group_by: str | None = None

class KPITrendPlan(StrictModel):
    kpis: list[KPIDefinition] = Field(default_factory=list, max_length=MAX_KPIS)
    trends: list[TrendDefinition] = Field(default_factory=list, max_length=MAX_TRENDS)
    limitations: list[str] = Field(default_factory=list)

class KPIResult(StrictModel):
    id: str
    title: str
    value: float | int
    raw_value: float | int
    aggregation: str
    measure: str
    dimension: str | None = None
    current_period: str | None = None
    previous_period: str | None = None
    previous_value: float | int | None = None
    change_percent: float | None = None
    baseline_period: str | None = None
    baseline_value: float | int | None = None
    baseline_change_percent: float | None = None

class TrendPoint(StrictModel):
    period: str
    value: float | int

class TrendSeries(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    granularity: str
    group: str | None = None
    points: list[TrendPoint] = Field(default_factory=list)

class KPITrendOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    kpis: list[KPIResult] = Field(default_factory=list)
    trends: list[TrendSeries] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

class AnomalyDefinition(StrictModel):
    id: str
    measure: str
    method: str
    aggregation: str = "sum"
    date_column: str | None = None
    granularity: str | None = None
    group_by: str | None = None

class AnomalyPlan(StrictModel):
    analyses: list[AnomalyDefinition] = Field(default_factory=list, max_length=MAX_ANALYSES)
    limitations: list[str] = Field(default_factory=list)

class AnomalyResult(StrictModel):
    id: str
    analysis_id: str
    metric: str
    aggregation: str
    granularity: str | None = None
    period: str | None = None
    observed_value: float
    expected_value: float | None = None
    deviation_percentage: float | None = None
    anomaly_score: float | None = None
    severity: Literal["informational", "warning", "critical"]
    method: str
    evidence: str

class AnomalyDetectionOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    anomalies: list[AnomalyResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ForecastDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    date_column: str
    granularity: str
    horizon: int = Field(default=DEFAULT_HORIZON, ge=1, le=MAX_FORECAST_HORIZON)
    group_by: str | None = None
    group_value: str | int | float | bool | None = None

class ForecastPlan(StrictModel):
    forecast: ForecastDefinition
    limitations: list[str] = Field(default_factory=list)

class HistoricalPoint(StrictModel):
    period: str
    value: float

class ForecastPoint(StrictModel):
    period: str
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None

class ForecastingOutput(StrictModel):
    status: Literal["complete", "partial"] = "partial"
    series_id: str | None = None
    title: str | None = None
    measure: str | None = None
    aggregation: str | None = None
    granularity: str | None = None
    horizon: int | None = None
    model: str | None = None
    confidence_level: float | None = None
    historical: list[HistoricalPoint] = Field(default_factory=list)
    forecast: list[ForecastPoint] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class EvidenceReference(StrictModel):
    source_type: Literal["kpi", "trend", "anomaly", "forecast", "dataset"]
    source_id: str = Field(min_length=1)

class SynthesisedInsight(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    importance: Literal["low", "medium", "high"]
    evidence: list[EvidenceReference] = Field(default_factory=list)

class Recommendation(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: Literal["low", "medium", "high"]
    evidence: list[EvidenceReference] = Field(default_factory=list)

class InsightSynthesisOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    executive_summary: str
    key_insights: list[SynthesisedInsight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(
        min_length=MIN_RECOMMENDATIONS,
        max_length=MAX_RECOMMENDATIONS,
    )
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

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

class GroundedChatDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    insufficient_context: bool
