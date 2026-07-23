"""Dashboard domain schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import Granularity, IndicatorKind, Priority, Severity, StrictModel, ValueFormat


MAX_SUPPORTING_CHARTS = 4


class Indicator(BaseModel):
    kind: IndicatorKind
    text: str = Field(min_length=1)

class Kpi(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    value: str
    rawValue: str | float | int | None = None
    indicator: Indicator

class TimelinePoint(BaseModel):
    period: str = Field(min_length=1)
    label: str | None = None
    value: float

class TimelineAnomaly(TimelinePoint):
    id: str = Field(min_length=1)
    severity: Severity
    reason: str = Field(min_length=1)

class ForecastPoint(TimelinePoint):
    lowerBound: float | None = None
    upperBound: float | None = None

class ForecastMetadata(BaseModel):
    available: bool
    model: str | None
    horizon: int = Field(ge=0)
    horizonUnit: Granularity
    target: str | None
    confidenceLevel: float | None = Field(default=None, ge=0, le=1)

class Timeline(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    subtitle: str | None = None
    granularity: Granularity
    unit: str | None
    valueFormat: ValueFormat
    actual: list[TimelinePoint]
    anomalies: list[TimelineAnomaly]
    forecast: list[ForecastPoint]
    forecastMetadata: ForecastMetadata

class ChartLayout(BaseModel):
    columnSpan: int = Field(default=1, ge=1, le=2)
    rowSpan: int = Field(default=1, ge=1, le=2)

class ChartSeries(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    data: list[float]

class CategoricalChart(BaseModel):
    id: str = Field(min_length=1)
    type: Literal[
        "bar",
        "horizontalBar",
        "line",
        "area",
        "stackedBar",
    ]
    title: str = Field(min_length=1)
    subtitle: str | None = None
    valueFormat: ValueFormat
    categories: list[str]
    series: list[ChartSeries] = Field(min_length=1)
    layout: ChartLayout

class Segment(BaseModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float

class ProportionalChart(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["pie", "donut"]
    title: str = Field(min_length=1)
    subtitle: str | None = None
    valueFormat: ValueFormat
    segments: list[Segment] = Field(min_length=1)
    layout: ChartLayout

class ChartAxis(BaseModel):
    title: str = Field(min_length=1)
    format: ValueFormat

class ScatterPoint(BaseModel):
    x: float
    y: float
    label: str | None = None

class ScatterChart(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["scatter"]
    title: str = Field(min_length=1)
    subtitle: str | None = None
    valueFormat: ValueFormat
    xAxis: ChartAxis
    yAxis: ChartAxis
    points: list[ScatterPoint] = Field(min_length=1)
    layout: ChartLayout

SupportingChart = Annotated[
    CategoricalChart | ProportionalChart | ScatterChart,
    Field(discriminator="type"),
]

class InsightItem(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Severity
    sourceIds: list[str] = Field(default_factory=list)

class RecommendedAction(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: Priority
    sourceIds: list[str] = Field(default_factory=list)

class DatasetPeriod(BaseModel):
    start: str
    end: str
    label: str

class DatasetQuality(BaseModel):
    completenessPercent: float = Field(ge=0, le=100)
    missingValueCount: int = Field(ge=0)
    duplicateRowCount: int = Field(ge=0)

class DatasetSummary(BaseModel):
    datasetId: str = ""
    fileName: str
    rowCount: int = Field(ge=0)
    columnCount: int = Field(ge=0)
    timeField: str | None
    period: DatasetPeriod | None
    measures: list[str]
    dimensions: list[str]
    quality: DatasetQuality
    generatedAt: str

class Section(BaseModel):
    id: Literal["kpis", "timeline", "supportingCharts", "details"]
    title: str
    order: int = Field(gt=0)
    visible: bool

class KpiLayout(BaseModel):
    columns: int = Field(ge=1, le=8)
    maxRows: int = Field(ge=1, le=2)

class TimelineLayout(BaseModel):
    columnSpan: Literal[12] = 12

class SupportingChartsLayout(BaseModel):
    columns: int = Field(ge=1, le=2)
    maxRows: int = Field(ge=1, le=2)

class DetailsLayout(BaseModel):
    columns: int = Field(ge=1, le=4)
    maxRows: int = Field(ge=1, le=2)

class DashboardLayout(BaseModel):
    kpis: KpiLayout
    timeline: TimelineLayout
    supportingCharts: SupportingChartsLayout
    details: DetailsLayout

class DashboardAnalysis(BaseModel):
    businessSummary: str
    keyFindings: list[str]

class DashboardInsights(BaseModel):
    criticalAnomalies: list[InsightItem]
    warnings: list[InsightItem]
    limitations: list[InsightItem]
    opportunities: list[InsightItem]

class Dashboard(BaseModel):
    title: str = Field(min_length=1)
    executiveSummary: str
    kpis: list[Kpi] = Field(max_length=8)
    timeline: Timeline | None
    supportingCharts: list[SupportingChart] = Field(max_length=4)
    analysis: DashboardAnalysis
    insights: DashboardInsights
    recommendedActions: list[RecommendedAction] = Field(
        min_length=3,
        max_length=5,
    )
    datasetSummaries: list[DatasetSummary] = Field(min_length=1)
    sections: list[Section]
    layout: DashboardLayout

    @model_validator(mode="before")
    @classmethod
    def normalize_dataset_summaries(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "datasetSummaries" not in payload and "datasetSummary" in payload:
            payload["datasetSummaries"] = [payload.pop("datasetSummary")]
        return payload

    @property
    def datasetSummary(self) -> DatasetSummary:
        """Compatibility accessor for code that still reads the first summary."""
        return self.datasetSummaries[0]


class DashboardSection(StrictModel):
    id: Literal["kpis", "timeline", "supportingCharts", "details"]
    chart_type: Literal["line", "bar", "area", "table"] | None = None

class SupportingChartSpec(StrictModel):
    id: str
    title: str
    type: Literal[
        "bar",
        "horizontalBar",
        "stackedBar",
        "donut",
        "pie",
        "scatter",
    ]
    dimension: str | None = None
    measure: str | None = None
    secondary_measure: str | None = None
    aggregation: Literal["sum", "mean"] | None = None
    x_measure: str | None = None
    y_measure: str | None = None

class DashboardLayoutPlan(StrictModel):
    title: str = Field(min_length=1)
    selected_kpi_ids: list[str] = Field(default_factory=list)
    selected_trend_ids: list[str] = Field(default_factory=list)
    selected_anomaly_ids: list[str] = Field(default_factory=list)
    selected_insight_ids: list[str] = Field(default_factory=list)
    selected_recommendation_ids: list[str] = Field(default_factory=list)
    include_forecast: bool = False
    supporting_chart_specs: list[SupportingChartSpec] = Field(
        default_factory=list,
        max_length=MAX_SUPPORTING_CHARTS,
    )
    section_order: list[DashboardSection] = Field(default_factory=list)

class DashboardGenerationOutput(StrictModel):
    status: Literal["complete", "partial"] = "partial"
    layout_plan: DashboardLayoutPlan
    dashboard: Any
    warnings: list[str] = Field(default_factory=list)
