from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


DashboardStatus = Literal["success", "partial", "failed"]
ValueFormat = Literal[
    "number",
    "currency",
    "percentage",
    "decimal",
    "text",
]
IndicatorKind = Literal["increase", "decrease", "note"]
Severity = Literal["info", "warning", "critical"]
Priority = Literal["low", "medium", "high", "critical"]
Granularity = Literal["day", "week", "month", "quarter", "year"]


class BusinessIntelligenceAgentInput(BaseModel):
    sessionId: str
    filePath: str
    fileName: str
    description: str | None = None


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
    recommendedActions: list[RecommendedAction]
    datasetSummary: DatasetSummary
    sections: list[Section]
    layout: DashboardLayout


class ApiMessage(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    component: str | None = None
    recoverable: bool | None = None


class DashboardResponse(BaseModel):
    status: DashboardStatus
    sessionId: str = Field(min_length=1)
    dashboard: Dashboard | None
    warnings: list[ApiMessage] = Field(default_factory=list)
    errors: list[ApiMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dashboard_response(self) -> "DashboardResponse":
        if self.status != "failed" and self.dashboard is None:
            raise ValueError(
                "A successful or partial response must contain a dashboard."
            )

        if self.dashboard is None:
            return self

        if self.status == "success":
            kpi_count = len(self.dashboard.kpis)

            if not 4 <= kpi_count <= 8:
                raise ValueError(
                    "A successful dashboard must contain between 4 and 8 KPIs."
                )

            chart_count = len(self.dashboard.supportingCharts)

            if not 2 <= chart_count <= 4:
                raise ValueError(
                    "A successful dashboard must contain between 2 and 4 "
                    "supporting charts."
                )

        chart_types = [
            chart.type for chart in self.dashboard.supportingCharts
        ]

        if len(chart_types) != len(set(chart_types)):
            raise ValueError(
                "Supporting charts must use different chart types."
            )

        return self


class UploadResponse(BaseModel):
    status: Literal["success"]
    sessionId: str = Field(min_length=1)
    fileName: str
    message: str


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1)
    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    response: str