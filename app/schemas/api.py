"""HTTP and application-boundary schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import DashboardStatus
from app.schemas.dashboard import Dashboard


class BusinessIntelligenceAgentInput(BaseModel):
    sessionId: str
    datasetId: str | None = None
    filePath: str
    fileName: str
    description: str | None = None

class ApiMessage(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    component: str | None = None
    recoverable: bool | None = None

class AgentModelMetadata(BaseModel):
    """A configured model assignment safe for public API responses."""
    agent: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None = None


class AgentModelUsage(AgentModelMetadata):
    """A model assignment and its dashboard execution outcome."""

    executionStatus: Literal["succeeded", "fallback", "configured"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    failureReason: str | None = Field(
        default=None,
        max_length=300,
        exclude_if=lambda value: value is None,
    )

class DashboardResponse(BaseModel):
    status: DashboardStatus
    sessionId: str = Field(min_length=1)
    dashboard: Dashboard | None
    warnings: list[ApiMessage] = Field(default_factory=list)
    errors: list[ApiMessage] = Field(default_factory=list)
    pipelineMode: Literal["single", "multi"] | None = None
    model: str | None = None
    agentModels: list[AgentModelUsage] = Field(default_factory=list)
    chatAgent: AgentModelMetadata | None = None

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

class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=50)
    total_rows: int = Field(ge=0)
    total_pages: int = Field(ge=0)

class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1)
    query: str = Field(min_length=1)

class ChatResponse(BaseModel):
    answer: str = Field(min_length=1)
    grounding: str = Field(min_length=1)
    agentMetadata: AgentModelMetadata


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    grounded: bool = False
    createdAt: str
    agentMetadata: AgentModelMetadata | None = None


class ChatHistoryResponse(BaseModel):
    sessionId: str
    messages: list[ChatMessage]


class UploadCandidate(BaseModel):
    """Framework-neutral upload content passed into application services."""

    file_name: str
    content_type: str
    content: bytes
