"""Capability-gated orchestrator for the multi-agent BI workflow."""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_TREND_PERIODS = 2
MIN_FORECAST_PERIODS = 4

AgentName: TypeAlias = Literal[
    "kpi_trend",
    "anomaly_detection",
    "forecasting",
]

AGENT_ORDER: tuple[AgentName, ...] = (
    "kpi_trend",
    "anomaly_detection",
    "forecasting",
)

logger = logging.getLogger(__name__)


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator cannot read its required input."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentDecision(StrictModel):
    agent: AgentName
    selected: bool
    reason: str = Field(min_length=1, max_length=300)


class OrchestrationPlan(StrictModel):
    selected_agents: list[AgentName] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)

    @field_validator("selected_agents")
    @classmethod
    def validate_unique_agents(
        cls,
        agents: list[AgentName],
    ) -> list[AgentName]:
        if len(agents) != len(set(agents)):
            raise ValueError("selected_agents must not contain duplicates.")

        return agents


def _as_positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _get_capabilities(
    prepared_dataset: dict[str, Any],
) -> dict[str, bool]:
    """
    Read and independently validate capability flags produced by
    the Data Preparation Agent.
    """
    flags = prepared_dataset.get("capability_flags")

    if not isinstance(flags, dict):
        raise OrchestratorError(
            "prepared_dataset.capability_flags is required."
        )

    temporal_profile = prepared_dataset.get("temporal_profile")
    temporal_profile = (
        temporal_profile
        if isinstance(temporal_profile, dict)
        else {}
    )

    date_column = (
        prepared_dataset.get("date_column")
        or temporal_profile.get("date_column")
    )

    unique_periods = _as_positive_int(
        temporal_profile.get("unique_periods")
    )

    primary_measures = prepared_dataset.get("primary_measures") or []
    time_series_candidates = (
        prepared_dataset.get("time_series_candidates") or []
    )

    has_numeric_measure = bool(primary_measures)
    has_time_series_candidate = bool(time_series_candidates)

    has_temporal_data = (
        flags.get("has_temporal_data") is True
        and bool(date_column)
        and unique_periods >= MIN_TREND_PERIODS
    )

    supports_kpis = (
        flags.get("supports_kpis") is True
        and has_numeric_measure
    )

    supports_trends = (
        flags.get("supports_trends") is True
        and has_temporal_data
        and has_numeric_measure
    )

    supports_anomalies = (
        flags.get("supports_anomalies") is True
        and has_numeric_measure
    )

    supports_forecasting = (
        flags.get("supports_forecasting") is True
        and has_temporal_data
        and unique_periods >= MIN_FORECAST_PERIODS
        and (has_numeric_measure or has_time_series_candidate)
    )

    return {
        "supports_kpis": supports_kpis,
        "supports_trends": supports_trends,
        "supports_anomalies": supports_anomalies,
        "supports_forecasting": supports_forecasting,
        "has_temporal_data": has_temporal_data,
    }


def _supported_agents(
    capabilities: dict[str, bool],
) -> set[AgentName]:
    supported: set[AgentName] = set()

    if (
        capabilities["supports_kpis"]
        or capabilities["supports_trends"]
    ):
        supported.add("kpi_trend")

    if capabilities["supports_anomalies"]:
        supported.add("anomaly_detection")

    if capabilities["supports_forecasting"]:
        supported.add("forecasting")

    return supported

def _fallback_plan(
    supported_agents: set[AgentName],
) -> OrchestrationPlan:
    selected_agents = [
        agent
        for agent in AGENT_ORDER
        if agent in supported_agents
    ]

    decisions = [
        AgentDecision(
            agent=agent,
            selected=agent in supported_agents,
            reason=(
                "Selected because the prepared dataset supports this analysis."
                if agent in supported_agents
                else "Not selected because the prepared dataset does not support this analysis."
            ),
        )
        for agent in AGENT_ORDER
    ]

    return OrchestrationPlan(
        selected_agents=selected_agents,
        decisions=decisions,
    )


class OrchestratorAgent:
    async def run(
        self,
        prepared_dataset: dict[str, Any],
    ) -> OrchestrationPlan:
        if not isinstance(prepared_dataset, dict):
            raise OrchestratorError(
                "prepared_dataset must be a dictionary."
            )

        capabilities = _get_capabilities(prepared_dataset)
        supported_agents = _supported_agents(capabilities)

        logger.info(
            "Orchestration started with capabilities: %s",
            capabilities,
        )

        result = _fallback_plan(supported_agents)
        logger.info("Deterministic capability routing completed.")

        logger.info(
            "Selected specialist agents: %s",
            result.selected_agents,
        )

        return result


orchestrator_agent = OrchestratorAgent()


async def orchestrator_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    prepared_dataset = state.get("prepared_dataset")

    if not isinstance(prepared_dataset, dict):
        raise OrchestratorError(
            "state.prepared_dataset is required."
        )

    result = await orchestrator_agent.run(prepared_dataset)

    return {
        "orchestration_plan": result.model_dump(mode="json"),
        "completed_agents": ["orchestrator"],
        "skipped_agents": [
            agent for agent in AGENT_ORDER if agent not in result.selected_agents
        ],
    }
