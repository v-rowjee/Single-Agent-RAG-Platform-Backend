"""Public, configuration-derived model metadata for dashboard responses."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from app.core.config import get_runtime_config


class ModelUsage(TypedDict):
    """One agent/model assignment that can safely be sent to the client."""

    agent: str
    model: str
    provider: str


_MULTI_AGENT_LABELS = {
    "data_preparation": "Data preparation",
    "orchestrator": "Orchestrator",
    "kpi_trend": "KPI and trend analysis",
    "anomaly_detection": "Anomaly detection",
    "forecasting": "Forecasting",
    "insight_synthesis": "Insight synthesis",
    "dashboard_generation": "Dashboard generation",
}

_MULTI_POLICY_KEYS = {
    "data_preparation": "data_preparation",
    "orchestrator": "orchestrator",
    "kpi_trend": "kpi_trend",
    "anomaly_detection": "anomaly_detection",
    "insight_synthesis": "insight_synthesis",
    "dashboard_generation": "dashboard_generation",
}


def single_dashboard_model_usage() -> list[ModelUsage]:
    """Return the configured model that generated a single-agent dashboard."""
    policy = get_runtime_config().agents["single_dashboard"]
    return [
        {
            "agent": "Business intelligence",
            "model": policy.model,
            "provider": policy.provider,
        }
    ]


def chat_model_usage(pipeline_mode: str) -> ModelUsage:
    """Return the configured model used to answer chat questions."""
    policy_name = "single_chat" if pipeline_mode == "single" else "chat"
    policy = get_runtime_config().agents[policy_name]
    return {
        "agent": "Chat assistant",
        "model": policy.model,
        "provider": policy.provider,
    }


def multi_dashboard_model_usage(
    selected_agents: Iterable[str],
) -> list[ModelUsage]:
    """Return models for the agents selected for one multi-agent dashboard.

    Preparation, orchestration, synthesis, and dashboard generation are always
    part of the workflow. Specialist models are included only when the
    orchestrator has selected the corresponding specialist.
    """
    selected = set(selected_agents)
    configured_agents = ["data_preparation", "orchestrator"]
    configured_agents.extend(
        agent
        for agent in ("kpi_trend", "anomaly_detection", "forecasting")
        if agent in selected
    )
    configured_agents.extend(["insight_synthesis", "dashboard_generation"])

    runtime = get_runtime_config()
    usage: list[ModelUsage] = []
    for agent in configured_agents:
        if agent == "forecasting":
            usage.append(
                {
                    "agent": _MULTI_AGENT_LABELS[agent],
                    "model": runtime.forecasting.model,
                    "provider": "engine",
                }
            )
            continue

        policy = runtime.agents[_MULTI_POLICY_KEYS[agent]]
        usage.append(
            {
                "agent": _MULTI_AGENT_LABELS[agent],
                "model": policy.model,
                "provider": policy.provider,
            }
        )

    return usage
