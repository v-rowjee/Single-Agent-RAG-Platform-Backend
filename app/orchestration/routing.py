"""Deterministic routing for specialist analysis nodes."""

from __future__ import annotations

from .state import AnalysisState


REGISTERED_SPECIALISTS = {
    "kpi_trend",
    "anomaly_detection",
    "forecasting",
}

SPECIALIST_ORDER = (
    "kpi_trend",
    "anomaly_detection",
    "forecasting",
)


def route_specialists(state: AnalysisState) -> list[str]:
    """Return selected specialist nodes in their deterministic execution order."""
    plan = state.get("orchestration_plan", {})
    selected_agents = plan.get("selected_agents", []) if isinstance(plan, dict) else []
    selected = {
        agent
        for agent in selected_agents
        if isinstance(agent, str) and agent in REGISTERED_SPECIALISTS
    } if isinstance(selected_agents, list) else set()

    routes = [
        specialist
        for specialist in SPECIALIST_ORDER
        if specialist in REGISTERED_SPECIALISTS and specialist in selected
    ]
    return routes or ["specialist_join"]
