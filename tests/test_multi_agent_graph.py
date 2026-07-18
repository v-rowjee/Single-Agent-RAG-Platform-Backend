from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from app.orchestration.business_intelligence_graph import (
    build_business_intelligence_graph,
)


def _node(
    name: str,
    events: list[str],
    update: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    async def run(state: dict[str, Any]) -> dict[str, Any]:
        events.append(name)
        result: dict[str, Any] = {"completed_agents": [name]}
        result.update(update or {})
        return result

    return run


def _run_graph(selected_agents: list[str]) -> tuple[dict[str, Any], list[str]]:
    events: list[str] = []
    overrides = {
        "generic_cleaning": _node("generic_cleaning", events),
        "data_preparation": _node(
            "data_preparation",
            events,
            {"prepared_dataset": {}},
        ),
        "orchestrator": _node(
            "orchestrator",
            events,
            {"orchestration_plan": {"selected_agents": selected_agents}},
        ),
        "kpi_trend": _node(
            "kpi_trend",
            events,
            {"kpi_trend_output": {"kpis": [], "trends": []}},
        ),
        "anomaly_detection": _node(
            "anomaly_detection",
            events,
            {"anomaly_output": {"anomalies": []}},
        ),
        "forecasting": _node(
            "forecasting",
            events,
            {"forecasting_output": {"forecast": []}},
        ),
        "specialist_join": _node("specialist_join", events),
        "insight_synthesis": _node(
            "insight_synthesis",
            events,
            {"synthesis_output": {}},
        ),
        "dashboard_generation": _node(
            "dashboard_generation",
            events,
            {"dashboard_output": {}},
        ),
        "retrieval_preparation": _node(
            "retrieval_preparation",
            events,
            {"retrieval_documents": []},
        ),
        "output_join": _node("output_join", events),
    }
    graph = build_business_intelligence_graph(node_overrides=overrides)
    result = asyncio.run(
        graph.ainvoke(
            {
                "session_id": "session",
                "dataset_id": "session",
                "warnings": [],
                "errors": [],
                "completed_agents": [],
                "failed_agents": [],
                "skipped_agents": [],
            }
        )
    )
    return result, events


@pytest.mark.parametrize(
    "selected_agents",
    [
        ["kpi_trend", "anomaly_detection", "forecasting"],
        ["kpi_trend"],
        [],
    ],
)
def test_capability_routing_and_joins_execute_once(
    selected_agents: list[str],
) -> None:
    _, events = _run_graph(selected_agents)

    for specialist in ("kpi_trend", "anomaly_detection", "forecasting"):
        assert events.count(specialist) == (
            1 if specialist in selected_agents else 0
        )
    assert events.count("specialist_join") == 1
    assert events.count("insight_synthesis") == 1
    assert events.count("output_join") == 1

    specialist_positions = [
        events.index(agent) for agent in selected_agents
    ]
    if specialist_positions:
        assert max(specialist_positions) < events.index("specialist_join")
    assert events.index("specialist_join") < events.index("insight_synthesis")
    assert events.index("dashboard_generation") < events.index("output_join")
    assert events.index("retrieval_preparation") < events.index("output_join")


def test_optional_specialist_exception_reaches_output_as_failure_state() -> None:
    events: list[str] = []

    async def failing_kpi(state: dict[str, Any]) -> dict[str, Any]:
        events.append("kpi_trend")
        raise RuntimeError("specialist unavailable")

    overrides = {
        "generic_cleaning": _node("generic_cleaning", events),
        "data_preparation": _node(
            "data_preparation", events, {"prepared_dataset": {}}
        ),
        "orchestrator": _node(
            "orchestrator",
            events,
            {"orchestration_plan": {"selected_agents": ["kpi_trend"]}},
        ),
        "kpi_trend": failing_kpi,
        "specialist_join": _node("specialist_join", events),
        "insight_synthesis": _node(
            "insight_synthesis", events, {"synthesis_output": {}}
        ),
        "dashboard_generation": _node(
            "dashboard_generation", events, {"dashboard_output": {}}
        ),
        "retrieval_preparation": _node(
            "retrieval_preparation", events, {"retrieval_documents": []}
        ),
        "output_join": _node("output_join", events),
    }
    graph = build_business_intelligence_graph(node_overrides=overrides)
    result = asyncio.run(
        graph.ainvoke(
            {
                "session_id": "session",
                "dataset_id": "session",
                "warnings": [],
                "errors": [],
                "completed_agents": [],
                "failed_agents": [],
                "skipped_agents": [],
            }
        )
    )

    assert "kpi_trend" in result["failed_agents"]
    assert result["kpi_trend_output"]["status"] == "partial"
    assert events.count("specialist_join") == 1
    assert events.count("output_join") == 1

