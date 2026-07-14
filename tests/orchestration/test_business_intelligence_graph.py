from __future__ import annotations

import asyncio

from app.orchestration.business_intelligence_graph import (
    build_business_intelligence_graph,
)


def build_test_graph(plan: dict[str, object]):
    async def cleaning_node(state):
        return {
            "generic_cleaned_file_path": "/tmp/generic_cleaned.csv",
            "generic_cleaning_report": {},
            "completed_agents": ["generic_cleaning"],
        }

    async def preparation_node(state):
        return {
            "prepared_file_path": "/tmp/prepared.csv",
            "prepared_dataset": {"dataset_profile": {"row_count": 3}},
            "completed_agents": ["data_preparation"],
        }

    async def orchestration_node(state):
        return {
            "orchestration_plan": plan,
            "completed_agents": ["orchestrator"],
        }

    return build_business_intelligence_graph(
        generic_cleaning_node_fn=cleaning_node,
        data_preparation_node_fn=preparation_node,
        orchestrator_node_fn=orchestration_node,
    )


def invoke(graph):
    return asyncio.run(graph.ainvoke({"session_id": "test"}))


def test_all_specialists_are_run() -> None:
    graph = build_test_graph(
        {"selected_agents": ["kpi_trend", "anomaly_detection", "forecasting"]}
    )

    result = invoke(graph)

    assert "kpi_trend_output" in result
    assert "anomaly_output" in result
    assert "forecasting_output" in result
    assert {"kpi_trend", "anomaly_detection", "forecasting", "specialist_join"} <= set(
        result["completed_agents"]
    )


def test_forecasting_is_not_run_when_not_selected() -> None:
    graph = build_test_graph(
        {"selected_agents": ["kpi_trend", "anomaly_detection"]}
    )

    result = invoke(graph)

    assert "kpi_trend_output" in result
    assert "anomaly_output" in result
    assert "forecasting_output" not in result
    assert "forecasting" not in result["completed_agents"]
    assert "specialist_join" in result["completed_agents"]


def test_only_kpi_runs_when_it_is_the_only_selection() -> None:
    graph = build_test_graph({"selected_agents": ["kpi_trend"]})

    result = invoke(graph)

    assert "kpi_trend_output" in result
    assert "anomaly_output" not in result
    assert "forecasting_output" not in result
    assert "specialist_join" in result["completed_agents"]


def test_unknown_and_duplicate_agents_are_ignored() -> None:
    graph = build_test_graph(
        {"selected_agents": ["kpi_trend", "unknown_agent", "kpi_trend"]}
    )

    result = invoke(graph)

    assert "kpi_trend_output" in result
    assert "anomaly_output" not in result
    assert "forecasting_output" not in result
    assert result["completed_agents"].count("kpi_trend") == 1


def test_no_specialists_route_straight_to_join() -> None:
    graph = build_test_graph({"selected_agents": []})

    result = invoke(graph)

    assert "kpi_trend_output" not in result
    assert "anomaly_output" not in result
    assert "forecasting_output" not in result
    assert "specialist_join" in result["completed_agents"]
