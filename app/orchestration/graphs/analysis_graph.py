"""LangGraph foundation for the multi-agent business intelligence workflow."""

from __future__ import annotations

from typing import Mapping

from langgraph.graph import END, START, StateGraph

from app.agents.multi.anomaly_detection import anomaly_detection_node
from app.agents.multi.dashboard_generation import dashboard_generation_node
from app.agents.multi.forecasting import forecasting_node
from app.agents.multi.insight_synthesis import insight_synthesis_node
from app.agents.multi.kpi_trend import kpi_trend_node
from app.agents.multi.orchestrator import orchestrator_node
from app.agents.multi.retrieval_preparation import retrieval_preparation_node
from app.orchestration.nodes.cleaning_node import generic_cleaning_node
from app.orchestration.nodes.output_nodes import (
    output_join_node,
    specialist_join_node,
)
from app.orchestration.nodes.preparation_node import data_preparation_graph_node
from app.orchestration.nodes.specialist_nodes import StateNode, _recoverable_node
from app.orchestration.routing import route_specialists
from app.orchestration.state import AnalysisState

def build_analysis_graph(
    *,
    generic_cleaning_node_fn: StateNode | None = None,
    data_preparation_node_fn: StateNode | None = None,
    orchestrator_node_fn: StateNode | None = None,
    node_overrides: Mapping[str, StateNode] | None = None,
):
    """Build the workflow through specialist analysis and output fan-in."""
    overrides = dict(node_overrides or {})
    if generic_cleaning_node_fn is not None:
        overrides["generic_cleaning"] = generic_cleaning_node_fn
    if data_preparation_node_fn is not None:
        overrides["data_preparation"] = data_preparation_node_fn
    if orchestrator_node_fn is not None:
        overrides["orchestrator"] = orchestrator_node_fn

    def selected(name: str, default: StateNode) -> StateNode:
        return overrides.get(name, default)

    graph = StateGraph(AnalysisState)
    graph.add_node(
        "generic_cleaning",
        selected("generic_cleaning", generic_cleaning_node),
    )
    graph.add_node(
        "data_preparation",
        selected("data_preparation", data_preparation_graph_node),
    )
    graph.add_node(
        "orchestrator",
        selected("orchestrator", orchestrator_node),
    )
    graph.add_node(
        "kpi_trend",
        _recoverable_node(
            "kpi_trend",
            selected("kpi_trend", kpi_trend_node),
            empty_update={
                "kpi_trend_output": {
                    "status": "partial",
                    "kpis": [],
                    "trends": [],
                    "warnings": [],
                    "limitations": ["KPI and trend analysis failed."],
                }
            },
        ),
    )
    graph.add_node(
        "anomaly_detection",
        _recoverable_node(
            "anomaly_detection",
            selected("anomaly_detection", anomaly_detection_node),
            empty_update={
                "anomaly_output": {
                    "status": "partial",
                    "anomalies": [],
                    "warnings": [],
                    "limitations": ["Anomaly detection failed."],
                }
            },
        ),
    )
    graph.add_node(
        "forecasting",
        _recoverable_node(
            "forecasting",
            selected("forecasting", forecasting_node),
            empty_update={
                "forecasting_output": {
                    "status": "partial",
                    "historical": [],
                    "forecast": [],
                    "warnings": [],
                    "limitations": ["Forecasting failed."],
                }
            },
        ),
    )
    graph.add_node(
        "specialist_join",
        selected("specialist_join", specialist_join_node),
    )
    graph.add_node(
        "insight_synthesis",
        _recoverable_node(
            "insight_synthesis",
            selected("insight_synthesis", insight_synthesis_node),
            empty_update={
                "synthesis_output": {
                    "status": "partial",
                    "executive_summary": (
                        "Specialist outputs are available, but insight synthesis failed."
                    ),
                    "key_insights": [],
                    "recommendations": [],
                    "warnings": [],
                    "limitations": ["Insight synthesis failed."],
                }
            },
        ),
    )
    graph.add_node(
        "dashboard_generation",
        _recoverable_node(
            "dashboard_generation",
            selected("dashboard_generation", dashboard_generation_node),
            empty_update={},
            required=True,
        ),
    )
    graph.add_node(
        "retrieval_preparation",
        _recoverable_node(
            "retrieval_preparation",
            selected("retrieval_preparation", retrieval_preparation_node),
            empty_update={
                "retrieval_documents": [],
            },
        ),
    )
    graph.add_node(
        "output_join",
        selected("output_join", output_join_node),
    )

    graph.add_edge(START, "generic_cleaning")
    graph.add_edge("generic_cleaning", "data_preparation")
    graph.add_edge("data_preparation", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_specialists,
        {
            "kpi_trend": "kpi_trend",
            "anomaly_detection": "anomaly_detection",
            "forecasting": "forecasting",
            "specialist_join": "specialist_join",
        },
    )
    graph.add_edge("kpi_trend", "specialist_join")
    graph.add_edge("anomaly_detection", "specialist_join")
    graph.add_edge("forecasting", "specialist_join")
    graph.add_edge("specialist_join", "insight_synthesis")
    graph.add_edge("insight_synthesis", "dashboard_generation")
    graph.add_edge("dashboard_generation", "retrieval_preparation")
    graph.add_edge("retrieval_preparation", "output_join")
    graph.add_edge("output_join", END)
    return graph.compile()


analysis_graph = build_analysis_graph()
