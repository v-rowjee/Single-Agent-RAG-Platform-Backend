"""LangGraph foundation for the multi-agent business intelligence workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from app.agents.multi.data_preparation_agent import (
    _generic_clean_csv,
    data_preparation_node,
)
from app.agents.multi.orchestrator_agent import orchestrator_node
from app.agents.multi.kpi_trend_agent import kpi_trend_node
from app.agents.multi.anomaly_detection_agent import anomaly_detection_node
from app.agents.multi.forecasting_agent import forecasting_node
from app.agents.multi.insight_synthesis_agent import (
    insight_synthesis_node,
)
from app.agents.multi.dashboard_generation_agent import (
    dashboard_generation_node,
)
from app.agents.multi.retrieval_preparation_agent import (
    retrieval_preparation_node,
)
from app.schemas.business_intelligence import DashboardResponse

from .routing import route_specialists
from .state import BusinessIntelligenceState


logger = logging.getLogger(__name__)

StateNode = Callable[
    [BusinessIntelligenceState],
    Awaitable[dict[str, Any]],
]


async def generic_cleaning_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    """Adapt the existing generic cleaner for use as a LangGraph node."""
    uploaded_file_path = str(state.get("uploaded_file_path", "")).strip()
    working_directory = str(state.get("working_directory") or "").strip()
    if not working_directory:
        raise RuntimeError("state.working_directory is required.")

    _, report = _generic_clean_csv(
        uploaded_file_path,
        Path(working_directory),
    )

    return {
        "generic_cleaned_file_path": report.cleaned_file_path,
        "generic_cleaning_report": report.model_dump(mode="json"),
        "warnings": report.warnings,
        "completed_agents": ["generic_cleaning"],
    }


async def data_preparation_graph_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    """Run the existing preparation node against the generic-cleaned dataset."""
    node_state = dict(state)
    node_state["file_name"] = node_state.get("file_name") or str(
        state.get("uploaded_file_path", "")
    ).replace("\\", "/").rsplit("/", 1)[-1]
    node_state["uploaded_file_path"] = state.get(
        "generic_cleaned_file_path",
        state.get("uploaded_file_path", ""),
    )
    update = await data_preparation_node(node_state)
    update.pop("generic_cleaned_file_path", None)
    update.pop("generic_cleaning_report", None)
    return update


async def specialist_join_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    selected = set(route_specialists(state)) & {
        "kpi_trend",
        "anomaly_detection",
        "forecasting",
    }
    completed = set(state.get("completed_agents", []))
    failed = set(state.get("failed_agents", []))
    missing = selected - completed - failed

    update: dict[str, Any] = {"completed_agents": ["specialist_join"]}
    if missing:
        update["warnings"] = [
            f"Selected specialists did not report completion: {', '.join(sorted(missing))}."
        ]
    return update


async def output_join_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    """Confirm downstream fan-in without turning a partial output into failure."""
    expected = {"dashboard_generation", "retrieval_preparation"}
    reported = set(state.get("completed_agents", [])) | set(state.get("failed_agents", []))
    update: dict[str, Any] = {
        "completed_agents": ["output_join"],
        "workflow_status": _workflow_status(state),
    }
    missing = expected - reported
    failed = expected & set(state.get("failed_agents", []))
    if missing or failed:
        names = sorted(missing | failed)
        update["warnings"] = [f"Output branch did not complete successfully: {', '.join(names)}."]
    return update


def _workflow_status(state: BusinessIntelligenceState) -> str:
    if "data_preparation" in set(state.get("failed_agents", [])):
        return "failed"
    try:
        response = DashboardResponse.model_validate(state.get("dashboard_output"))
    except Exception:
        return "failed"
    if response.dashboard is None:
        return "failed"

    dashboard = response.dashboard
    selected = set(
        (state.get("orchestration_plan") or {}).get("selected_agents", [])
    )
    failed = set(state.get("failed_agents", []))
    completed = set(state.get("completed_agents", []))
    chart_types = [chart.type for chart in dashboard.supportingCharts]
    meets_success_shape = (
        4 <= len(dashboard.kpis) <= 8
        and 2 <= len(dashboard.supportingCharts) <= 4
        and len(chart_types) == len(set(chart_types))
    )
    optional_failure = bool(
        ({"anomaly_detection", "forecasting"} & selected & failed)
        or "kpi_trend" in failed
        or "insight_synthesis" in failed
        or "retrieval_preparation" in failed
        or ("forecasting" in selected and not (state.get("forecasting_output") or {}).get("forecast"))
    )
    selected_complete = selected <= completed
    required_complete = {
        "dashboard_generation",
        "retrieval_preparation",
    } <= completed
    if meets_success_shape and selected_complete and required_complete and not optional_failure:
        return "success"
    return "partial"


def _recoverable_node(
    name: str,
    node: StateNode,
    *,
    empty_update: Mapping[str, Any],
    required: bool = False,
) -> StateNode:
    """Keep optional branch failures in state so graph fan-in can still finish."""

    async def run(state: BusinessIntelligenceState) -> dict[str, Any]:
        try:
            return await node(state)
        except Exception as exc:
            logger.exception("Multi-agent node failed node=%s", name)
            message = f"{name.replace('_', ' ').title()} failed: {exc}"
            update = dict(empty_update)
            update["failed_agents"] = [name]
            update["errors" if required else "warnings"] = [message]
            return update

    return run


def build_business_intelligence_graph(
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

    graph = StateGraph(BusinessIntelligenceState)
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
                "retrieval_output": {
                    "status": "partial",
                    "documents": [],
                    "warnings": [],
                    "limitations": ["Retrieval preparation failed."],
                },
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
    graph.add_edge("insight_synthesis", "retrieval_preparation")
    graph.add_edge(
        ["dashboard_generation", "retrieval_preparation"],
        "output_join",
    )
    graph.add_edge("output_join", END)
    return graph.compile()


business_intelligence_graph = build_business_intelligence_graph()
