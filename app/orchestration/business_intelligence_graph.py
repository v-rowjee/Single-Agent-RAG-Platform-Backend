"""LangGraph foundation for the multi-agent business intelligence workflow."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.multi.data_preparation_agent import (
    _generic_clean_csv,
    _session_dir,
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
from app.services.business_intelligence_persistence_service import (
    business_intelligence_persistence_service,
)
from app.services.retrieval_indexing_service import retrieval_indexing_service

from .routing import route_specialists
from .state import BusinessIntelligenceState


StateNode = Callable[
    [BusinessIntelligenceState],
    Awaitable[dict[str, Any]],
]


async def generic_cleaning_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    """Adapt the existing generic cleaner for use as a LangGraph node."""
    session_id = str(state.get("session_id", "")).strip()
    uploaded_file_path = str(state.get("uploaded_file_path", "")).strip()
    _, report = _generic_clean_csv(uploaded_file_path, _session_dir(session_id))

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
    expected = {"dashboard_generation", "retrieval_indexing"}
    reported = set(state.get("completed_agents", [])) | set(state.get("failed_agents", []))
    update: dict[str, Any] = {"completed_agents": ["output_join"]}
    missing = expected - reported
    failed = expected & set(state.get("failed_agents", []))
    if missing or failed:
        names = sorted(missing | failed)
        update["warnings"] = [f"Output branch did not complete successfully: {', '.join(names)}."]
    return update


async def retrieval_indexing_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    result = retrieval_indexing_service.index_documents(
        session_id=str(state.get("session_id") or ""),
        dataset_id=str(state.get("dataset_id") or state.get("session_id") or ""),
        retrieval_documents=list(state.get("retrieval_documents") or []),
    )
    if result.get("status") == "success":
        return {
            "retrieval_indexing_result": result,
            "completed_agents": ["retrieval_indexing"],
        }
    return {
        "retrieval_indexing_result": result,
        "failed_agents": ["retrieval_indexing"],
        "warnings": ["Retrieval indexing failed."],
    }


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
        or "retrieval_preparation" in failed
        or "retrieval_indexing" in failed
        or ("forecasting" in selected and not (state.get("forecasting_output") or {}).get("forecast"))
    )
    selected_complete = selected <= completed
    required_complete = {
        "dashboard_generation",
        "retrieval_preparation",
        "retrieval_indexing",
    } <= completed
    if meets_success_shape and selected_complete and required_complete and not optional_failure:
        return "success"
    return "partial"


async def persistence_node(
    state: BusinessIntelligenceState,
) -> dict[str, Any]:
    final_status = _workflow_status(state)
    try:
        response = DashboardResponse.model_validate(state.get("dashboard_output"))
        validated_response = response.model_copy(
            update={"status": final_status}
        )
        dashboard_output = DashboardResponse.model_validate(
            validated_response.model_dump(mode="json")
        ).model_dump(mode="json")
    except Exception as exc:
        return {
            "workflow_status": "failed",
            "persistence_result": {
                "status": "failed",
                "session_id": str(state.get("session_id") or ""),
                "dataset_id": str(state.get("dataset_id") or state.get("session_id") or ""),
                "message": str(exc),
            },
            "failed_agents": ["persistence"],
            "errors": ["Final workflow persistence failed."],
        }

    bundle = {
        key: state.get(key)
        for key in (
            "session_id", "dataset_id", "prepared_dataset", "orchestration_plan",
            "kpi_trend_output", "anomaly_output", "forecasting_output",
            "synthesis_output", "retrieval_documents", "retrieval_indexing_result",
            "warnings", "errors", "completed_agents", "failed_agents", "skipped_agents",
        )
    }
    bundle["workflow_status"] = final_status
    bundle["dashboard_output"] = dashboard_output
    result = business_intelligence_persistence_service.persist_workflow(bundle)
    if result.get("status") == "success":
        return {
            "workflow_status": final_status,
            "dashboard_output": dashboard_output,
            "persistence_result": result,
            "completed_agents": ["persistence"],
        }
    return {
        "workflow_status": "failed",
        "persistence_result": result,
        "failed_agents": ["persistence"],
        "errors": ["Final workflow persistence failed."],
    }


def build_business_intelligence_graph(
    *,
    generic_cleaning_node_fn: StateNode | None = None,
    data_preparation_node_fn: StateNode | None = None,
    orchestrator_node_fn: StateNode | None = None,
):
    """Build the workflow through specialist analysis and output fan-in."""
    graph = StateGraph(BusinessIntelligenceState)
    graph.add_node("generic_cleaning", generic_cleaning_node_fn or generic_cleaning_node)
    graph.add_node(
        "data_preparation",
        data_preparation_node_fn or data_preparation_graph_node,
    )
    graph.add_node("orchestrator", orchestrator_node_fn or orchestrator_node)
    graph.add_node("kpi_trend", kpi_trend_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("forecasting", forecasting_node)
    graph.add_node("specialist_join", specialist_join_node)
    graph.add_node("insight_synthesis", insight_synthesis_node)
    graph.add_node("dashboard_generation", dashboard_generation_node)
    graph.add_node("retrieval_preparation", retrieval_preparation_node)
    graph.add_node("retrieval_indexing", retrieval_indexing_node)
    graph.add_node("output_join", output_join_node)
    graph.add_node("persistence", persistence_node)

    graph.add_edge(START, "generic_cleaning")
    graph.add_edge("generic_cleaning", "data_preparation")
    graph.add_edge("data_preparation", "orchestrator")
    graph.add_conditional_edges("orchestrator", route_specialists)
    graph.add_edge("kpi_trend", "specialist_join")
    graph.add_edge("anomaly_detection", "specialist_join")
    graph.add_edge("forecasting", "specialist_join")
    graph.add_edge("specialist_join", "insight_synthesis")
    graph.add_edge("insight_synthesis", "dashboard_generation")
    graph.add_edge("insight_synthesis", "retrieval_preparation")
    graph.add_edge("dashboard_generation", "output_join")
    graph.add_edge("retrieval_preparation", "retrieval_indexing")
    graph.add_edge("retrieval_indexing", "output_join")
    graph.add_edge("output_join", "persistence")
    graph.add_edge("persistence", END)
    return graph.compile()


business_intelligence_graph = build_business_intelligence_graph()
