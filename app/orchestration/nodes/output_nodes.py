"""Analysis fan-in and output node adapters."""

from typing import Any

from app.orchestration.routing import route_specialists
from app.orchestration.state import AnalysisState
from app.schemas.api import DashboardResponse


async def specialist_join_node(
    state: AnalysisState,
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
    state: AnalysisState,
) -> dict[str, Any]:
    """Confirm downstream fan-in without turning a partial output into failure."""
    expected = {"dashboard_generation", "retrieval_preparation"}
    reported = set(state.get("completed_agents", [])) | set(
        state.get("failed_agents", [])
    )
    update: dict[str, Any] = {
        "completed_agents": ["output_join"],
        "workflow_status": _workflow_status(state),
    }
    missing = expected - reported
    failed = expected & set(state.get("failed_agents", []))
    if missing or failed:
        names = sorted(missing | failed)
        update["warnings"] = [
            f"Output branch did not complete successfully: {', '.join(names)}."
        ]
    return update


def _workflow_status(state: AnalysisState) -> str:
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
    if (
        meets_success_shape
        and selected_complete
        and required_complete
        and not optional_failure
    ):
        return "success"
    return "partial"
