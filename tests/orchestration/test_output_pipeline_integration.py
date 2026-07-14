"""Integration coverage for synthesis, dashboard, and retrieval output branches."""
from __future__ import annotations

import asyncio

from app.agents.multi import dashboard_generation_agent as dashboard_module
from app.agents.multi import insight_synthesis_agent as synthesis_module
from app.agents.multi.dashboard_generation_agent import DashboardLayoutPlan, DashboardSection
from app.agents.multi.insight_synthesis_agent import (
    EvidenceReference,
    InsightSynthesisOutput,
    Recommendation,
    SynthesisedInsight,
)
from app.agents.multi.retrieval_preparation_agent import retrieval_preparation_agent
from app.orchestration import business_intelligence_graph as graph_module


PREPARED = {
    "session_id": "output-test",
    "file_name": "sales.csv",
    "date_column": "month",
    "primary_measures": ["revenue"],
    "dimension_candidates": ["branch"],
    "dataset_profile": {
        "row_count": 24,
        "column_count": 3,
        "column_profiles": [
            {"name": "month", "null_count": 0, "date_minimum": "2025-01-01", "date_maximum": "2025-03-01"},
            {"name": "revenue", "null_count": 0},
        ],
    },
}
KPI = {"status": "complete", "kpis": [{"id": "kpi_total_revenue", "title": "Total Revenue", "value": 1200.0, "raw_value": 1200.0, "aggregation": "sum", "measure": "revenue", "dimension": None}], "trends": [{"id": "trend_monthly_revenue", "title": "Monthly Revenue", "measure": "revenue", "aggregation": "sum", "granularity": "month", "group": None, "points": [{"period": "2025-01", "value": 400.0}, {"period": "2025-02", "value": 500.0}, {"period": "2025-03", "value": 300.0}]}], "warnings": [], "limitations": []}
ANOMALY = {"status": "complete", "anomalies": [{"id": "anomaly_revenue_2025_03", "analysis_id": "revenue_change", "metric": "revenue", "period": "2025-03", "observed_value": 300.0, "expected_value": 500.0, "deviation_percentage": -40.0, "anomaly_score": None, "severity": "critical", "method": "percentage_change", "evidence": "Observed 300.0 versus expected 500.0"}], "warnings": [], "limitations": []}
FORECAST = {"status": "complete", "series_id": "forecast_monthly_revenue", "title": "Revenue Forecast", "measure": "revenue", "aggregation": "sum", "granularity": "month", "horizon": 2, "historical": [{"period": "2025-03", "value": 300.0}], "forecast": [{"period": "2025-04", "value": 350.0, "lower_bound": 320.0, "upper_bound": 380.0}, {"period": "2025-05", "value": 375.0, "lower_bound": 330.0, "upper_bound": 410.0}], "warnings": [], "limitations": []}


async def _synthesis(_payload):
    return InsightSynthesisOutput(executive_summary="Revenue results include a reported March anomaly.", key_insights=[SynthesisedInsight(id="insight_revenue", title="Revenue anomaly", description="March revenue was reported below the specialist expected value.", importance="high", evidence=[EvidenceReference(source_type="anomaly", source_id="anomaly_revenue_2025_03")])], recommendations=[Recommendation(id="recommendation_review_revenue", title="Review March revenue", description="Review the reported March revenue result before taking action.", priority="high", evidence=[EvidenceReference(source_type="anomaly", source_id="anomaly_revenue_2025_03")])])


async def _layout(_payload):
    return DashboardLayoutPlan(title="Sales performance", selected_kpi_ids=["kpi_total_revenue"], selected_trend_ids=["trend_monthly_revenue"], selected_anomaly_ids=["anomaly_revenue_2025_03"], selected_insight_ids=["insight_revenue"], selected_recommendation_ids=["recommendation_review_revenue"], include_forecast=True, section_order=[DashboardSection(id="kpis", chart_type="table"), DashboardSection(id="timeline", chart_type="line"), DashboardSection(id="details", chart_type="table")])


def _graph(monkeypatch, selected: list[str], forecasting: bool = True):
    async def cleaning(_state):
        return {"completed_agents": ["generic_cleaning"]}
    async def preparation(_state):
        return {"prepared_dataset": PREPARED, "completed_agents": ["data_preparation"]}
    async def orchestrator(_state):
        return {"orchestration_plan": {"selected_agents": selected}, "completed_agents": ["orchestrator"]}
    async def kpi(_state):
        return {"kpi_trend_output": KPI, "completed_agents": ["kpi_trend"]}
    async def anomaly(_state):
        return {"anomaly_output": ANOMALY, "completed_agents": ["anomaly_detection"]}
    async def forecast(_state):
        return {"forecasting_output": FORECAST, "completed_agents": ["forecasting"]}
    monkeypatch.setattr(graph_module, "kpi_trend_node", kpi)
    monkeypatch.setattr(graph_module, "anomaly_detection_node", anomaly)
    monkeypatch.setattr(graph_module, "forecasting_node", forecast)
    monkeypatch.setattr(synthesis_module, "_request_groq_synthesis", _synthesis)
    monkeypatch.setattr(dashboard_module, "_request_groq_layout", _layout)
    return graph_module.build_business_intelligence_graph(generic_cleaning_node_fn=cleaning, data_preparation_node_fn=preparation, orchestrator_node_fn=orchestrator)


def _invoke(graph):
    return asyncio.run(graph.ainvoke({"session_id": "output-test"}))


def test_complete_outputs_flow_to_both_parallel_branches(monkeypatch) -> None:
    result = _invoke(_graph(monkeypatch, ["kpi_trend", "anomaly_detection", "forecasting"]))
    completed = set(result["completed_agents"])
    assert {"insight_synthesis", "dashboard_generation", "retrieval_preparation", "output_join"} <= completed
    assert result["synthesis_output"] and result["dashboard_output"] and result["retrieval_documents"]
    assert result["synthesis_output"]["key_insights"][0]["evidence"][0]["source_id"] == "anomaly_revenue_2025_03"
    dashboard = result["dashboard_output"]["dashboard"]
    assert dashboard["kpis"][0]["id"] == "kpi_total_revenue"
    assert dashboard["kpis"][0]["rawValue"] == KPI["kpis"][0]["raw_value"]
    assert dashboard["timeline"]["forecast"][0]["value"] == FORECAST["forecast"][0]["value"]
    for point_type in ("actual", "anomalies", "forecast"):
        for point in dashboard["timeline"][point_type]:
            assert point["label"] == point["period"]


def test_missing_forecast_still_completes(monkeypatch) -> None:
    result = _invoke(_graph(monkeypatch, ["kpi_trend", "anomaly_detection"]))
    dashboard = result["dashboard_output"]["dashboard"]
    assert "forecasting_output" not in result
    assert dashboard["timeline"]["forecast"] == []
    assert result["retrieval_documents"]
    assert "output_join" in result["completed_agents"]


def test_kpi_only_produces_limited_valid_outputs(monkeypatch) -> None:
    result = _invoke(_graph(monkeypatch, ["kpi_trend"]))
    dashboard = result["dashboard_output"]["dashboard"]
    document_types = {document["document_type"] for document in result["retrieval_documents"]}
    assert result["synthesis_output"]["status"] in {"complete", "partial"}
    assert dashboard["kpis"] and dashboard["kpis"][0]["id"] == "kpi_total_revenue"
    assert {"dataset_summary", "kpi"} <= document_types
    assert "anomaly" not in document_types and "forecast" not in document_types
    assert "output_join" in result["completed_agents"]


def test_invalid_layout_ids_are_removed_and_valid_ids_remain(monkeypatch) -> None:
    async def invalid_layout(_payload):
        return DashboardLayoutPlan(title="Sales performance", selected_kpi_ids=["unknown", "kpi_total_revenue", "kpi_total_revenue"], selected_trend_ids=["unknown", "trend_monthly_revenue", "trend_monthly_revenue"], selected_anomaly_ids=["unknown"], selected_insight_ids=["unknown", "insight_revenue"], selected_recommendation_ids=["unknown", "recommendation_review_revenue"], include_forecast=True, section_order=[DashboardSection(id="kpis", chart_type="table")])
    graph = _graph(monkeypatch, ["kpi_trend", "anomaly_detection", "forecasting"])
    monkeypatch.setattr(dashboard_module, "_request_groq_layout", invalid_layout)
    result = _invoke(graph)
    plan = result["dashboard_layout_plan"]
    assert plan["selected_kpi_ids"] == ["kpi_total_revenue"]
    assert plan["selected_trend_ids"] == ["trend_monthly_revenue"]
    assert plan["selected_anomaly_ids"] == ["anomaly_revenue_2025_03"]
    assert result["dashboard_output"]["dashboard"]["kpis"]


def test_retrieval_preparation_is_deterministic() -> None:
    first = asyncio.run(retrieval_preparation_agent.run(PREPARED, KPI, ANOMALY, FORECAST, asyncio.run(_synthesis({})).model_dump(mode="json")))
    second = asyncio.run(retrieval_preparation_agent.run(PREPARED, KPI, ANOMALY, FORECAST, asyncio.run(_synthesis({})).model_dump(mode="json")))
    assert [document.id for document in first.documents] == [document.id for document in second.documents]
    assert "kpi_total_revenue" in [document.id for document in first.documents]
    kpi_document = next(document for document in first.documents if document.id == "kpi_total_revenue")
    assert kpi_document.source_ids == ["kpi_total_revenue"]
