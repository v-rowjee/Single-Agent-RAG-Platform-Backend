"""Full graph integration tests with external model, vector, and database boundaries mocked."""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from app.agents.multi import anomaly_detection_agent as anomaly_module
from app.agents.multi import dashboard_generation_agent as dashboard_module
from app.agents.multi import data_preparation_agent as preparation_module
from app.agents.multi import forecasting_agent as forecasting_module
from app.agents.multi import insight_synthesis_agent as synthesis_module
from app.agents.multi import kpi_trend_agent as kpi_module
from app.agents.multi import orchestrator_agent as orchestrator_module
from app.agents.multi.anomaly_detection_agent import AnomalyPlan
from app.agents.multi.dashboard_generation_agent import DashboardLayoutPlan, DashboardSection
from app.agents.multi.forecasting_agent import ForecastPlan
from app.agents.multi.insight_synthesis_agent import InsightSynthesisOutput
from app.agents.multi.kpi_trend_agent import KPITrendPlan
from app.agents.multi.orchestrator_agent import OrchestrationPlan
from app.orchestration import business_intelligence_graph as graph_module
from app.schemas.business_intelligence import DashboardResponse
from app.services.timesfm_service import TimesFMForecast


def _csv(tmp_path, *, temporal: bool = True, anomaly: bool = True):
    periods = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    for index, date in enumerate(periods):
        revenue = 1000 if not anomaly else 1000 + index * 25
        if anomaly and index == 17:
            revenue = 7000
        for branch, factor in (("North", 0.6), ("South", 0.4)):
            row = {
                "revenue": revenue * factor,
                "orders": 50 + index,
                "customers": 30 + index,
                "profit": revenue * factor * 0.2,
                "branch": branch,
                "product_category": "Hardware" if index % 2 else "Software",
            }
            if temporal:
                row["transaction_date"] = date.strftime("%Y-%m-%d")
            rows.append(row)
    path = tmp_path / ("monthly_sales.csv" if temporal else "sales.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _install_mocks(monkeypatch, *, selected=None, indexing="success", persistence="success", timesfm_fails=False):
    selected = selected or ["kpi_trend", "anomaly_detection", "forecasting"]

    async def preparation_plan(profile):
        return preparation_module._fallback_plan(profile), "fallback", ["Test fallback planning."]

    async def orchestration_plan(_prepared, _capabilities):
        return OrchestrationPlan.model_validate({"selected_agents": selected, "decisions": [{"agent": name, "selected": name in selected, "reason": "Test plan"} for name in ("kpi_trend", "anomaly_detection", "forecasting")]})

    async def kpi_plan(_prepared):
        return KPITrendPlan.model_validate({"kpis": [{"id": f"kpi_{measure}", "title": f"Total {measure.title()}", "measure": measure, "aggregation": "sum"} for measure in ("revenue", "orders", "customers", "profit")], "trends": [{"id": "trend_revenue", "title": "Monthly Revenue", "measure": "revenue", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month"}]})

    async def anomaly_plan(_prepared):
        return AnomalyPlan.model_validate({"analyses": [{"id": "revenue_rolling", "measure": "revenue", "method": "rolling_deviation", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month"}]})

    async def forecast_plan(_prepared):
        return ForecastPlan.model_validate({"forecast": {"id": "forecast_revenue", "title": "Revenue forecast", "measure": "revenue", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month", "horizon": 3}})

    async def layout(_payload):
        return DashboardLayoutPlan(title="Sales dashboard", selected_kpi_ids=[f"kpi_{measure}" for measure in ("revenue", "orders", "customers", "profit")], selected_trend_ids=["trend_revenue"], include_forecast=True, section_order=[DashboardSection(id="kpis"), DashboardSection(id="timeline", chart_type="line"), DashboardSection(id="supportingCharts"), DashboardSection(id="details")])

    async def synthesis(_payload):
        return InsightSynthesisOutput(executive_summary="Deterministic test synthesis.")

    async def forecast(_values, horizon):
        if timesfm_fails:
            raise RuntimeError("TimesFM unavailable in test")
        return TimesFMForecast(values=[2000.0 + index * 10 for index in range(horizon)], lower_bounds=[1900.0] * horizon, upper_bounds=[2100.0] * horizon)

    monkeypatch.setattr(preparation_module, "_plan_with_groq_or_fallback", preparation_plan)
    monkeypatch.setattr(orchestrator_module, "_request_groq_plan", orchestration_plan)
    monkeypatch.setattr(kpi_module, "_request_groq_plan", kpi_plan)
    monkeypatch.setattr(anomaly_module, "_request_groq_plan", anomaly_plan)
    monkeypatch.setattr(forecasting_module, "_request_groq_plan", forecast_plan)
    monkeypatch.setattr(dashboard_module, "_request_groq_layout", layout)
    monkeypatch.setattr(synthesis_module, "_request_groq_synthesis", synthesis)
    monkeypatch.setattr(forecasting_module.timesfm_service, "forecast", forecast)

    def index_documents(session_id, dataset_id, retrieval_documents):
        if indexing == "failed":
            return {"status": "failed", "document_count": len(retrieval_documents), "indexed_count": 0, "failed_count": len(retrieval_documents), "message": "Embedding failed"}
        return {"status": "success", "document_count": len(retrieval_documents), "indexed_count": len(retrieval_documents), "failed_count": 0}

    persisted = []
    def persist(bundle):
        persisted.append(bundle)
        if persistence == "failed":
            return {"status": "failed", "session_id": bundle["session_id"], "dataset_id": bundle["dataset_id"], "message": "Write failed"}
        return {"status": "success", "session_id": bundle["session_id"], "dataset_id": bundle["dataset_id"]}

    monkeypatch.setattr(graph_module.retrieval_indexing_service, "index_documents", index_documents)
    monkeypatch.setattr(graph_module.business_intelligence_persistence_service, "persist_workflow", persist)
    return persisted


def _run(path, session="full-pipeline"):
    return asyncio.run(graph_module.build_business_intelligence_graph().ainvoke({"session_id": session, "dataset_id": session, "uploaded_file_path": str(path)}))


def test_complete_successful_pipeline(tmp_path, monkeypatch):
    persisted = _install_mocks(monkeypatch)
    result = _run(_csv(tmp_path))
    response = DashboardResponse.model_validate(result["dashboard_output"])
    assert {"kpi_trend_output", "anomaly_output", "forecasting_output", "synthesis_output", "retrieval_documents", "retrieval_indexing_result", "persistence_result"} <= set(result)
    assert result["workflow_status"] == "success"
    assert result["retrieval_indexing_result"]["status"] == result["persistence_result"]["status"] == "success"
    assert result["skipped_agents"] == []
    assert "persistence" in result["completed_agents"] and persisted
    assert 4 <= len(response.dashboard.kpis) <= 8
    assert 2 <= len(response.dashboard.supportingCharts) <= 4
    assert len({chart.type for chart in response.dashboard.supportingCharts}) == len(response.dashboard.supportingCharts)
    assert response.dashboard.timeline and response.dashboard.timeline.forecastMetadata.model == "TimesFM"


def test_no_temporal_data_skips_forecast(tmp_path, monkeypatch):
    _install_mocks(monkeypatch)
    result = _run(_csv(tmp_path, temporal=False), "non-temporal")
    assert "forecasting" in result["skipped_agents"]
    assert "forecasting" not in result.get("failed_agents", [])
    assert "forecasting_output" not in result
    assert DashboardResponse.model_validate(result["dashboard_output"]).dashboard is not None


def test_timesfm_and_indexing_failures_are_partial(tmp_path, monkeypatch):
    _install_mocks(monkeypatch, indexing="failed", timesfm_fails=True)
    result = _run(_csv(tmp_path), "recoverable-failures")
    assert result["forecasting_output"]["status"] == "partial"
    assert result["retrieval_indexing_result"]["status"] == "failed"
    assert result["workflow_status"] == "partial"
    assert "persistence" in result["completed_agents"]


def test_no_anomalies_and_kpi_only_routing(tmp_path, monkeypatch):
    _install_mocks(monkeypatch, selected=["kpi_trend"])
    result = _run(_csv(tmp_path, anomaly=False), "kpi-only")
    assert "anomaly_output" not in result and "forecasting_output" not in result
    assert {"anomaly_detection", "forecasting"} <= set(result["skipped_agents"])
    assert not ({"anomaly_detection", "forecasting"} & set(result.get("failed_agents", [])))
    assert {"dataset_summary", "kpi"} <= {document["document_type"] for document in result["retrieval_documents"]}


def test_no_anomalies_is_a_completed_analysis(tmp_path, monkeypatch):
    _install_mocks(monkeypatch)
    result = _run(_csv(tmp_path, anomaly=False), "stable-data")
    assert result["anomaly_output"]["anomalies"] == []
    assert "anomaly_detection" in result["completed_agents"]
    assert "anomaly_detection" not in result.get("failed_agents", [])
    assert DashboardResponse.model_validate(result["dashboard_output"]).dashboard is not None


def test_groq_unavailable_uses_deterministic_fallbacks(tmp_path, monkeypatch):
    _install_mocks(monkeypatch)

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("Groq unavailable")

    monkeypatch.setattr(orchestrator_module, "_request_groq_plan", unavailable)
    monkeypatch.setattr(kpi_module, "_request_groq_plan", unavailable)
    monkeypatch.setattr(anomaly_module, "_request_groq_plan", unavailable)
    monkeypatch.setattr(forecasting_module, "_request_groq_plan", unavailable)
    monkeypatch.setattr(synthesis_module, "_request_groq_synthesis", unavailable)
    monkeypatch.setattr(dashboard_module, "_request_groq_layout", unavailable)
    result = _run(_csv(tmp_path), "groq-unavailable")
    assert result["kpi_trend_output"]["kpis"]
    assert result["dashboard_output"]["dashboard"] is not None
    assert any("Deterministic" in warning for warning in result["warnings"])
    assert DashboardResponse.model_validate(result["dashboard_output"])


def test_persistence_failure_is_final_failure(tmp_path, monkeypatch):
    _install_mocks(monkeypatch, persistence="failed")
    result = _run(_csv(tmp_path), "persistence-failure")
    assert result["workflow_status"] == "failed"
    assert "persistence" in result["failed_agents"]
    assert result["persistence_result"]["status"] == "failed"
