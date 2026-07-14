"""Integration coverage for the real specialist nodes with model boundaries mocked."""
from __future__ import annotations

import asyncio

import pandas as pd

from app.agents.multi import anomaly_detection_agent as anomaly_module
from app.agents.multi import forecasting_agent as forecasting_module
from app.agents.multi import kpi_trend_agent as kpi_module
from app.agents.multi.anomaly_detection_agent import AnomalyDetectionOutput, AnomalyPlan
from app.agents.multi.forecasting_agent import ForecastPlan, ForecastingOutput
from app.agents.multi.kpi_trend_agent import KPITrendOutput, KPITrendPlan
from app.orchestration.business_intelligence_graph import build_business_intelligence_graph
from app.services.timesfm_service import TimesFMForecast


def _prepared_dataset(tmp_path) -> dict[str, object]:
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    revenue = [1000 + index * 25 for index in range(24)]
    revenue[17] = 5000  # obvious period anomaly
    rows = []
    for index, date in enumerate(dates):
        for branch, share in (("North", 0.6), ("South", 0.4)):
            rows.append({"transaction_date": date.strftime("%Y-%m-%d"), "revenue": revenue[index] * share, "orders": 20 + index, "branch": branch})
    path = tmp_path / "prepared.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return {
        "prepared_file_path": str(path),
        "date_column": "transaction_date",
        "primary_measures": ["revenue", "orders"],
        "dimension_candidates": ["branch"],
        "time_series_candidates": ["revenue"],
        "temporal_profile": {"date_column": "transaction_date", "unique_periods": 24, "inferred_frequency": "month"},
        "capability_flags": {"supports_kpis": True, "supports_trends": True, "supports_anomalies": True, "supports_forecasting": True},
        "dataset_profile": {"row_count": len(rows), "column_count": 4, "column_profiles": [{"name": "transaction_date", "inferred_type": "date", "unique_count": 24}, {"name": "revenue", "inferred_type": "numeric", "unique_count": 24}, {"name": "orders", "inferred_type": "numeric", "unique_count": 24}, {"name": "branch", "inferred_type": "categorical", "unique_count": 2}]},
        "limitations": [],
    }


def _graph(prepared: dict[str, object], selected_agents: list[str]):
    async def cleaning_node(state):
        return {"completed_agents": ["generic_cleaning"]}

    async def preparation_node(state):
        return {"prepared_dataset": prepared, "prepared_file_path": prepared["prepared_file_path"], "completed_agents": ["data_preparation"]}

    async def orchestration_node(state):
        return {"orchestration_plan": {"selected_agents": selected_agents}, "completed_agents": ["orchestrator"]}

    return build_business_intelligence_graph(generic_cleaning_node_fn=cleaning_node, data_preparation_node_fn=preparation_node, orchestrator_node_fn=orchestration_node)


def _mock_plans(monkeypatch) -> None:
    async def kpi_plan(prepared):
        return KPITrendPlan.model_validate({"kpis": [{"id": "total_revenue", "title": "Total Revenue", "measure": "revenue", "aggregation": "sum"}], "trends": [{"id": "monthly_revenue", "title": "Monthly Revenue", "measure": "revenue", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month"}]})

    async def anomaly_plan(prepared):
        return AnomalyPlan.model_validate({"analyses": [{"id": "revenue_rolling", "measure": "revenue", "method": "rolling_deviation", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month"}]})

    async def forecast_plan(prepared):
        return ForecastPlan.model_validate({"forecast": {"id": "revenue_forecast", "title": "Revenue Forecast", "measure": "revenue", "aggregation": "sum", "date_column": "transaction_date", "granularity": "month", "horizon": 3}})

    monkeypatch.setattr(kpi_module, "_request_groq_plan", kpi_plan)
    monkeypatch.setattr(anomaly_module, "_request_groq_plan", anomaly_plan)
    monkeypatch.setattr(forecasting_module, "_request_groq_plan", forecast_plan)


def test_specialists_calculate_and_join(tmp_path, monkeypatch) -> None:
    prepared = _prepared_dataset(tmp_path)
    _mock_plans(monkeypatch)

    async def forecast(values, horizon):
        assert len(values) == 24
        assert horizon == 3
        return TimesFMForecast(values=[1700.0, 1725.0, 1750.0], lower_bounds=[1600.0, 1625.0, 1650.0], upper_bounds=[1800.0, 1825.0, 1850.0])

    monkeypatch.setattr(forecasting_module.timesfm_service, "forecast", forecast)
    result = asyncio.run(_graph(prepared, ["kpi_trend", "anomaly_detection", "forecasting"]).ainvoke({"session_id": "specialist-test"}))

    kpis = KPITrendOutput.model_validate(result["kpi_trend_output"])
    anomalies = AnomalyDetectionOutput.model_validate(result["anomaly_output"])
    forecast_output = ForecastingOutput.model_validate(result["forecasting_output"])
    assert kpis.kpis and kpis.kpis[0].raw_value == 34475.0
    assert kpis.trends and len(kpis.trends[0].points) == 24
    assert anomalies.anomalies
    assert forecast_output.historical and [point.value for point in forecast_output.forecast] == [1700.0, 1725.0, 1750.0]
    assert {"kpi_trend", "anomaly_detection", "forecasting", "specialist_join"} <= set(result["completed_agents"])
    assert all(output.get("status") != "placeholder" for output in (result["kpi_trend_output"], result["anomaly_output"], result["forecasting_output"]))


def test_specialist_routing_skips_forecasting(tmp_path, monkeypatch) -> None:
    prepared = _prepared_dataset(tmp_path)
    _mock_plans(monkeypatch)
    called = False

    async def forecast(values, horizon):
        nonlocal called
        called = True
        return TimesFMForecast(values=[])

    monkeypatch.setattr(forecasting_module.timesfm_service, "forecast", forecast)
    result = asyncio.run(_graph(prepared, ["kpi_trend", "anomaly_detection"]).ainvoke({"session_id": "routing-test"}))

    assert "kpi_trend_output" in result and "anomaly_output" in result
    assert "forecasting_output" not in result
    assert not called
    assert "specialist_join" in result["completed_agents"]
