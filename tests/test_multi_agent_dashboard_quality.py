from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agents.multi import dashboard_generation_agent as dashboard_module
from app.agents.multi import forecasting_agent as forecasting_module
from app.services.series import (
    aggregation_for_measure,
    infer_time_granularity,
)
from app.agents.multi.data_preparation_agent import _generic_clean_csv
from app.agents.multi.dashboard_generation_agent import DashboardGenerationAgent
from app.agents.multi.forecasting_agent import ForecastingAgent
from app.agents.multi.insight_synthesis_agent import _fallback as synthesis_fallback
from app.agents.multi.kpi_trend_agent import KPITrendAgent
from app.agents.multi.orchestrator_agent import (
    AgentDecision,
    OrchestrationPlan,
    OrchestratorAgent,
)
from app.core.config import configured_agent_models


def _rows(periods: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="MS")
    records: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        for category, branch, multiplier in (
            ("Membership", "Northside", 1.0),
            ("Personal Training", "Riverside", 1.25),
            ("Supplement", "City Centre", 0.75),
        ):
            revenue = (1000 + index * 20) * multiplier
            records.append(
                {
                    "transaction_date": date.isoformat(),
                    "year": date.year,
                    "quarter": f"Q{date.quarter}",
                    "month": date.month,
                    "month_name": date.strftime("%B"),
                    "product_category": category,
                    "branch": branch,
                    "customer_segment": "Student" if category == "Supplement" else "General",
                    "sales_channel": "Website" if category == "Membership" else "Front Desk",
                    "net_revenue_gbp": revenue,
                    "profit_gbp": revenue * 0.3,
                    "quantity": 10 + index,
                    "unit_price_gbp": revenue / (10 + index),
                    "discount_pct": 5.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _prepared(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    profiles = [
        {
            "name": column,
            "inferred_type": "date" if column == "transaction_date" else "numeric" if pd.api.types.is_numeric_dtype(frame[column]) else "categorical",
            "unique_count": int(frame[column].nunique(dropna=True)),
            "null_count": int(frame[column].isna().sum()),
        }
        for column in frame.columns
    ]
    return {
        "prepared_file_path": str(path),
        "file_name": "gym_sales.csv",
        "date_column": "transaction_date",
        "time_granularity": "month",
        "primary_measures": [
            "quantity",
            "unit_price_gbp",
            "discount_pct",
            "net_revenue_gbp",
            "profit_gbp",
        ],
        "time_series_candidates": ["net_revenue_gbp", "profit_gbp", "quantity"],
        "dimension_candidates": [
            "year",
            "quarter",
            "month",
            "month_name",
            "product_category",
            "branch",
            "customer_segment",
            "sales_channel",
        ],
        "temporal_profile": {
            "date_column": "transaction_date",
            "unique_periods": 24,
            "minimum_date": "2023-01-01",
            "maximum_date": "2024-12-01",
            "inferred_frequency": "month",
        },
        "capability_flags": {
            "supports_kpis": True,
            "supports_trends": True,
            "supports_anomalies": True,
            "supports_forecasting": True,
            "has_temporal_data": True,
        },
        "dataset_profile": {
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "business_description": "A multi-branch gym sales business",
            "column_profiles": profiles,
        },
    }


def test_calendar_helper_columns_are_not_coerced_to_dates(tmp_path: Path) -> None:
    source = tmp_path / "calendar.csv"
    _rows(2).to_csv(source, index=False)

    cleaned, _ = _generic_clean_csv(str(source), tmp_path / "cleaned")

    assert pd.api.types.is_numeric_dtype(cleaned["year"])
    assert pd.api.types.is_numeric_dtype(cleaned["month"])
    assert not pd.api.types.is_datetime64_any_dtype(cleaned["month_name"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["transaction_date"])


def test_kpis_use_latest_period_and_percentage_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sales.csv"
    _rows().to_csv(path, index=False)
    prepared = _prepared(path)

    async def no_llm(_: dict[str, Any]):
        raise RuntimeError("offline test")

    monkeypatch.setattr(
        "app.agents.multi.kpi_trend_agent._request_plan",
        no_llm,
    )
    result = asyncio.run(KPITrendAgent().run(prepared))
    revenue = next(item for item in result.kpis if item.measure == "net_revenue_gbp")

    assert revenue.current_period == "2024-12"
    assert revenue.previous_period == "2024-11"
    assert revenue.change_percent is not None and revenue.change_percent > 0
    assert revenue.baseline_period == "2023-01"
    assert (
        revenue.baseline_change_percent is not None
        and revenue.baseline_change_percent > 0
    )
    assert aggregation_for_measure("discount_pct") == "mean"
    assert result.trends[0].measure == "net_revenue_gbp"


def test_multi_year_transaction_history_uses_a_readable_monthly_grain() -> None:
    dates = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    # Real transaction histories can have days with no purchases.  The series
    # remains more complete and useful once grouped into calendar months.
    irregular_dates = pd.Series(dates.delete(slice(0, len(dates), 6)))

    assert infer_time_granularity(irregular_dates, "day") == "month"


def test_forecast_falls_back_and_keeps_primary_timeline_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sales.csv"
    _rows().to_csv(path, index=False)
    prepared = _prepared(path)

    async def unavailable(*_: Any, **__: Any):
        raise RuntimeError("Chronos-2 offline")

    monkeypatch.setattr(forecasting_module.chronos_service, "forecast", unavailable)
    result = asyncio.run(ForecastingAgent().run(prepared))

    assert result.measure == "net_revenue_gbp"
    assert result.aggregation == "sum"
    assert result.granularity == "month"
    assert result.model == "seasonal_naive"
    assert [point.period for point in result.forecast] == [
        "2025-01",
        "2025-02",
        "2025-03",
    ]
    assert len(result.forecast) == 3


def test_dashboard_has_non_temporal_charts_forecast_and_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sales.csv"
    frame = _rows()
    frame.to_csv(path, index=False)
    prepared = _prepared(path)
    monthly = frame.groupby(pd.to_datetime(frame["transaction_date"]).dt.to_period("M"))["net_revenue_gbp"].sum()
    points = [
        {"period": str(period), "value": float(value)}
        for period, value in monthly.items()
    ]
    kpi_output = {
        "kpis": [
            {
                "id": f"kpi_{name}",
                "title": name.replace("_", " ").title(),
                "value": value,
                "raw_value": value,
                "aggregation": aggregation,
                "measure": name,
                "change_percent": 8.0,
                "previous_period": "2024-11",
            }
            for name, value, aggregation in (
                ("net_revenue_gbp", 5000.0, "sum"),
                ("profit_gbp", 1500.0, "sum"),
                ("quantity", 100.0, "sum"),
                ("unit_price_gbp", 50.0, "mean"),
            )
        ],
        "trends": [
            {
                "id": "trend_net_revenue_gbp_month",
                "title": "Monthly Net Revenue",
                "measure": "net_revenue_gbp",
                "aggregation": "sum",
                "granularity": "month",
                "points": points,
            }
        ],
    }
    forecasting_output = {
        "series_id": "forecast_net_revenue_gbp",
        "measure": "net_revenue_gbp",
        "aggregation": "sum",
        "granularity": "month",
        "model": "seasonal_naive",
        "confidence_level": 0.95,
        "forecast": [
            {"period": "2025-01", "value": 6000.0, "lower_bound": 5000.0, "upper_bound": 7000.0},
            {"period": "2025-02", "value": 6100.0, "lower_bound": 5100.0, "upper_bound": 7100.0},
            {"period": "2025-03", "value": 6200.0, "lower_bound": 5200.0, "upper_bound": 7200.0},
        ],
    }
    synthesis = {
        "executive_summary": "A multi-branch gym sales business is represented by recent revenue, profit, quantity, and pricing data. Net revenue improved in the latest month compared with the prior period. The historical series provides a baseline for monitoring performance by category and branch. The three-month forecast indicates a modest increase, so management should compare incoming actuals with the outlook.",
        "key_insights": [],
        "recommendations": [
            {"id": "action_one", "title": "Review KPI drivers", "description": "Review category and branch drivers.", "priority": "medium", "evidence": [{"source_type": "kpi", "source_id": "kpi_net_revenue_gbp"}]},
            {"id": "action_two", "title": "Plan for forecast", "description": "Check capacity against forecast.", "priority": "medium", "evidence": [{"source_type": "forecast", "source_id": "forecast_net_revenue_gbp"}]},
            {"id": "action_three", "title": "Review data quality", "description": "Maintain complete reporting data.", "priority": "low", "evidence": [{"source_type": "dataset", "source_id": "dataset_summary"}]},
        ],
        "limitations": [],
    }

    async def no_layout(_: dict[str, Any]):
        raise RuntimeError("offline test")

    monkeypatch.setattr(dashboard_module, "_request_layout", no_layout)
    result = asyncio.run(
        DashboardGenerationAgent().run(
            prepared,
            kpi_output,
            {
                "anomalies": [
                    {
                        "id": "anomaly_profit_month",
                        "metric": "profit_gbp",
                        "aggregation": "sum",
                        "granularity": "month",
                        "period": "2024-12",
                        "observed_value": 1500.0,
                        "severity": "warning",
                        "evidence": "A profit anomaly, not a revenue anomaly.",
                    }
                ]
            },
            forecasting_output,
            synthesis,
        )
    )
    dashboard = result.dashboard.dashboard
    assert dashboard is not None
    assert dashboard.timeline is not None
    assert len(dashboard.timeline.forecast) == 3
    assert dashboard.timeline.forecastMetadata.target == "net_revenue_gbp"
    assert dashboard.timeline.anomalies == []
    assert len(dashboard.supportingCharts) >= 2
    assert len({chart.type for chart in dashboard.supportingCharts}) == len(dashboard.supportingCharts)
    assert all(
        not any(token in chart.title.lower() for token in ("year", "quarter", "month"))
        for chart in dashboard.supportingCharts
    )
    assert len(dashboard.recommendedActions) >= 3
    assert dashboard.executiveSummary == dashboard.analysis.businessSummary


def test_synthesis_fallback_is_grounded_and_has_three_actions() -> None:
    result = synthesis_fallback(
        {
            "file_name": "gym_sales.csv",
            "dataset_profile": {"row_count": 100, "business_description": "A gym business"},
            "temporal_profile": {"minimum_date": "2024-01-01", "maximum_date": "2024-12-31"},
        },
        {"kpis": [{"id": "kpi_revenue", "title": "Revenue", "value": 1200.0, "change_percent": 10.0, "previous_period": "2024-11"}]},
        {"anomalies": []},
        {"series_id": "forecast_revenue", "measure": "revenue", "granularity": "month", "forecast": [{"value": 1300.0}, {"value": 1400.0}, {"value": 1500.0}]},
        "offline",
    )

    assert "Specialist results are available" not in result.executive_summary
    assert len(result.recommendations) >= 3
    assert 60 <= len(result.executive_summary.split()) <= 100


def test_deterministic_routing_and_active_model_defaults() -> None:
    prepared = {
        "date_column": "transaction_date",
        "primary_measures": ["net_revenue_gbp"],
        "time_series_candidates": ["net_revenue_gbp"],
        "temporal_profile": {"unique_periods": 24},
        "capability_flags": {
            "supports_kpis": True,
            "supports_trends": True,
            "supports_anomalies": True,
            "supports_forecasting": True,
            "has_temporal_data": True,
        },
    }
    plan = asyncio.run(OrchestratorAgent(planner=None).run(prepared))

    assert plan.selected_agents == ["kpi_trend", "anomaly_detection", "forecasting"]
    assert configured_agent_models() == {
        "data_preparation": "openai/gpt-oss-20b",
        "orchestrator": "groq/compound",
        "kpi_trend": "openai/gpt-oss-120b",
        "anomaly_detection": "nvidia/nemotron-3-super-120b-a12b:free",
        "dashboard_generation": "poolside/laguna-xs-2.1:free",
        "insight_synthesis": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "chat": "openai/gpt-oss-120b",
        "single_dashboard": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "single_chat": "qwen/qwen3.6-27b",
    }


def test_compound_plan_remains_inside_deterministic_capability_gates() -> None:
    prepared = {
        "date_column": "transaction_date",
        "primary_measures": ["net_revenue_gbp"],
        "time_series_candidates": [],
        "temporal_profile": {"unique_periods": 2},
        "capability_flags": {
            "supports_kpis": True,
            "supports_trends": True,
            "supports_anomalies": False,
            "supports_forecasting": False,
            "has_temporal_data": True,
        },
    }

    async def propose(
        capabilities: dict[str, bool],
        eligible_agents: set[str],
    ) -> OrchestrationPlan:
        assert capabilities["supports_kpis"] is True
        assert eligible_agents == {"kpi_trend"}
        return OrchestrationPlan(
            selected_agents=["kpi_trend", "anomaly_detection"],
            decisions=[
                AgentDecision(
                    agent="kpi_trend",
                    selected=True,
                    reason="KPI analysis is relevant.",
                ),
                AgentDecision(
                    agent="anomaly_detection",
                    selected=True,
                    reason="Requested despite missing capability.",
                ),
            ],
        )

    plan = asyncio.run(OrchestratorAgent(planner=propose).run(prepared))

    assert plan.selected_agents == ["kpi_trend"]
    anomaly = next(
        decision
        for decision in plan.decisions
        if decision.agent == "anomaly_detection"
    )
    assert anomaly.selected is False
    assert "does not support" in anomaly.reason
