import asyncio

import pytest

from app.agents.multi import forecasting as forecasting_module
from app.core.model_policy import (
    multi_dashboard_model_usage,
    single_dashboard_model_usage,
)
from app.schemas.api import AgentModelUsage
from app.schemas.specialists import ForecastingOutput


def test_single_dashboard_usage_uses_the_configured_single_agent_model() -> None:
    assert single_dashboard_model_usage() == [
        {
            "agent": "Business intelligence",
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "provider": "openrouter",
            "executionStatus": "configured",
        }
    ]


def test_multi_dashboard_usage_lists_only_selected_specialists() -> None:
    usage = multi_dashboard_model_usage(["kpi_trend", "forecasting"])

    assert [item["agent"] for item in usage] == [
        "Data preparation",
        "Orchestrator",
        "KPI and trend analysis",
        "Forecasting",
        "Insight synthesis",
        "Dashboard generation",
    ]
    assert {item["model"] for item in usage} == {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "amazon/chronos-2",
    }
    assert {item["executionStatus"] for item in usage} == {"configured"}


def test_actual_invocation_status_replaces_configured_metadata() -> None:
    usage = multi_dashboard_model_usage(
        ["anomaly_detection"],
        [
            {
                "agent": "Anomaly detection",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "provider": "openrouter",
                "executionStatus": "succeeded",
            },
            {
                "agent": "Insight synthesis",
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "provider": "openrouter",
                "executionStatus": "fallback",
            },
        ],
    )

    statuses = {item["agent"]: item["executionStatus"] for item in usage}
    assert statuses["Anomaly detection"] == "succeeded"
    assert statuses["Insight synthesis"] == "fallback"
    assert statuses["Dashboard generation"] == "configured"


def test_legacy_usage_omits_unknown_execution_status_when_serialized() -> None:
    usage = AgentModelUsage(
        agent="Legacy agent",
        model="provider/model",
        provider="groq",
    )

    assert usage.model_dump() == {
        "agent": "Legacy agent",
        "model": "provider/model",
        "provider": "groq",
    }


@pytest.mark.parametrize(
    ("model", "expected_status"),
    [
        ("Chronos-2", "succeeded"),
        ("linear_trend", "fallback"),
        (None, "fallback"),
    ],
)
def test_forecasting_node_reports_actual_engine_outcome(
    monkeypatch: pytest.MonkeyPatch,
    model: str | None,
    expected_status: str,
) -> None:
    async def fake_run(_: dict[str, object]) -> ForecastingOutput:
        return ForecastingOutput(status="complete", model=model)

    monkeypatch.setattr(forecasting_module.forecasting_agent, "run", fake_run)

    update = asyncio.run(forecasting_module.forecasting_node({}))

    assert update["model_invocations"] == [
        {
            "agent": "Forecasting",
            "model": "amazon/chronos-2",
            "provider": "engine",
            "executionStatus": expected_status,
        }
    ]
