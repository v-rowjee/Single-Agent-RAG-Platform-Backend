from app.core.model_usage import (
    multi_dashboard_model_usage,
    single_dashboard_model_usage,
)


def test_single_dashboard_usage_uses_the_configured_single_agent_model() -> None:
    assert single_dashboard_model_usage() == [
        {
            "agent": "Business intelligence",
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
        }
    ]


def test_multi_dashboard_usage_lists_only_selected_specialists() -> None:
    usage = multi_dashboard_model_usage(["kpi_trend", "forecasting"])

    assert [item["agent"] for item in usage] == [
        "Data preparation",
        "KPI and trend analysis",
        "Forecasting",
        "Insight synthesis",
        "Dashboard generation",
    ]
    assert {item["model"] for item in usage} == {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "google/timesfm-2.5-200m-pytorch",
    }
