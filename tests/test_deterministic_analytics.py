from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.agents.single.business_intelligence_agent import BusinessIntelligenceAgent
from app.rag.rag_service import DeterministicAnalytics, RagService
from app.schemas.business_intelligence import BusinessIntelligenceAgentInput


def _analytics(tmp_path: Path) -> DeterministicAnalytics:
    data_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "Year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "Product": ["Basic", "Premium"] * 4,
            "Price_USD": [10, 25, 11, 27, 12, 30, 14, 32],
            "Sales_Volume": [100, 80, 110, 90, 120, 100, 130, 110],
        }
    ).to_csv(data_path, index=False)
    profile = {
        "summary": {
            "measures": ["Price_USD", "Sales_Volume"],
            "dimensions": ["Product"],
            "timeField": "Year",
        }
    }
    return DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath=str(data_path),
            fileName=data_path.name,
        ),
        profile,
    )


def test_total_revenue_is_derived_from_price_and_volume(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("What is total revenue?")

    assert result is not None
    assert "Sum Revenue: 16,420.00" in result.text
    assert "Price_USD, Sales_Volume" in result.text
    assert result.direct_answer is not None
    assert "Revenue` derived as `Price_USD` × `Sales_Volume`" in result.direct_answer


def test_best_product_defaults_to_revenue_performance(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("Which product performed best?")

    assert result is not None
    assert "Top Revenue by Product: Premium: 10,950.00" in result.text
    assert result.direct_answer is not None
    assert "`Product`" in result.direct_answer


def test_generic_forecast_question_forecasts_next_year_revenue(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("What forecast information is available?")

    assert result is not None
    assert "Forecasted total Revenue for the next year (2025)" in result.text
    assert "linear trend on annual totals from 2021 to 2024" in result.text
    assert result.direct_answer is not None
    assert "`Year` from 2021 to 2024" in result.direct_answer


def test_best_product_is_routed_to_deterministic_calculation() -> None:
    profile = {"summary": {"measures": ["Revenue"], "dimensions": ["Product"]}}

    assert RagService().route_query("Which product performed best?", profile) == "calculation"


def test_deterministic_answer_takes_priority_over_retrieved_context(tmp_path: Path) -> None:
    agent_input = BusinessIntelligenceAgentInput(
        sessionId="test-session",
        filePath=str(tmp_path / "sales.csv"),
        fileName="sales.csv",
    )
    direct_answer = "**Answer:** Revenue is 100.\n\n**Grounding:** Calculated evidence."

    result = BusinessIntelligenceAgent()._answer_chat(
        {
            "agent_input": agent_input,
            "retrieved_context": "Unrelated retrieved document.",
            "calculated_evidence": "Calculated evidence: Revenue is 100.",
            "direct_answer": direct_answer,
            "retrieved_documents": [],
            "reranked_documents": [],
        }
    )

    assert result == {"chat_response": direct_answer}
