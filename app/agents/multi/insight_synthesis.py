"""Grounded synthesis of specialist business-intelligence results."""
from __future__ import annotations

from typing import Any

from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import (
    EvidenceReference,
    InsightSynthesisOutput,
    Recommendation,
    SynthesisedInsight,
)


MAX_INSIGHTS = 6
MIN_RECOMMENDATIONS = 3
MAX_RECOMMENDATIONS = 5


def _compact(
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    temporal = prepared.get("temporal_profile") or {}
    return {
        "dataset": {
            "id": "dataset_summary",
            "file_name": prepared.get("file_name") or prepared.get("source_file_name"),
            "business_description": profile.get("business_description"),
            "row_count": profile.get("row_count"),
            "measures": (prepared.get("primary_measures") or [])[:12],
            "date_column": prepared.get("date_column"),
            "period_start": temporal.get("minimum_date"),
            "period_end": temporal.get("maximum_date"),
        },
        "warnings": (prepared.get("warnings") or [])[:6],
        "kpis": (kpi or {}).get("kpis", [])[:8],
        "trends": [
            {
                **{key: value for key, value in item.items() if key != "points"},
                "points": (item.get("points") or [])[-4:],
            }
            for item in (kpi or {}).get("trends", [])[:3]
        ],
        "anomalies": (anomaly or {}).get("anomalies", [])[:4],
        "forecast": {
            key: value
            for key, value in (forecast or {}).items()
            if key not in {"historical", "forecast"}
        }
        | {
            "historical": ((forecast or {}).get("historical") or [])[-2:],
            "forecast": ((forecast or {}).get("forecast") or [])[:3],
        },
    }


def _source_ids(
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> set[tuple[str, str]]:
    sources = {("dataset", "dataset_summary")}
    sources.update(
        ("kpi", str(item.get("id")))
        for item in (kpi or {}).get("kpis", [])
        if item.get("id")
    )
    sources.update(
        ("trend", str(item.get("id")))
        for item in (kpi or {}).get("trends", [])
        if item.get("id")
    )
    sources.update(
        ("anomaly", str(item.get("id")))
        for item in (anomaly or {}).get("anomalies", [])
        if item.get("id")
    )
    if (forecast or {}).get("series_id"):
        sources.add(("forecast", str(forecast["series_id"])))
    return sources


async def _request_synthesis(
    payload: dict[str, Any],
) -> InsightSynthesisOutput:
    prompts = render_agent_prompts("multi/insight_synthesis", payload=payload)
    return await request_structured(
        policy=agent_model_policy("insight_synthesis"),
        response_model=InsightSynthesisOutput,
        schema_name="insight_synthesis",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _limitations(
    prepared: dict[str, Any],
    *outputs: dict[str, Any] | None,
) -> list[str]:
    values = list(prepared.get("limitations") or [])
    for output in outputs:
        values.extend((output or {}).get("limitations") or [])
    return list(
        dict.fromkeys(str(value) for value in values if str(value).strip())
    )


def _number(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:,.2f}"
    return str(value)


def _deterministic_summary(
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> str:
    profile = prepared.get("dataset_profile") or {}
    temporal = prepared.get("temporal_profile") or {}
    description = str(
        profile.get("business_description")
        or prepared.get("file_name")
        or prepared.get("source_file_name")
        or "The uploaded business dataset"
    ).strip()
    row_count = int(profile.get("row_count") or 0)
    period = ""
    if temporal.get("minimum_date") and temporal.get("maximum_date"):
        period = (
            f" from {temporal['minimum_date']} to {temporal['maximum_date']}"
        )
    sentences = [
        f"{description} is represented by {row_count:,} analysed records{period}.",
    ]

    first_kpi = next(
        (item for item in (kpi or {}).get("kpis", []) if item.get("id")),
        None,
    )
    if first_kpi:
        change = first_kpi.get("change_percent")
        movement = (
            f", {'up' if change > 0 else 'down'} {abs(float(change)):.1f}% "
            f"from {first_kpi.get('previous_period')}"
            if isinstance(change, (float, int)) and change != 0
            else ", unchanged from the previous period"
            if change == 0
            else ""
        )
        sentences.append(
            f"The latest {first_kpi.get('title') or 'primary KPI'} was "
            f"{_number(first_kpi.get('value'))}{movement}."
        )

    first_anomaly = next(
        iter((anomaly or {}).get("anomalies", [])),
        None,
    )
    if first_anomaly:
        sentences.append(
            "Historical analysis highlighted "
            f"{first_anomaly.get('evidence') or 'an unusual movement requiring review'}."
        )
    elif (kpi or {}).get("trends"):
        sentences.append(
            "The historical series provides a clear baseline for monitoring "
            "whether the latest movement persists."
        )

    points = (forecast or {}).get("forecast") or []
    if points:
        direction = (
            "increase"
            if float(points[-1]["value"]) > float(points[0]["value"])
            else "decrease"
            if float(points[-1]["value"]) < float(points[0]["value"])
            else "remain broadly stable"
        )
        sentences.append(
            f"The {len(points)}-period forecast expects "
            f"{forecast.get('measure') or 'the primary measure'} to {direction}, "
            f"moving from {_number(points[0].get('value'))} to "
            f"{_number(points[-1].get('value'))}, so plans should be checked "
            "against actual results as new data arrives."
        )
    else:
        sentences.append(
            "No reliable future series was available, so near-term decisions "
            "should be updated when additional observations arrive."
        )
    sentences.append(
        "Leaders should pair these results with category, branch, and channel "
        "breakdowns before committing resources."
    )
    summary = " ".join(sentences)
    if len(summary.split()) < 60:
        summary += (
            " Decisions should remain proportionate to the available evidence, "
            "with owners validating the next refresh before making longer-term "
            "commercial commitments."
        )
    words = summary.split()
    return " ".join(words[:100])


def _deterministic_recommendations(
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> list[Recommendation]:
    actions: list[Recommendation] = []
    first_anomaly = next(
        (
            item
            for item in (anomaly or {}).get("anomalies", [])
            if item.get("id")
        ),
        None,
    )
    if first_anomaly:
        actions.append(
            Recommendation(
                id="action_investigate_anomaly",
                title="Investigate the leading anomaly",
                description=(
                    f"Review transactions and operational changes around "
                    f"{first_anomaly.get('period') or 'the flagged observation'} "
                    "to confirm whether the movement is genuine and repeatable."
                ),
                priority=(
                    "high"
                    if first_anomaly.get("severity") == "critical"
                    else "medium"
                ),
                evidence=[
                    EvidenceReference(
                        source_type="anomaly",
                        source_id=str(first_anomaly["id"]),
                    )
                ],
            )
        )

    first_kpi = next(
        (
            item
            for item in (kpi or {}).get("kpis", [])
            if item.get("id")
        ),
        None,
    )
    if first_kpi:
        change = first_kpi.get("change_percent")
        action = "reverse the decline" if isinstance(change, (float, int)) and change < 0 else "sustain the improvement"
        actions.append(
            Recommendation(
                id="action_review_kpi_drivers",
                title="Review KPI drivers",
                description=(
                    f"Break down {first_kpi.get('title') or 'the primary KPI'} "
                    f"by the strongest available business dimensions to {action} "
                    "and assign an owner to monitor the next period."
                ),
                priority="high" if isinstance(change, (float, int)) and change < 0 else "medium",
                evidence=[
                    EvidenceReference(
                        source_type="kpi",
                        source_id=str(first_kpi["id"]),
                    )
                ],
            )
        )

    if (forecast or {}).get("series_id") and (forecast or {}).get("forecast"):
        actions.append(
            Recommendation(
                id="action_plan_for_forecast",
                title="Plan against the forecast",
                description=(
                    f"Check capacity, budget, and commercial plans against the "
                    f"next {len(forecast.get('forecast') or [])} "
                    f"{forecast.get('granularity') or 'forecast'} periods and "
                    "compare each prediction with the next actual result."
                ),
                priority="medium",
                evidence=[
                    EvidenceReference(
                        source_type="forecast",
                        source_id=str(forecast["series_id"]),
                    )
                ],
            )
        )

    fallback_actions = [
        Recommendation(
            id="action_monitor_dashboard",
            title="Establish a review cadence",
            description=(
                "Review the KPI, segment, and anomaly outputs each reporting "
                "period so material changes are escalated consistently."
            ),
            priority="medium",
            evidence=[
                EvidenceReference(
                    source_type="dataset",
                    source_id="dataset_summary",
                )
            ],
        ),
        Recommendation(
            id="action_data_quality",
            title="Protect data quality",
            description=(
                "Resolve missing values and duplicate-record causes before the "
                "next refresh so comparisons and forecasts remain dependable."
            ),
            priority="medium",
            evidence=[
                EvidenceReference(
                    source_type="dataset",
                    source_id="dataset_summary",
                )
            ],
        ),
    ]
    for action in fallback_actions:
        if len(actions) >= MIN_RECOMMENDATIONS:
            break
        actions.append(action)
    return actions[:MAX_RECOMMENDATIONS]


def _fallback(
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
    warning: str,
) -> InsightSynthesisOutput:
    insights: list[SynthesisedInsight] = []
    for item in (kpi or {}).get("kpis", [])[:3]:
        if item.get("id"):
            change = item.get("change_percent")
            description = (
                f"Latest value: {_number(item.get('value'))}; "
                f"change versus {item.get('previous_period')}: {float(change):+.1f}%."
                if isinstance(change, (float, int))
                else f"Latest authoritative value: {_number(item.get('value'))}."
            )
            insights.append(
                SynthesisedInsight(
                    id=f"insight_{item['id']}",
                    title=str(item.get("title") or item["id"]),
                    description=description,
                    importance="medium",
                    evidence=[
                        EvidenceReference(
                            source_type="kpi",
                            source_id=str(item["id"]),
                        )
                    ],
                )
            )
    for item in (anomaly or {}).get("anomalies", [])[:2]:
        if item.get("id"):
            insights.append(
                SynthesisedInsight(
                    id=f"insight_{item['id']}",
                    title=f"Anomaly: {item.get('metric', item['id'])}",
                    description=str(
                        item.get("evidence")
                        or "An anomaly was reported by the specialist."
                    ),
                    importance=(
                        "high"
                        if item.get("severity") == "critical"
                        else "medium"
                    ),
                    evidence=[
                        EvidenceReference(
                            source_type="anomaly",
                            source_id=str(item["id"]),
                        )
                    ],
                )
            )
    return InsightSynthesisOutput(
        status="partial",
        executive_summary=_deterministic_summary(
            prepared,
            kpi,
            anomaly,
            forecast,
        ),
        key_insights=insights[:MAX_INSIGHTS],
        recommendations=_deterministic_recommendations(
            prepared,
            kpi,
            anomaly,
            forecast,
        ),
        limitations=_limitations(prepared, kpi, anomaly, forecast),
        warnings=[warning],
    )


def _validate(
    result: InsightSynthesisOutput,
    available: set[tuple[str, str]],
    prepared: dict[str, Any],
    kpi: dict[str, Any] | None,
    anomaly: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> InsightSynthesisOutput:
    def valid(
        items: list[SynthesisedInsight] | list[Recommendation],
        limit: int,
    ) -> list[SynthesisedInsight] | list[Recommendation]:
        kept: list[SynthesisedInsight] | list[Recommendation] = []
        for item in items[:limit]:
            evidence = [
                ref
                for ref in item.evidence
                if (ref.source_type, ref.source_id) in available
            ]
            if evidence:
                kept.append(item.model_copy(update={"evidence": evidence}))
        return kept

    insights = valid(result.key_insights, MAX_INSIGHTS)
    recommendations = list(valid(result.recommendations, MAX_RECOMMENDATIONS))
    existing_ids = {item.id for item in recommendations}
    for action in _deterministic_recommendations(
        prepared,
        kpi,
        anomaly,
        forecast,
    ):
        if len(recommendations) >= MIN_RECOMMENDATIONS:
            break
        if action.id not in existing_ids:
            recommendations.append(action)
            existing_ids.add(action.id)
    summary = result.executive_summary.strip()
    if not 60 <= len(summary.split()) <= 100:
        summary = _deterministic_summary(prepared, kpi, anomaly, forecast)
    return result.model_copy(
        update={
            "executive_summary": summary,
            "key_insights": insights,
            "recommendations": recommendations[:MAX_RECOMMENDATIONS],
        }
    )


class InsightSynthesisAgent:
    async def run(
        self,
        prepared_dataset: dict[str, Any],
        kpi_trend_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
    ) -> InsightSynthesisOutput:
        result, _ = await self.run_with_status(
            prepared_dataset,
            kpi_trend_output,
            anomaly_output,
            forecasting_output,
        )
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
        kpi_trend_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
    ) -> tuple[InsightSynthesisOutput, ModelExecutionStatus]:
        prepared = (
            prepared_dataset if isinstance(prepared_dataset, dict) else {}
        )
        available = _source_ids(
            prepared,
            kpi_trend_output,
            anomaly_output,
            forecasting_output,
        )
        try:
            result = await _request_synthesis(
                _compact(
                    prepared,
                    kpi_trend_output,
                    anomaly_output,
                    forecasting_output,
                )
            )
            result = _validate(
                result,
                available,
                prepared,
                kpi_trend_output,
                anomaly_output,
                forecasting_output,
            )
            return (
                result.model_copy(
                    update={
                        "warnings": list(
                            dict.fromkeys(
                                [
                                    *(prepared.get("warnings") or []),
                                    *result.warnings,
                                ]
                            )
                        )
                    }
                ),
                "succeeded",
            )
        except Exception as exc:
            return (
                _fallback(
                    prepared,
                    kpi_trend_output,
                    anomaly_output,
                    forecasting_output,
                    f"Deterministic synthesis was used: {exc}",
                ),
                "fallback",
            )


insight_synthesis_agent = InsightSynthesisAgent()


async def insight_synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["warnings"] = [
        *(prepared.get("warnings") or []),
        *(state.get("warnings") or []),
    ]
    result, execution_status = await insight_synthesis_agent.run_with_status(
        prepared,
        state.get("kpi_trend_output"),
        state.get("anomaly_output"),
        state.get("forecasting_output"),
    )
    return {
        "synthesis_output": result.model_dump(mode="json"),
        "completed_agents": ["insight_synthesis"],
        "model_invocations": [
            agent_model_usage("insight_synthesis", execution_status)
        ],
    }
