"""Grounded synthesis of specialist business-intelligence results."""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_INSIGHTS = 6
MAX_RECOMMENDATIONS = 5


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceReference(StrictModel):
    source_type: Literal["kpi", "trend", "anomaly", "forecast", "dataset"]
    source_id: str = Field(min_length=1)


class SynthesisedInsight(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    importance: Literal["low", "medium", "high"]
    evidence: list[EvidenceReference] = Field(default_factory=list)


class Recommendation(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: Literal["low", "medium", "high"]
    evidence: list[EvidenceReference] = Field(default_factory=list)


class InsightSynthesisOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    executive_summary: str
    key_insights: list[SynthesisedInsight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _compact(prepared: dict[str, Any], kpi: dict[str, Any] | None, anomaly: dict[str, Any] | None, forecast: dict[str, Any] | None) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {
        "dataset": {"id": "dataset_summary", "measures": prepared.get("primary_measures", []), "date_column": prepared.get("date_column")},
        "warnings": (prepared.get("warnings") or [])[:10],
        "kpis": (kpi or {}).get("kpis", [])[:8],
        "trends": [{**{key: value for key, value in item.items() if key != "points"}, "points": (item.get("points") or [])[:2] + (item.get("points") or [])[-2:]} for item in (kpi or {}).get("trends", [])[:3]],
        "anomalies": (anomaly or {}).get("anomalies", [])[:6],
        "forecast": {key: value for key, value in (forecast or {}).items() if key not in {"historical", "forecast"}} | {"historical": ((forecast or {}).get("historical") or [])[-2:], "forecast": ((forecast or {}).get("forecast") or [])[:4]},
    }


def _source_ids(prepared: dict[str, Any], kpi: dict[str, Any] | None, anomaly: dict[str, Any] | None, forecast: dict[str, Any] | None) -> set[tuple[str, str]]:
    sources = {("dataset", "dataset_summary")}
    sources.update(("kpi", str(item.get("id"))) for item in (kpi or {}).get("kpis", []) if item.get("id"))
    sources.update(("trend", str(item.get("id"))) for item in (kpi or {}).get("trends", []) if item.get("id"))
    sources.update(("anomaly", str(item.get("id"))) for item in (anomaly or {}).get("anomalies", []) if item.get("id"))
    if (forecast or {}).get("series_id"):
        sources.add(("forecast", str(forecast["series_id"])))
    return sources


async def _request_groq_synthesis(payload: dict[str, Any]) -> InsightSynthesisOutput:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    response = await AsyncGroq(api_key=key).chat.completions.create(
        model=MODEL_NAME, temperature=0.2, max_completion_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON only with executive_summary, key_insights, recommendations, limitations and warnings. Use only supplied IDs and numbers. Every insight and recommendation needs evidence [{source_type,source_id}]. Do not calculate values, invent context, claim causation, or create anomaly scores or forecasts. Distinguish observations from recommendations."},
            {"role": "user", "content": json.dumps(payload, default=str, separators=(",", ":"))},
        ],
    )
    try:
        return InsightSynthesisOutput.model_validate_json(response.choices[0].message.content or "{}")
    except ValidationError as exc:
        raise RuntimeError(f"Invalid Groq synthesis: {exc}") from exc


def _limitations(prepared: dict[str, Any], *outputs: dict[str, Any] | None) -> list[str]:
    values = list(prepared.get("limitations") or [])
    for output in outputs:
        values.extend((output or {}).get("limitations") or [])
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _fallback(prepared: dict[str, Any], kpi: dict[str, Any] | None, anomaly: dict[str, Any] | None, forecast: dict[str, Any] | None, warning: str) -> InsightSynthesisOutput:
    insights: list[SynthesisedInsight] = []
    for item in (kpi or {}).get("kpis", [])[:3]:
        if item.get("id"):
            insights.append(SynthesisedInsight(id=f"insight_{item['id']}", title=str(item.get("title") or item["id"]), description=f"Authoritative KPI value: {item.get('value')}.", importance="medium", evidence=[EvidenceReference(source_type="kpi", source_id=str(item["id"]))]))
    for item in (anomaly or {}).get("anomalies", [])[:2]:
        if item.get("id"):
            insights.append(SynthesisedInsight(id=f"insight_{item['id']}", title=f"Anomaly: {item.get('metric', item['id'])}", description=str(item.get("evidence") or "An anomaly was reported by the specialist."), importance="high" if item.get("severity") == "critical" else "medium", evidence=[EvidenceReference(source_type="anomaly", source_id=str(item["id"]))]))
    if (forecast or {}).get("series_id") and (forecast or {}).get("forecast"):
        insights.append(SynthesisedInsight(id=f"insight_{forecast['series_id']}", title=str(forecast.get("title") or "Forecast"), description="Forecast specialist returned future forecast points.", importance="medium", evidence=[EvidenceReference(source_type="forecast", source_id=str(forecast["series_id"]))]))
    summary = "Specialist results are available for review." if insights else "Limited specialist results are available; use the dataset summary and listed limitations when interpreting this analysis."
    return InsightSynthesisOutput(status="partial", executive_summary=summary, key_insights=insights[:MAX_INSIGHTS], limitations=_limitations(prepared, kpi, anomaly, forecast), warnings=[warning])


def _validate(result: InsightSynthesisOutput, available: set[tuple[str, str]]) -> InsightSynthesisOutput:
    def valid(items: list[SynthesisedInsight] | list[Recommendation], limit: int):
        kept = []
        for item in items[:limit]:
            evidence = [ref for ref in item.evidence if (ref.source_type, ref.source_id) in available]
            if evidence:
                kept.append(item.model_copy(update={"evidence": evidence}))
        return kept
    return result.model_copy(update={"key_insights": valid(result.key_insights, MAX_INSIGHTS), "recommendations": valid(result.recommendations, MAX_RECOMMENDATIONS)})


class InsightSynthesisAgent:
    async def run(self, prepared_dataset: dict[str, Any], kpi_trend_output: dict[str, Any] | None, anomaly_output: dict[str, Any] | None, forecasting_output: dict[str, Any] | None) -> InsightSynthesisOutput:
        prepared = prepared_dataset if isinstance(prepared_dataset, dict) else {}
        available = _source_ids(prepared, kpi_trend_output, anomaly_output, forecasting_output)
        try:
            result = await _request_groq_synthesis(_compact(prepared, kpi_trend_output, anomaly_output, forecasting_output))
            result = _validate(result, available)
            return result.model_copy(update={"warnings": list(dict.fromkeys([*(prepared.get("warnings") or []), *result.warnings]))})
        except Exception as exc:
            return _fallback(prepared, kpi_trend_output, anomaly_output, forecasting_output, f"Deterministic synthesis was used: {exc}")


insight_synthesis_agent = InsightSynthesisAgent()


async def insight_synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["warnings"] = [*(prepared.get("warnings") or []), *(state.get("warnings") or [])]
    result = await insight_synthesis_agent.run(prepared, state.get("kpi_trend_output"), state.get("anomaly_output"), state.get("forecasting_output"))
    return {"synthesis_output": result.model_dump(mode="json"), "completed_agents": ["insight_synthesis"]}
