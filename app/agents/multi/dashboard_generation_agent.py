"""Build the canonical dashboard from authoritative specialist outputs."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.business_intelligence import DashboardResponse

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_DASHBOARD_KPIS, MAX_DASHBOARD_TRENDS, MAX_DASHBOARD_ANOMALIES = 8, 3, 6
MAX_DASHBOARD_INSIGHTS, MAX_DASHBOARD_RECOMMENDATIONS = 6, 5


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DashboardSection(StrictModel):
    id: Literal["kpis", "timeline", "supportingCharts", "details"]
    chart_type: Literal["line", "bar", "area", "table"] | None = None


class DashboardLayoutPlan(StrictModel):
    title: str = Field(min_length=1)
    selected_kpi_ids: list[str] = Field(default_factory=list)
    selected_trend_ids: list[str] = Field(default_factory=list)
    selected_anomaly_ids: list[str] = Field(default_factory=list)
    selected_insight_ids: list[str] = Field(default_factory=list)
    selected_recommendation_ids: list[str] = Field(default_factory=list)
    include_forecast: bool = False
    section_order: list[DashboardSection] = Field(default_factory=list)


class DashboardGenerationOutput(StrictModel):
    status: Literal["complete", "partial"] = "partial"
    layout_plan: DashboardLayoutPlan
    dashboard: DashboardResponse
    warnings: list[str] = Field(default_factory=list)


def _ids(items: list[dict[str, Any]], limit: int) -> list[str]:
    return [str(item["id"]) for item in items[:limit] if item.get("id")]


def _dedupe_selected(values: list[str], valid: set[str], limit: int) -> list[str]:
    output: list[str] = []
    for value in values:
        if value in valid and value not in output:
            output.append(value)
        if len(output) == limit:
            break
    return output


async def _request_groq_layout(payload: dict[str, Any]) -> DashboardLayoutPlan:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    response = await AsyncGroq(api_key=key).chat.completions.create(
        model=MODEL_NAME, temperature=0.2, max_completion_tokens=700,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON only: title, selected_kpi_ids, selected_trend_ids, selected_anomaly_ids, selected_insight_ids, selected_recommendation_ids, include_forecast, section_order:[{id,chart_type?}]. Select only supplied IDs. This is a layout plan only: never rewrite or calculate values."},
            {"role": "user", "content": json.dumps(payload, default=str, separators=(",", ":"))},
        ],
    )
    try:
        return DashboardLayoutPlan.model_validate_json(response.choices[0].message.content or "{}")
    except ValidationError as exc:
        raise RuntimeError(f"Invalid Groq dashboard layout: {exc}") from exc


def _fallback_plan(kpis: list[dict[str, Any]], trends: list[dict[str, Any]], anomalies: list[dict[str, Any]], insights: list[dict[str, Any]], recommendations: list[dict[str, Any]], forecast: dict[str, Any] | None) -> DashboardLayoutPlan:
    return DashboardLayoutPlan(title="Business Intelligence Dashboard", selected_kpi_ids=_ids(kpis, MAX_DASHBOARD_KPIS), selected_trend_ids=_ids(trends, 1), selected_anomaly_ids=_ids(anomalies, MAX_DASHBOARD_ANOMALIES), selected_insight_ids=_ids(insights, MAX_DASHBOARD_INSIGHTS), selected_recommendation_ids=_ids(recommendations, MAX_DASHBOARD_RECOMMENDATIONS), include_forecast=bool((forecast or {}).get("forecast")), section_order=[DashboardSection(id="kpis", chart_type="table"), DashboardSection(id="timeline", chart_type="line"), DashboardSection(id="details", chart_type="table")])


def _validated_plan(plan: DashboardLayoutPlan, fallback: DashboardLayoutPlan, kpis: list[dict[str, Any]], trends: list[dict[str, Any]], anomalies: list[dict[str, Any]], insights: list[dict[str, Any]], recommendations: list[dict[str, Any]], forecast: dict[str, Any] | None) -> DashboardLayoutPlan:
    def select(values: list[str], items: list[dict[str, Any]], defaults: list[str], limit: int) -> list[str]:
        valid = {str(item["id"]) for item in items if item.get("id")}
        selected = _dedupe_selected(values, valid, limit)
        return selected or _dedupe_selected(defaults, valid, limit)
    sections: list[DashboardSection] = []
    seen_sections: set[str] = set()
    for section in plan.section_order:
        if section.id not in seen_sections:
            sections.append(section)
            seen_sections.add(section.id)
    return plan.model_copy(update={
        "selected_kpi_ids": select(plan.selected_kpi_ids, kpis, fallback.selected_kpi_ids, MAX_DASHBOARD_KPIS),
        "selected_trend_ids": select(plan.selected_trend_ids, trends, fallback.selected_trend_ids, MAX_DASHBOARD_TRENDS),
        "selected_anomaly_ids": select(plan.selected_anomaly_ids, anomalies, fallback.selected_anomaly_ids, MAX_DASHBOARD_ANOMALIES),
        "selected_insight_ids": select(plan.selected_insight_ids, insights, fallback.selected_insight_ids, MAX_DASHBOARD_INSIGHTS),
        "selected_recommendation_ids": select(plan.selected_recommendation_ids, recommendations, fallback.selected_recommendation_ids, MAX_DASHBOARD_RECOMMENDATIONS),
        "include_forecast": bool(plan.include_forecast and (forecast or {}).get("forecast")),
        "section_order": sections or fallback.section_order,
    })


def _format(value: Any) -> str:
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def _dataset_summary(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    columns = profile.get("column_profiles") or []
    date = prepared.get("date_column")
    date_profile = next((item for item in columns if item.get("name") == date), {})
    missing = sum(int(item.get("null_count") or 0) for item in columns if isinstance(item, dict))
    rows, count = int(profile.get("row_count") or 0), int(profile.get("column_count") or len(columns))
    cells = rows * count
    return {"fileName": str(prepared.get("file_name") or prepared.get("source_file_name") or "Prepared dataset"), "rowCount": rows, "columnCount": count, "timeField": date, "period": ({"start": str(date_profile.get("date_minimum")), "end": str(date_profile.get("date_maximum")), "label": f"{date_profile.get('date_minimum')} to {date_profile.get('date_maximum')}"} if date_profile.get("date_minimum") and date_profile.get("date_maximum") else None), "measures": list(prepared.get("primary_measures") or []), "dimensions": list(prepared.get("dimension_candidates") or []), "quality": {"completenessPercent": round((cells - missing) / cells * 100, 2) if cells else 100.0, "missingValueCount": missing, "duplicateRowCount": int((prepared.get("cleaning_report") or {}).get("duplicate_rows_removed") or 0)}, "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


def _supporting_charts(
    prepared: dict[str, Any],
    trends: dict[str, dict[str, Any]],
    selected_trends: list[str],
) -> list[dict[str, Any]]:
    """Build deterministic, schema-compatible supporting charts from source data."""
    charts: list[dict[str, Any]] = []
    used_types: set[str] = set()
    for trend_id in selected_trends[1:]:
        trend = trends[trend_id]
        points = trend.get("points") or []
        if not points or "line" in used_types:
            continue
        charts.append({"id": f"chart_{trend_id}", "type": "line", "title": str(trend.get("title") or trend_id), "subtitle": None, "valueFormat": "number", "categories": [str(point.get("period")) for point in points], "series": [{"id": f"{trend_id}_values", "name": str(trend.get("measure") or "Value"), "data": [float(point.get("value") or 0) for point in points]}], "layout": {"columnSpan": 1, "rowSpan": 1}})
        used_types.add("line")

    path = Path(str(prepared.get("prepared_file_path") or ""))
    if not path.is_file():
        return charts[:4]
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return charts[:4]
    measures = [str(value) for value in prepared.get("primary_measures") or [] if value in df]
    dimensions = [str(value) for value in prepared.get("dimension_candidates") or [] if value in df]
    if not measures:
        return charts[:4]
    measure = measures[0]
    for dimension in dimensions:
        values = pd.to_numeric(df[measure], errors="coerce")
        grouped = pd.DataFrame({dimension: df[dimension].astype(str), measure: values}).dropna().groupby(dimension, observed=True)[measure].sum().sort_values(ascending=False).head(10)
        if grouped.empty:
            continue
        if "bar" not in used_types:
            charts.append({"id": f"chart_{dimension}_{measure}", "type": "bar", "title": f"{measure.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}", "subtitle": None, "valueFormat": "number", "categories": [str(label) for label in grouped.index], "series": [{"id": f"{dimension}_{measure}", "name": measure.replace("_", " ").title(), "data": [float(value) for value in grouped.values]}], "layout": {"columnSpan": 1, "rowSpan": 1}})
            used_types.add("bar")
        elif "donut" not in used_types:
            charts.append({"id": f"chart_share_{dimension}_{measure}", "type": "donut", "title": f"{measure.replace('_', ' ').title()} share by {dimension.replace('_', ' ').title()}", "subtitle": None, "valueFormat": "number", "segments": [{"id": f"{dimension}_{index}", "label": str(label), "value": float(value)} for index, (label, value) in enumerate(grouped.items())], "layout": {"columnSpan": 1, "rowSpan": 1}})
            used_types.add("donut")
        if len(charts) >= 4:
            break
    return charts[:4]


def _build_dashboard(prepared: dict[str, Any], plan: DashboardLayoutPlan, kpi_output: dict[str, Any] | None, anomaly_output: dict[str, Any] | None, forecasting_output: dict[str, Any] | None, synthesis: dict[str, Any]) -> DashboardResponse:
    kpis = {str(item.get("id")): item for item in (kpi_output or {}).get("kpis", []) if item.get("id")}
    trends = {str(item.get("id")): item for item in (kpi_output or {}).get("trends", []) if item.get("id")}
    anomalies = {str(item.get("id")): item for item in (anomaly_output or {}).get("anomalies", []) if item.get("id")}
    insights = {str(item.get("id")): item for item in synthesis.get("key_insights", []) if item.get("id")}
    recommendations = {str(item.get("id")): item for item in synthesis.get("recommendations", []) if item.get("id")}
    selected_kpis = _dedupe_selected(plan.selected_kpi_ids, set(kpis), MAX_DASHBOARD_KPIS)
    selected_trends = _dedupe_selected(plan.selected_trend_ids, set(trends), MAX_DASHBOARD_TRENDS)
    selected_anomalies = _dedupe_selected(plan.selected_anomaly_ids, set(anomalies), MAX_DASHBOARD_ANOMALIES)
    selected_insights = _dedupe_selected(plan.selected_insight_ids, set(insights), MAX_DASHBOARD_INSIGHTS)
    selected_recommendations = _dedupe_selected(plan.selected_recommendation_ids, set(recommendations), MAX_DASHBOARD_RECOMMENDATIONS)
    dashboard_kpis = [{"id": item_id, "title": str(kpis[item_id].get("title") or item_id), "value": _format(kpis[item_id].get("value")), "rawValue": kpis[item_id].get("raw_value", kpis[item_id].get("value")), "indicator": {"kind": "note", "text": f"{kpis[item_id].get('aggregation', 'calculated').title()} of {kpis[item_id].get('measure', 'measure')}"}} for item_id in selected_kpis]
    timeline = None
    if selected_trends:
        trend = trends[selected_trends[0]]
        actual = [
            {
                "period": str(point.get("period")),
                "label": str(point.get("period")),
                "value": point.get("value"),
            }
            for point in trend.get("points", [])
        ]
        forecast_ok = bool(plan.include_forecast and (forecasting_output or {}).get("forecast") and ((forecasting_output or {}).get("measure") == trend.get("measure")))
        forecast = [
            {
                "period": str(point.get("period")),
                "label": str(point.get("period")),
                "value": point.get("value"),
                "lowerBound": point.get("lower_bound"),
                "upperBound": point.get("upper_bound"),
            }
            for point in ((forecasting_output or {}).get("forecast") or [])
        ] if forecast_ok else []
        timeline = {"id": str(trend["id"]), "title": str(trend.get("title") or trend["id"]), "subtitle": None, "granularity": trend.get("granularity", "month"), "unit": trend.get("measure"), "valueFormat": "number", "actual": actual, "anomalies": [{"id": item_id, "period": str(anomalies[item_id].get("period") or "Observed"), "label": str(anomalies[item_id].get("period") or "Observed"), "value": anomalies[item_id].get("observed_value"), "severity": {"informational": "info", "warning": "warning", "critical": "critical"}.get(anomalies[item_id].get("severity"), "info"), "reason": str(anomalies[item_id].get("evidence") or "Specialist anomaly result.")} for item_id in selected_anomalies], "forecast": forecast, "forecastMetadata": {"available": bool(forecast), "model": "TimesFM" if forecast else None, "horizon": int((forecasting_output or {}).get("horizon") or len(forecast)), "horizonUnit": (forecasting_output or {}).get("granularity") or trend.get("granularity", "month"), "target": (forecasting_output or {}).get("measure") if forecast else None, "confidenceLevel": None}}
    supporting = _supporting_charts(prepared, trends, selected_trends)
    anomaly_items = [{"id": item_id, "title": f"Anomaly: {anomalies[item_id].get('metric', item_id)}", "description": str(anomalies[item_id].get("evidence") or "Specialist anomaly result."), "severity": {"informational": "info", "warning": "warning", "critical": "critical"}.get(anomalies[item_id].get("severity"), "info"), "sourceIds": [item_id]} for item_id in selected_anomalies]
    insight_items = [{"id": item_id, "title": insights[item_id]["title"], "description": insights[item_id]["description"], "severity": {"low": "info", "medium": "warning", "high": "critical"}[insights[item_id]["importance"]], "sourceIds": [ref["source_id"] for ref in insights[item_id].get("evidence", [])]} for item_id in selected_insights]
    limitations = [{"id": f"limitation_{index}", "title": "Limitation", "description": str(value), "severity": "info", "sourceIds": []} for index, value in enumerate(synthesis.get("limitations", [])[:6], 1)]
    sections = []
    seen = set()
    for order, section in enumerate(plan.section_order + [DashboardSection(id="kpis"), DashboardSection(id="timeline"), DashboardSection(id="supportingCharts"), DashboardSection(id="details")], 1):
        if section.id not in seen:
            seen.add(section.id); sections.append({"id": section.id, "title": {"kpis": "Key Performance Indicators", "timeline": "Performance Over Time", "supportingCharts": "Supporting Analysis", "details": "Insights and Recommendations"}[section.id], "order": len(sections) + 1, "visible": section.id != "timeline" or timeline is not None})
    api_warnings = []
    if forecasting_output and not forecasting_output.get("forecast"):
        api_warnings.append({"code": "FORECAST_UNAVAILABLE", "message": "Forecasting was unavailable for this dataset.", "component": "forecasting", "recoverable": True})
    if anomaly_output and anomaly_output.get("status") == "partial":
        api_warnings.append({"code": "ANOMALY_ANALYSIS_PARTIAL", "message": "Anomaly analysis completed with limitations.", "component": "anomaly_detection", "recoverable": True})
    response = {"status": "partial", "sessionId": str(prepared.get("session_id") or "pending"), "dashboard": {"title": plan.title, "executiveSummary": str(synthesis.get("executive_summary") or "Specialist results are available for review."), "kpis": dashboard_kpis, "timeline": timeline, "supportingCharts": supporting, "analysis": {"businessSummary": str(synthesis.get("executive_summary") or ""), "keyFindings": [item["description"] for item in insight_items]}, "insights": {"criticalAnomalies": [item for item in anomaly_items if item["severity"] == "critical"], "warnings": [item for item in anomaly_items if item["severity"] != "critical"], "limitations": limitations, "opportunities": insight_items}, "recommendedActions": [{"id": item_id, "title": recommendations[item_id]["title"], "description": recommendations[item_id]["description"], "priority": recommendations[item_id]["priority"], "sourceIds": [ref["source_id"] for ref in recommendations[item_id].get("evidence", [])]} for item_id in selected_recommendations], "datasetSummary": _dataset_summary(prepared), "sections": sections, "layout": {"kpis": {"columns": max(1, min(8, len(dashboard_kpis) or 1)), "maxRows": 2}, "timeline": {"columnSpan": 12}, "supportingCharts": {"columns": 2, "maxRows": 2}, "details": {"columns": 2, "maxRows": 2}}}, "warnings": api_warnings, "errors": []}
    return DashboardResponse.model_validate(response)


class DashboardGenerationAgent:
    async def run(self, prepared_dataset: dict[str, Any], kpi_trend_output: dict[str, Any] | None, anomaly_output: dict[str, Any] | None, forecasting_output: dict[str, Any] | None, synthesis_output: dict[str, Any]) -> DashboardGenerationOutput:
        prepared, synthesis = prepared_dataset if isinstance(prepared_dataset, dict) else {}, synthesis_output if isinstance(synthesis_output, dict) else {}
        kpis, trends, anomalies = (kpi_trend_output or {}).get("kpis", []), (kpi_trend_output or {}).get("trends", []), (anomaly_output or {}).get("anomalies", [])
        insights, recommendations = synthesis.get("key_insights", []), synthesis.get("recommendations", [])
        fallback = _fallback_plan(kpis, trends, anomalies, insights, recommendations, forecasting_output)
        warning = ""
        try:
            plan = await _request_groq_layout({"kpis": [{"id": item.get("id"), "title": item.get("title")} for item in kpis], "trends": [{"id": item.get("id"), "title": item.get("title")} for item in trends], "anomalies": [{"id": item.get("id"), "severity": item.get("severity")} for item in anomalies], "insights": [{"id": item.get("id"), "title": item.get("title")} for item in insights], "recommendations": [{"id": item.get("id"), "title": item.get("title")} for item in recommendations], "forecast_available": bool((forecasting_output or {}).get("forecast"))})
        except Exception as exc:
            plan, warning = fallback, f"Deterministic dashboard layout was used: {exc}"
        plan = _validated_plan(plan, fallback, kpis, trends, anomalies, insights, recommendations, forecasting_output)
        dashboard = _build_dashboard(prepared, plan, kpi_trend_output, anomaly_output, forecasting_output, synthesis)
        return DashboardGenerationOutput(status="complete" if dashboard.dashboard and dashboard.dashboard.kpis else "partial", layout_plan=plan, dashboard=dashboard, warnings=[warning] if warning else [])


dashboard_generation_agent = DashboardGenerationAgent()


async def dashboard_generation_node(state: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["session_id"] = state.get("session_id", prepared.get("session_id", ""))
    result = await dashboard_generation_agent.run(prepared, state.get("kpi_trend_output"), state.get("anomaly_output"), state.get("forecasting_output"), state.get("synthesis_output", {}))
    return {"dashboard_output": result.dashboard.model_dump(mode="json"), "dashboard_layout_plan": result.layout_plan.model_dump(mode="json"), "warnings": result.warnings, "completed_agents": ["dashboard_generation"]}
