"""KPI and trend specialist.  Groq chooses definitions; pandas calculates values."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_KPIS = 8
MAX_TRENDS = 3
MAX_TREND_SERIES = 10
SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "count", "distinct_count", "min", "max"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class KPITrendError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KPIDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    dimension: str | None = None
    dimension_value: str | int | float | bool | None = None


class TrendDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    date_column: str
    granularity: str
    group_by: str | None = None


class KPITrendPlan(StrictModel):
    kpis: list[KPIDefinition] = Field(default_factory=list, max_length=MAX_KPIS)
    trends: list[TrendDefinition] = Field(default_factory=list, max_length=MAX_TRENDS)
    limitations: list[str] = Field(default_factory=list)


class KPIResult(StrictModel):
    id: str
    title: str
    value: float | int
    raw_value: float | int
    aggregation: str
    measure: str
    dimension: str | None = None


class TrendPoint(StrictModel):
    period: str
    value: float | int


class TrendSeries(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    granularity: str
    group: str | None = None
    points: list[TrendPoint] = Field(default_factory=list)


class KPITrendOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    kpis: list[KPIResult] = Field(default_factory=list)
    trends: list[TrendSeries] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def _path(prepared: dict[str, Any]) -> Path:
    value = prepared.get("prepared_file_path")
    path = Path(str(value or ""))
    if not path.is_file():
        raise KPITrendError("prepared_dataset must contain an existing prepared CSV path.")
    return path


def _columns_metadata(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = (prepared.get("dataset_profile") or {}).get("column_profiles") or []
    return [
        {"name": item.get("name"), "type": item.get("inferred_type"), "unique_count": item.get("unique_count")}
        for item in profiles if isinstance(item, dict)
    ][:80]


def _planning_payload(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {
        "columns": _columns_metadata(prepared), "row_count": profile.get("row_count"),
        "primary_measures": prepared.get("primary_measures") or [],
        "dimension_candidates": prepared.get("dimension_candidates") or [],
        "date_column": prepared.get("date_column"),
        "temporal_profile": prepared.get("temporal_profile") or {"inferred_frequency": prepared.get("time_granularity")},
        "time_series_candidates": prepared.get("time_series_candidates") or [],
        "capability_flags": prepared.get("capability_flags") or {},
        "limitations": prepared.get("limitations") or [],
    }


async def _request_groq_plan(prepared: dict[str, Any]) -> KPITrendPlan:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise KPITrendError("GROQ_API_KEY is missing.")
    response = await AsyncGroq(api_key=api_key).chat.completions.create(
        model=MODEL_NAME, temperature=0.1, max_completion_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON only: {kpis:[{id,title,measure,aggregation,dimension?,dimension_value?}],trends:[{id,title,measure,aggregation,date_column,granularity,group_by?}],limitations:[]}. Use only supplied columns and supported aggregations sum, mean, median, count, distinct_count, min, max and granularities day, week, month, quarter, year. Do not calculate values."},
            {"role": "user", "content": json.dumps(_planning_payload(prepared), default=str, separators=(",", ":"))},
        ],
    )
    try:
        return KPITrendPlan.model_validate_json(response.choices[0].message.content or "{}")
    except ValidationError as exc:
        raise KPITrendError(f"Invalid Groq KPI plan: {exc}") from exc


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df and pd.api.types.is_numeric_dtype(df[column])


def _frequency(granularity: str) -> str:
    return {"day": "D", "week": "W-MON", "month": "M", "quarter": "Q", "year": "Y"}[granularity]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "metric"


def _fallback_aggregation(measure: str) -> str:
    additive_words = ("revenue", "sales", "amount", "cost", "profit", "order", "quantity", "count")
    return "sum" if any(word in measure.lower() for word in additive_words) else "mean"


def _fallback_plan(prepared: dict[str, Any], df: pd.DataFrame) -> KPITrendPlan:
    measures = [str(v) for v in prepared.get("primary_measures") or [] if _is_numeric(df, str(v))]
    measures = measures or [str(c) for c in df if _is_numeric(df, str(c))]
    kpis = [KPIDefinition(id=f"kpi_{_slug(measure)}", title=f"{_fallback_aggregation(measure).title()} {measure.replace('_', ' ').title()}", measure=measure, aggregation=_fallback_aggregation(measure)) for measure in measures[:4]]
    date = prepared.get("date_column")
    trends = []
    if measures and isinstance(date, str) and date in df:
        trends.append(TrendDefinition(id=f"trend_{_slug(measures[0])}_monthly", title=f"Monthly {measures[0].replace('_', ' ').title()}", measure=measures[0], aggregation=_fallback_aggregation(measures[0]), date_column=date, granularity="month"))
    return KPITrendPlan(kpis=kpis, trends=trends, limitations=["Deterministic planning was used because Groq planning was unavailable or invalid."])


def _valid_plan(plan: KPITrendPlan, df: pd.DataFrame, prepared: dict[str, Any]) -> tuple[KPITrendPlan, list[str]]:
    warnings: list[str] = []
    kpis: list[KPIDefinition] = []
    trends: list[TrendDefinition] = []
    used: set[str] = set()
    for item in plan.kpis[:MAX_KPIS]:
        if item.id in used or item.measure not in df or item.aggregation not in SUPPORTED_AGGREGATIONS:
            warnings.append(f"Rejected KPI definition `{item.id}`."); continue
        if item.aggregation not in {"count", "distinct_count"} and not _is_numeric(df, item.measure):
            warnings.append(f"Rejected KPI `{item.id}` because its measure is not numeric."); continue
        if item.dimension and (item.dimension not in df or df[item.dimension].nunique(dropna=True) > MAX_TREND_SERIES):
            warnings.append(f"Rejected KPI `{item.id}` because its dimension is invalid or high-cardinality."); continue
        used.add(item.id); kpis.append(item)
    for item in plan.trends[:MAX_TRENDS]:
        if item.id in used or item.measure not in df or item.date_column not in df or item.aggregation not in SUPPORTED_AGGREGATIONS or item.granularity not in SUPPORTED_GRANULARITIES:
            warnings.append(f"Rejected trend definition `{item.id}`."); continue
        if item.aggregation not in {"count", "distinct_count"} and not _is_numeric(df, item.measure):
            warnings.append(f"Rejected trend `{item.id}` because its measure is not numeric."); continue
        if item.group_by and (item.group_by not in df or df[item.group_by].nunique(dropna=True) > MAX_TREND_SERIES):
            warnings.append(f"Rejected trend `{item.id}` because its grouping is invalid or high-cardinality."); continue
        used.add(item.id); trends.append(item)
    return KPITrendPlan(kpis=kpis, trends=trends, limitations=plan.limitations), warnings


def _aggregate(series: pd.Series, aggregation: str) -> float | int | None:
    if aggregation == "count": return int(series.count())
    if aggregation == "distinct_count": return int(series.nunique(dropna=True))
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty: return None
    value = float(getattr(values, aggregation)())
    return round(value, 6) if math.isfinite(value) else None


def _calculate_kpis(df: pd.DataFrame, plan: KPITrendPlan) -> list[KPIResult]:
    results: list[KPIResult] = []
    for item in plan.kpis:
        source = df
        if item.dimension and item.dimension_value is not None:
            source = df[df[item.dimension].astype(str) == str(item.dimension_value)]
        value = _aggregate(source[item.measure], item.aggregation)
        if value is not None:
            results.append(KPIResult(id=item.id, title=item.title, value=value, raw_value=value, aggregation=item.aggregation, measure=item.measure, dimension=item.dimension))
    return results


def _calculate_trends(df: pd.DataFrame, plan: KPITrendPlan) -> tuple[list[TrendSeries], list[str]]:
    result: list[TrendSeries] = []; warnings: list[str] = []
    for item in plan.trends:
        cols = [item.date_column, item.measure] + ([item.group_by] if item.group_by else [])
        data = df[cols].copy(); data[item.date_column] = pd.to_datetime(data[item.date_column], errors="coerce")
        data = data.dropna(subset=[item.date_column])
        if data.empty: warnings.append(f"Trend `{item.id}` has no valid dates."); continue
        data["_period"] = data[item.date_column].dt.to_period(_frequency(item.granularity))
        groups = [(None, data)] if not item.group_by else [(str(group), group_df) for group, group_df in data.groupby(item.group_by, observed=True)]
        for group, group_df in groups[:MAX_TREND_SERIES]:
            values = group_df.groupby("_period", observed=True)[item.measure].apply(lambda x: _aggregate(x, item.aggregation)).dropna()
            points = [TrendPoint(period=str(period), value=value) for period, value in values.sort_index().items()]
            if points:
                suffix = f"_{_slug(group)}" if group is not None else ""
                result.append(TrendSeries(id=f"{item.id}{suffix}", title=item.title, measure=item.measure, aggregation=item.aggregation, granularity=item.granularity, group=group, points=points))
    return result, warnings


class KPITrendAgent:
    async def run(self, prepared_dataset: dict[str, Any]) -> KPITrendOutput:
        if not isinstance(prepared_dataset, dict): raise KPITrendError("prepared_dataset must be a dictionary.")
        df = pd.read_csv(_path(prepared_dataset), low_memory=False)
        if df.empty: return KPITrendOutput(status="partial", limitations=["Prepared dataset contains no rows."])
        warnings: list[str] = []
        try:
            proposed = await _request_groq_plan(prepared_dataset)
            plan, validation_warnings = _valid_plan(proposed, df, prepared_dataset)
            warnings.extend(validation_warnings)
            if not plan.kpis and not plan.trends: raise KPITrendError("Groq plan has no valid definitions.")
        except Exception as exc:
            warnings.append(f"{exc}")
            plan = _fallback_plan(prepared_dataset, df)
        kpis = _calculate_kpis(df, plan)
        trends, trend_warnings = _calculate_trends(df, plan)
        warnings.extend(trend_warnings)
        return KPITrendOutput(status="complete" if kpis or trends else "partial", kpis=kpis, trends=trends, warnings=warnings, limitations=[*(prepared_dataset.get("limitations") or []), *plan.limitations])


kpi_trend_agent = KPITrendAgent()


async def kpi_trend_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await kpi_trend_agent.run(state.get("prepared_dataset", {}))
    except KPITrendError as exc:
        result = KPITrendOutput(status="partial", limitations=[str(exc)])
    return {"kpi_trend_output": result.model_dump(mode="json"), "completed_agents": ["kpi_trend"]}
