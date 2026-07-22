"""KPI and trend specialist. The LLM plans definitions; pandas calculates values."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import (
    KPIDefinition,
    KPIResult,
    KPITrendOutput,
    KPITrendPlan,
    TrendDefinition,
    TrendPoint,
    TrendSeries,
)
from app.services.data.series import (
    aggregation_for_measure,
    period_frequency,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)

MAX_KPIS = 8
MAX_TRENDS = 3
MAX_TREND_SERIES = 10
SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "count", "distinct_count", "min", "max"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class KPITrendError(RuntimeError):
    pass


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
        "source_datasets": prepared.get("source_datasets") or [],
    }


async def _request_plan(prepared: dict[str, Any]) -> KPITrendPlan:
    prompts = render_agent_prompts(
        "multi/kpi_trend",
        payload=_planning_payload(prepared),
    )
    return await request_structured(
        policy=agent_model_policy("kpi_trend"),
        response_model=KPITrendPlan,
        schema_name="kpi_trend_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df and pd.api.types.is_numeric_dtype(df[column])


def _frequency(granularity: str) -> str:
    return period_frequency(granularity)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "metric"


def _fallback_aggregation(measure: str) -> str:
    return aggregation_for_measure(measure)


def _fallback_plan(prepared: dict[str, Any], df: pd.DataFrame) -> KPITrendPlan:
    measures = ranked_measures(prepared, df)
    kpis = [KPIDefinition(id=f"kpi_{_slug(measure)}", title=f"{_fallback_aggregation(measure).title()} {measure.replace('_', ' ').title()}", measure=measure, aggregation=_fallback_aggregation(measure)) for measure in measures[:4]]
    date = selected_date_column(prepared, df)
    primary = select_primary_series(prepared, df)
    trends = []
    if primary and date:
        trends.append(TrendDefinition(id=f"trend_{_slug(primary.measure)}_{primary.granularity}", title=f"{primary.granularity.title()} {primary.measure.replace('_', ' ').title()}", measure=primary.measure, aggregation=primary.aggregation, date_column=date, granularity=primary.granularity))
    return KPITrendPlan(kpis=kpis, trends=trends, limitations=["Deterministic planning was used because LLM planning was unavailable or invalid."])


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
        if item.aggregation not in {"count", "distinct_count"}:
            expected = aggregation_for_measure(item.measure)
            if item.aggregation != expected:
                warnings.append(
                    f"Adjusted KPI `{item.id}` aggregation from "
                    f"`{item.aggregation}` to `{expected}`."
                )
                item = item.model_copy(update={"aggregation": expected})
        used.add(item.id); kpis.append(item)
    for item in plan.trends[:MAX_TRENDS]:
        if item.id in used or item.measure not in df or item.date_column not in df or item.aggregation not in SUPPORTED_AGGREGATIONS or item.granularity not in SUPPORTED_GRANULARITIES:
            warnings.append(f"Rejected trend definition `{item.id}`."); continue
        if item.aggregation not in {"count", "distinct_count"} and not _is_numeric(df, item.measure):
            warnings.append(f"Rejected trend `{item.id}` because its measure is not numeric."); continue
        if item.group_by and (item.group_by not in df or df[item.group_by].nunique(dropna=True) > MAX_TREND_SERIES):
            warnings.append(f"Rejected trend `{item.id}` because its grouping is invalid or high-cardinality."); continue
        if item.aggregation not in {"count", "distinct_count"}:
            expected = aggregation_for_measure(item.measure)
            if item.aggregation != expected:
                warnings.append(
                    f"Adjusted trend `{item.id}` aggregation from "
                    f"`{item.aggregation}` to `{expected}`."
                )
                item = item.model_copy(update={"aggregation": expected})
        used.add(item.id); trends.append(item)
    return KPITrendPlan(kpis=kpis, trends=trends, limitations=plan.limitations), warnings


def _ensure_core_definitions(
    plan: KPITrendPlan,
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> KPITrendPlan:
    """Guarantee useful KPI coverage and one forecast-aligned primary trend."""
    kpis = list(plan.kpis)
    used_measures = {item.measure for item in kpis}
    for measure in ranked_measures(prepared, df):
        if len(kpis) >= 4:
            break
        if measure in used_measures:
            continue
        aggregation = aggregation_for_measure(measure)
        kpis.append(
            KPIDefinition(
                id=f"kpi_{_slug(measure)}",
                title=f"{aggregation.title()} {measure.replace('_', ' ').title()}",
                measure=measure,
                aggregation=aggregation,
            )
        )
        used_measures.add(measure)

    trends = list(plan.trends)
    primary = select_primary_series(prepared, df)
    if primary:
        matching = next(
            (
                item
                for item in trends
                if item.measure == primary.measure
                and item.date_column == primary.date_column
            ),
            None,
        )
        primary_trend = (
            matching.model_copy(
                update={
                    "aggregation": primary.aggregation,
                    "granularity": primary.granularity,
                    "group_by": None,
                }
            )
            if matching
            else TrendDefinition(
                id=f"trend_{_slug(primary.measure)}_{primary.granularity}",
                title=(
                    f"{primary.granularity.title()} "
                    f"{primary.measure.replace('_', ' ').title()}"
                ),
                measure=primary.measure,
                aggregation=primary.aggregation,
                date_column=primary.date_column,
                granularity=primary.granularity,
            )
        )
        trends = [
            primary_trend,
            *[
                item
                for item in trends
                if item.id != primary_trend.id and item is not matching
            ],
        ]
    return plan.model_copy(update={"kpis": kpis[:MAX_KPIS], "trends": trends[:MAX_TRENDS]})


def _aggregate(series: pd.Series, aggregation: str) -> float | int | None:
    if aggregation == "count": return int(series.count())
    if aggregation == "distinct_count": return int(series.nunique(dropna=True))
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty: return None
    value = float(getattr(values, aggregation)())
    return round(value, 6) if math.isfinite(value) else None


def _calculate_kpis(
    df: pd.DataFrame,
    plan: KPITrendPlan,
    prepared: dict[str, Any],
) -> list[KPIResult]:
    results: list[KPIResult] = []
    date_column = selected_date_column(prepared, df)
    granularity = selected_granularity(prepared)
    for item in plan.kpis:
        source = df
        if item.dimension and item.dimension_value is not None:
            source = df[df[item.dimension].astype(str) == str(item.dimension_value)]
        current_period: str | None = None
        previous_period: str | None = None
        previous_value: float | int | None = None
        change_percent: float | None = None
        baseline_period: str | None = None
        baseline_value: float | int | None = None
        baseline_change_percent: float | None = None
        value: float | int | None = None
        if date_column:
            data = source[[date_column, item.measure]].copy()
            data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
            data = data.dropna(subset=[date_column])
            if not data.empty:
                data["_period"] = data[date_column].dt.to_period(
                    _frequency(granularity)
                )
                grouped = (
                    data.groupby("_period", observed=True)[item.measure]
                    .apply(lambda series: _aggregate(series, item.aggregation))
                    .dropna()
                    .sort_index()
                )
                if not grouped.empty:
                    current_period = str(grouped.index[-1])
                    value = grouped.iloc[-1]
                    baseline_period = str(grouped.index[0])
                    baseline_value = grouped.iloc[0]
                    current_number = float(grouped.iloc[-1])
                    baseline_number = float(grouped.iloc[0])
                    if baseline_number != 0:
                        baseline_change_percent = round(
                            (current_number - baseline_number)
                            / abs(baseline_number)
                            * 100,
                            2,
                        )
                    elif current_number == 0:
                        baseline_change_percent = 0.0
                if len(grouped) >= 2:
                    previous_period = str(grouped.index[-2])
                    previous_value = grouped.iloc[-2]
                    current_number = float(grouped.iloc[-1])
                    previous_number = float(grouped.iloc[-2])
                    if previous_number != 0:
                        change_percent = round(
                            (current_number - previous_number)
                            / abs(previous_number)
                            * 100,
                            2,
                        )
                    elif current_number == 0:
                        change_percent = 0.0
        if value is None:
            value = _aggregate(source[item.measure], item.aggregation)
        if value is not None:
            results.append(
                KPIResult(
                    id=item.id,
                    title=item.title,
                    value=value,
                    raw_value=value,
                    aggregation=item.aggregation,
                    measure=item.measure,
                    dimension=item.dimension,
                    current_period=current_period,
                    previous_period=previous_period,
                    previous_value=previous_value,
                    change_percent=change_percent,
                    baseline_period=baseline_period,
                    baseline_value=baseline_value,
                    baseline_change_percent=baseline_change_percent,
                )
            )
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
            proposed = await _request_plan(prepared_dataset)
            plan, validation_warnings = _valid_plan(proposed, df, prepared_dataset)
            warnings.extend(validation_warnings)
            if not plan.kpis and not plan.trends: raise KPITrendError("LLM plan has no valid definitions.")
        except Exception as exc:
            warnings.append(f"{exc}")
            plan = _fallback_plan(prepared_dataset, df)
        plan = _ensure_core_definitions(plan, prepared_dataset, df)
        kpis = _calculate_kpis(df, plan, prepared_dataset)
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
