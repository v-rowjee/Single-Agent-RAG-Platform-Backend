"""Validated dashboard assembly from authoritative specialist outputs."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.services.series import (
    aggregation_for_measure,
    is_numeric_measure,
    is_temporal_dimension,
    ranked_measures,
    value_format_for_measure,
)
from app.core.config import agent_model_policy
from app.core.currency import format_currency
from app.core.llm import request_structured
from app.core.prompts import render_agent_prompts
from app.schemas.business_intelligence import DashboardResponse


MAX_DASHBOARD_KPIS, MAX_DASHBOARD_TRENDS = 8, 3
MAX_DASHBOARD_ANOMALIES, MAX_DASHBOARD_INSIGHTS = 6, 6
MAX_DASHBOARD_RECOMMENDATIONS = 5
MIN_SUPPORTING_CHARTS, MAX_SUPPORTING_CHARTS = 2, 4
MAX_SCATTER_POINTS = 200
SUPPORTED_CHART_TYPES = {
    "bar",
    "horizontalBar",
    "stackedBar",
    "donut",
    "pie",
    "scatter",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DashboardSection(StrictModel):
    id: Literal["kpis", "timeline", "supportingCharts", "details"]
    chart_type: Literal["line", "bar", "area", "table"] | None = None


class SupportingChartSpec(StrictModel):
    id: str
    title: str
    type: Literal[
        "bar",
        "horizontalBar",
        "stackedBar",
        "donut",
        "pie",
        "scatter",
    ]
    dimension: str | None = None
    measure: str | None = None
    secondary_measure: str | None = None
    aggregation: Literal["sum", "mean"] | None = None
    x_measure: str | None = None
    y_measure: str | None = None


class DashboardLayoutPlan(StrictModel):
    title: str = Field(min_length=1)
    selected_kpi_ids: list[str] = Field(default_factory=list)
    selected_trend_ids: list[str] = Field(default_factory=list)
    selected_anomaly_ids: list[str] = Field(default_factory=list)
    selected_insight_ids: list[str] = Field(default_factory=list)
    selected_recommendation_ids: list[str] = Field(default_factory=list)
    include_forecast: bool = False
    supporting_chart_specs: list[SupportingChartSpec] = Field(
        default_factory=list,
        max_length=MAX_SUPPORTING_CHARTS,
    )
    section_order: list[DashboardSection] = Field(default_factory=list)


class DashboardGenerationOutput(StrictModel):
    status: Literal["complete", "partial"] = "partial"
    layout_plan: DashboardLayoutPlan
    dashboard: DashboardResponse
    warnings: list[str] = Field(default_factory=list)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


def _ids(items: list[dict[str, Any]], limit: int) -> list[str]:
    return [str(item["id"]) for item in items[:limit] if item.get("id")]


def _dedupe_selected(
    values: list[str],
    valid: set[str],
    limit: int,
) -> list[str]:
    output: list[str] = []
    for value in values:
        if value in valid and value not in output:
            output.append(value)
        if len(output) == limit:
            break
    return output


def _chart_candidates(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    profiles = {
        str(item.get("name")): item
        for item in (prepared.get("dataset_profile") or {}).get(
            "column_profiles",
            [],
        )
        if isinstance(item, dict) and item.get("name")
    }
    dimensions = [
        str(value)
        for value in prepared.get("dimension_candidates") or []
        if value in df
        and not is_temporal_dimension(str(value), prepared)
        and 2 <= df[str(value)].nunique(dropna=True) <= 30
    ]
    measures = ranked_measures(prepared, df)
    return {
        "dimensions": [
            {
                "name": value,
                "unique_count": int(df[value].nunique(dropna=True)),
                "type": profiles.get(value, {}).get("inferred_type"),
            }
            for value in dimensions[:20]
        ],
        "measures": [
            {
                "name": value,
                "aggregation": aggregation_for_measure(value),
                "value_format": value_format_for_measure(value, prepared),
            }
            for value in measures[:16]
        ],
        "requirements": {
            "count": "2-4",
            "unique_types": True,
            "allowed_types": sorted(SUPPORTED_CHART_TYPES),
            "forbid_temporal_dimensions": True,
        },
    }


async def _request_layout(
    payload: dict[str, Any],
) -> DashboardLayoutPlan:
    prompts = render_agent_prompts("multi/dashboard_generation", payload=payload)
    return await request_structured(
        policy=agent_model_policy("dashboard_generation"),
        response_model=DashboardLayoutPlan,
        schema_name="dashboard_layout_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _dimension_score(value: str, cardinality: int) -> int:
    lowered = value.lower()
    priority = (
        ("product_category", 100),
        ("customer_segment", 95),
        ("branch", 90),
        ("sales_channel", 85),
        ("campaign", 80),
        ("membership", 75),
        ("payment", 70),
        ("product", 65),
    )
    semantic = max(
        (score for token, score in priority if token in lowered),
        default=50,
    )
    return semantic - max(0, cardinality - 12)


def _ranked_dimensions(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[str]:
    values = [
        str(value)
        for value in prepared.get("dimension_candidates") or []
        if value in df
        and not is_temporal_dimension(str(value), prepared)
        and 2 <= df[str(value)].nunique(dropna=True) <= 30
    ]
    return sorted(
        values,
        key=lambda value: (
            -_dimension_score(value, int(df[value].nunique(dropna=True))),
            value,
        ),
    )


def _fallback_chart_specs(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[SupportingChartSpec]:
    dimensions = _ranked_dimensions(prepared, df)
    measures = ranked_measures(prepared, df)
    if not dimensions or not measures:
        return []
    primary = measures[0]
    specs: list[SupportingChartSpec] = [
        SupportingChartSpec(
            id=f"chart_{_slug(dimensions[0])}_{_slug(primary)}",
            title=(
                f"{primary.replace('_', ' ').title()} by "
                f"{dimensions[0].replace('_', ' ').title()}"
            ),
            type="horizontalBar",
            dimension=dimensions[0],
            measure=primary,
            aggregation=aggregation_for_measure(primary),
        )
    ]
    proportional_dimension = next(
        (
            value
            for value in dimensions[1:] + dimensions[:1]
            if df[value].nunique(dropna=True) <= 10
        ),
        None,
    )
    if proportional_dimension:
        specs.append(
            SupportingChartSpec(
                id=f"chart_share_{_slug(proportional_dimension)}_{_slug(primary)}",
                title=(
                    f"{primary.replace('_', ' ').title()} share by "
                    f"{proportional_dimension.replace('_', ' ').title()}"
                ),
                type="donut",
                dimension=proportional_dimension,
                measure=primary,
                aggregation=aggregation_for_measure(primary),
            )
        )
    if len(measures) >= 2:
        specs.append(
            SupportingChartSpec(
                id=f"chart_{_slug(measures[0])}_vs_{_slug(measures[1])}",
                title=(
                    f"{measures[0].replace('_', ' ').title()} versus "
                    f"{measures[1].replace('_', ' ').title()}"
                ),
                type="scatter",
                dimension=dimensions[0],
                x_measure=measures[0],
                y_measure=measures[1],
            )
        )
        secondary = next(
            (
                value
                for value in measures[1:]
                if value_format_for_measure(value, prepared)
                == value_format_for_measure(primary, prepared)
            ),
            None,
        )
        if secondary:
            specs.append(
                SupportingChartSpec(
                    id=(
                        f"chart_{_slug(primary)}_{_slug(secondary)}_"
                        f"by_{_slug(dimensions[-1])}"
                    ),
                    title=(
                        f"{primary.replace('_', ' ').title()} and "
                        f"{secondary.replace('_', ' ').title()} by "
                        f"{dimensions[-1].replace('_', ' ').title()}"
                    ),
                    type="stackedBar",
                    dimension=dimensions[-1],
                    measure=primary,
                    secondary_measure=secondary,
                    aggregation=aggregation_for_measure(primary),
                )
            )
    if len(specs) < MIN_SUPPORTING_CHARTS:
        specs.append(
            SupportingChartSpec(
                id=f"chart_bar_{_slug(dimensions[0])}_{_slug(primary)}",
                title=(
                    f"{primary.replace('_', ' ').title()} by "
                    f"{dimensions[0].replace('_', ' ').title()}"
                ),
                type="bar",
                dimension=dimensions[0],
                measure=primary,
                aggregation=aggregation_for_measure(primary),
            )
        )
    return specs[:MAX_SUPPORTING_CHARTS]


def _valid_chart_spec(
    spec: SupportingChartSpec,
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> SupportingChartSpec | None:
    if spec.type not in SUPPORTED_CHART_TYPES:
        return None
    if spec.type == "scatter":
        if (
            not spec.x_measure
            or not spec.y_measure
            or spec.x_measure == spec.y_measure
            or not is_numeric_measure(df, spec.x_measure)
            or not is_numeric_measure(df, spec.y_measure)
        ):
            return None
        dimension = (
            spec.dimension
            if spec.dimension in df
            and not is_temporal_dimension(str(spec.dimension), prepared)
            else None
        )
        return spec.model_copy(update={"dimension": dimension})

    if (
        not spec.dimension
        or spec.dimension not in df
        or is_temporal_dimension(spec.dimension, prepared)
        or not 2 <= df[spec.dimension].nunique(dropna=True) <= 30
        or not spec.measure
        or not is_numeric_measure(df, spec.measure)
    ):
        return None
    if spec.type in {"donut", "pie"} and df[spec.dimension].nunique(dropna=True) > 10:
        return None
    secondary = spec.secondary_measure
    if spec.type == "stackedBar":
        if (
            not secondary
            or secondary == spec.measure
            or not is_numeric_measure(df, secondary)
            or value_format_for_measure(secondary, prepared)
            != value_format_for_measure(spec.measure, prepared)
        ):
            return None
    else:
        secondary = None
    return spec.model_copy(
        update={
            "aggregation": aggregation_for_measure(spec.measure),
            "secondary_measure": secondary,
        }
    )


def _validated_chart_specs(
    proposed: list[SupportingChartSpec],
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[SupportingChartSpec]:
    output: list[SupportingChartSpec] = []
    used_types: set[str] = set()
    for spec in [*proposed, *_fallback_chart_specs(prepared, df)]:
        validated = _valid_chart_spec(spec, prepared, df)
        if not validated or validated.type in used_types:
            continue
        output.append(validated)
        used_types.add(validated.type)
        if len(output) >= MAX_SUPPORTING_CHARTS:
            break
    return output


def _fallback_plan(
    kpis: list[dict[str, Any]],
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    insights: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    forecast: dict[str, Any] | None,
) -> DashboardLayoutPlan:
    return DashboardLayoutPlan(
        title="Business Intelligence Dashboard",
        selected_kpi_ids=_ids(kpis, MAX_DASHBOARD_KPIS),
        selected_trend_ids=_ids(trends, 1),
        selected_anomaly_ids=_ids(anomalies, MAX_DASHBOARD_ANOMALIES),
        selected_insight_ids=_ids(insights, MAX_DASHBOARD_INSIGHTS),
        selected_recommendation_ids=_ids(
            recommendations,
            MAX_DASHBOARD_RECOMMENDATIONS,
        ),
        include_forecast=bool((forecast or {}).get("forecast")),
        section_order=[
            DashboardSection(id="kpis", chart_type="table"),
            DashboardSection(id="timeline", chart_type="line"),
            DashboardSection(id="supportingCharts", chart_type="bar"),
            DashboardSection(id="details", chart_type="table"),
        ],
    )


def _validated_plan(
    plan: DashboardLayoutPlan,
    fallback: DashboardLayoutPlan,
    kpis: list[dict[str, Any]],
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    insights: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    forecast: dict[str, Any] | None,
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> DashboardLayoutPlan:
    def select(
        values: list[str],
        items: list[dict[str, Any]],
        defaults: list[str],
        limit: int,
    ) -> list[str]:
        valid = {str(item["id"]) for item in items if item.get("id")}
        selected = _dedupe_selected(values, valid, limit)
        return selected or _dedupe_selected(defaults, valid, limit)

    sections: list[DashboardSection] = []
    seen_sections: set[str] = set()
    for section in plan.section_order:
        if section.id not in seen_sections:
            sections.append(section)
            seen_sections.add(section.id)
    return plan.model_copy(
        update={
            "selected_kpi_ids": select(
                plan.selected_kpi_ids,
                kpis,
                fallback.selected_kpi_ids,
                MAX_DASHBOARD_KPIS,
            ),
            "selected_trend_ids": select(
                plan.selected_trend_ids,
                trends,
                fallback.selected_trend_ids,
                MAX_DASHBOARD_TRENDS,
            ),
            "selected_anomaly_ids": select(
                plan.selected_anomaly_ids,
                anomalies,
                fallback.selected_anomaly_ids,
                MAX_DASHBOARD_ANOMALIES,
            ),
            "selected_insight_ids": select(
                plan.selected_insight_ids,
                insights,
                fallback.selected_insight_ids,
                MAX_DASHBOARD_INSIGHTS,
            ),
            "selected_recommendation_ids": select(
                plan.selected_recommendation_ids,
                recommendations,
                fallback.selected_recommendation_ids,
                MAX_DASHBOARD_RECOMMENDATIONS,
            ),
            "include_forecast": bool((forecast or {}).get("forecast")),
            "supporting_chart_specs": _validated_chart_specs(
                plan.supporting_chart_specs,
                prepared,
                df,
            ),
            "section_order": sections or fallback.section_order,
        }
    )


def _period_label(value: Any, granularity: str | None) -> str:
    text = str(value or "")
    try:
        if granularity == "month":
            return pd.Period(text, freq="M").strftime("%b %Y")
        if granularity == "quarter":
            period = pd.Period(text, freq="Q")
            return f"Q{period.quarter} {period.year}"
        if granularity == "year":
            return str(pd.Period(text, freq="Y").year)
        if granularity == "day":
            return pd.Timestamp(text).strftime("%d %b %Y")
    except Exception:
        pass
    return text or "previous period"


def _format_kpi(
    value: Any,
    measure: str,
    prepared: dict[str, Any],
) -> str:
    if not isinstance(value, (float, int)):
        return str(value)
    value_format = value_format_for_measure(measure, prepared)
    if value_format == "currency":
        currency = (prepared.get("dataset_profile") or {}).get("currency")
        return format_currency(float(value), currency)
    if value_format == "percentage":
        return f"{float(value):,.2f}%"
    return f"{float(value):,.2f}"


def _dataset_summary(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    columns = profile.get("column_profiles") or []
    date = prepared.get("date_column")
    date_profile = next(
        (item for item in columns if item.get("name") == date),
        {},
    )
    missing = sum(
        int(item.get("null_count") or 0)
        for item in columns
        if isinstance(item, dict)
    )
    rows = int(profile.get("row_count") or 0)
    count = int(profile.get("column_count") or len(columns))
    cells = rows * count
    temporal = prepared.get("temporal_profile") or {}
    start = date_profile.get("date_minimum") or temporal.get("minimum_date")
    end = date_profile.get("date_maximum") or temporal.get("maximum_date")
    return {
        "fileName": str(
            prepared.get("file_name")
            or prepared.get("source_file_name")
            or "Prepared dataset"
        ),
        "rowCount": rows,
        "columnCount": count,
        "timeField": date,
        "period": (
            {"start": str(start), "end": str(end), "label": f"{start} to {end}"}
            if start and end
            else None
        ),
        "measures": list(prepared.get("primary_measures") or []),
        "dimensions": list(prepared.get("dimension_candidates") or []),
        "quality": {
            "completenessPercent": (
                round((cells - missing) / cells * 100, 2) if cells else 100.0
            ),
            "missingValueCount": missing,
            "duplicateRowCount": int(
                (prepared.get("cleaning_report") or {}).get(
                    "duplicate_rows_removed"
                )
                or 0
            ),
        },
        "generatedAt": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _aggregate_grouped(
    df: pd.DataFrame,
    dimension: str,
    measure: str,
    aggregation: str,
) -> pd.Series:
    data = pd.DataFrame(
        {
            dimension: df[dimension].fillna("Missing").astype(str),
            measure: pd.to_numeric(df[measure], errors="coerce"),
        }
    ).dropna(subset=[measure])
    grouped = data.groupby(dimension, observed=True)[measure].agg(aggregation)
    return grouped.sort_values(ascending=False)


def _build_supporting_chart(
    spec: SupportingChartSpec,
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    layout = {"columnSpan": 1, "rowSpan": 1}
    if spec.type == "scatter" and spec.x_measure and spec.y_measure:
        columns = [spec.x_measure, spec.y_measure]
        if spec.dimension:
            columns.append(spec.dimension)
        data = df[columns].copy()
        data[spec.x_measure] = pd.to_numeric(
            data[spec.x_measure],
            errors="coerce",
        )
        data[spec.y_measure] = pd.to_numeric(
            data[spec.y_measure],
            errors="coerce",
        )
        data = data.dropna(subset=[spec.x_measure, spec.y_measure])
        if data.empty:
            return None
        if len(data) > MAX_SCATTER_POINTS:
            positions = np.linspace(
                0,
                len(data) - 1,
                MAX_SCATTER_POINTS,
                dtype=int,
            )
            data = data.iloc[positions]
        return {
            "id": spec.id,
            "type": "scatter",
            "title": spec.title,
            "subtitle": None,
            "valueFormat": value_format_for_measure(
                spec.y_measure,
                prepared,
            ),
            "xAxis": {
                "title": spec.x_measure.replace("_", " ").title(),
                "format": value_format_for_measure(
                    spec.x_measure,
                    prepared,
                ),
            },
            "yAxis": {
                "title": spec.y_measure.replace("_", " ").title(),
                "format": value_format_for_measure(
                    spec.y_measure,
                    prepared,
                ),
            },
            "points": [
                {
                    "x": float(row[spec.x_measure]),
                    "y": float(row[spec.y_measure]),
                    "label": (
                        str(row[spec.dimension])
                        if spec.dimension and pd.notna(row[spec.dimension])
                        else None
                    ),
                }
                for _, row in data.iterrows()
            ],
            "layout": layout,
        }

    if not spec.dimension or not spec.measure or not spec.aggregation:
        return None
    grouped = _aggregate_grouped(
        df,
        spec.dimension,
        spec.measure,
        spec.aggregation,
    )
    if grouped.empty:
        return None
    if spec.type in {"donut", "pie"}:
        grouped = grouped.head(10)
        return {
            "id": spec.id,
            "type": spec.type,
            "title": spec.title,
            "subtitle": None,
            "valueFormat": value_format_for_measure(
                spec.measure,
                prepared,
            ),
            "segments": [
                {
                    "id": f"{_slug(spec.dimension)}_{index}",
                    "label": str(label),
                    "value": float(value),
                }
                for index, (label, value) in enumerate(grouped.items())
            ],
            "layout": layout,
        }

    grouped = grouped.head(10)
    series = [
        {
            "id": f"{_slug(spec.measure)}_values",
            "name": spec.measure.replace("_", " ").title(),
            "data": [float(value) for value in grouped.values],
        }
    ]
    if spec.type == "stackedBar" and spec.secondary_measure:
        secondary = _aggregate_grouped(
            df,
            spec.dimension,
            spec.secondary_measure,
            aggregation_for_measure(spec.secondary_measure),
        ).reindex(grouped.index, fill_value=0)
        series.append(
            {
                "id": f"{_slug(spec.secondary_measure)}_values",
                "name": spec.secondary_measure.replace("_", " ").title(),
                "data": [float(value) for value in secondary.values],
            }
        )
    return {
        "id": spec.id,
        "type": spec.type,
        "title": spec.title,
        "subtitle": None,
        "valueFormat": value_format_for_measure(spec.measure, prepared),
        "categories": [str(label) for label in grouped.index],
        "series": series,
        "layout": layout,
    }


def _supporting_charts(
    prepared: dict[str, Any],
    specs: list[SupportingChartSpec],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    charts = [
        chart
        for spec in specs
        if (chart := _build_supporting_chart(spec, prepared, df)) is not None
    ]
    return charts[:MAX_SUPPORTING_CHARTS]


def _fallback_dashboard_actions(
    selected_kpis: list[str],
    selected_anomalies: list[str],
    forecasting_output: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if selected_anomalies:
        actions.append(
            {
                "id": "action_investigate_anomaly",
                "title": "Investigate the leading anomaly",
                "description": (
                    "Review the records and operating context behind the leading "
                    "anomaly, then document whether corrective action is needed."
                ),
                "priority": "high",
                "sourceIds": [selected_anomalies[0]],
            }
        )
    if selected_kpis:
        actions.append(
            {
                "id": "action_review_kpi",
                "title": "Review KPI drivers",
                "description": (
                    "Break down the primary KPI by the available business "
                    "dimensions and assign an owner for the next-period review."
                ),
                "priority": "medium",
                "sourceIds": [selected_kpis[0]],
            }
        )
    if (forecasting_output or {}).get("series_id") and (
        forecasting_output or {}
    ).get("forecast"):
        actions.append(
            {
                "id": "action_plan_forecast",
                "title": "Plan against the forecast",
                "description": (
                    "Check capacity and budget assumptions against the three "
                    "forecast periods and compare predictions with new actuals."
                ),
                "priority": "medium",
                "sourceIds": [str(forecasting_output["series_id"])],
            }
        )
    generic = [
        {
            "id": "action_review_cadence",
            "title": "Establish a review cadence",
            "description": (
                "Review KPI, segment, and anomaly results every reporting period "
                "so material changes are escalated consistently."
            ),
            "priority": "medium",
            "sourceIds": ["dataset_summary"],
        },
        {
            "id": "action_data_quality",
            "title": "Protect data quality",
            "description": (
                "Resolve recurring missing-value and duplicate-record causes "
                "before the next dashboard refresh."
            ),
            "priority": "medium",
            "sourceIds": ["dataset_summary"],
        },
    ]
    for action in generic:
        if len(actions) >= 3:
            break
        actions.append(action)
    return actions[:MAX_DASHBOARD_RECOMMENDATIONS]


def _build_dashboard(
    prepared: dict[str, Any],
    plan: DashboardLayoutPlan,
    kpi_output: dict[str, Any] | None,
    anomaly_output: dict[str, Any] | None,
    forecasting_output: dict[str, Any] | None,
    synthesis: dict[str, Any],
    df: pd.DataFrame,
) -> DashboardResponse:
    kpis = {
        str(item.get("id")): item
        for item in (kpi_output or {}).get("kpis", [])
        if item.get("id")
    }
    trends = {
        str(item.get("id")): item
        for item in (kpi_output or {}).get("trends", [])
        if item.get("id")
    }
    anomalies = {
        str(item.get("id")): item
        for item in (anomaly_output or {}).get("anomalies", [])
        if item.get("id")
    }
    insights = {
        str(item.get("id")): item
        for item in synthesis.get("key_insights", [])
        if item.get("id")
    }
    recommendations = {
        str(item.get("id")): item
        for item in synthesis.get("recommendations", [])
        if item.get("id")
    }
    selected_kpis = _dedupe_selected(
        plan.selected_kpi_ids,
        set(kpis),
        MAX_DASHBOARD_KPIS,
    )
    selected_trends = _dedupe_selected(
        plan.selected_trend_ids,
        set(trends),
        MAX_DASHBOARD_TRENDS,
    )
    selected_anomalies = _dedupe_selected(
        plan.selected_anomaly_ids,
        set(anomalies),
        MAX_DASHBOARD_ANOMALIES,
    )
    selected_insights = _dedupe_selected(
        plan.selected_insight_ids,
        set(insights),
        MAX_DASHBOARD_INSIGHTS,
    )
    selected_recommendations = _dedupe_selected(
        plan.selected_recommendation_ids,
        set(recommendations),
        MAX_DASHBOARD_RECOMMENDATIONS,
    )

    dashboard_kpis = []
    for item_id in selected_kpis:
        item = kpis[item_id]
        change = item.get("baseline_change_percent")
        if not isinstance(change, (float, int)):
            change = item.get("change_percent")
        baseline_label = _period_label(
            item.get("baseline_period") or item.get("previous_period"),
            (kpi_output or {}).get("trends", [{}])[0].get("granularity")
            if (kpi_output or {}).get("trends")
            else None,
        )
        current_label = _period_label(
            item.get("current_period"),
            (kpi_output or {}).get("trends", [{}])[0].get("granularity")
            if (kpi_output or {}).get("trends")
            else None,
        )
        comparison_range = (
            f"from {baseline_label} to {current_label}"
            if item.get("baseline_period") and item.get("current_period")
            else f"vs {baseline_label}"
        )
        if isinstance(change, (float, int)) and change > 0:
            kind, text = "increase", f"+{float(change):.1f}% {comparison_range}"
        elif isinstance(change, (float, int)) and change < 0:
            kind, text = "decrease", f"{float(change):.1f}% {comparison_range}"
        elif change == 0:
            kind, text = "note", f"0.0% {comparison_range}"
        else:
            kind, text = "note", "No previous-period comparison"
        dashboard_kpis.append(
            {
                "id": item_id,
                "title": str(item.get("title") or item_id),
                "value": _format_kpi(
                    item.get("value"),
                    str(item.get("measure") or ""),
                    prepared,
                ),
                "rawValue": item.get("raw_value", item.get("value")),
                "indicator": {"kind": kind, "text": text},
            }
        )

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
        actual_periods = {point["period"] for point in actual}
        forecast_output = forecasting_output or {}
        forecast_ok = bool(
            plan.include_forecast
            and forecast_output.get("forecast")
            and forecast_output.get("measure") == trend.get("measure")
            and forecast_output.get("aggregation") == trend.get("aggregation")
            and forecast_output.get("granularity") == trend.get("granularity")
        )
        forecast = (
            [
                {
                    "period": str(point.get("period")),
                    "label": str(point.get("period")),
                    "value": point.get("value"),
                    "lowerBound": point.get("lower_bound"),
                    "upperBound": point.get("upper_bound"),
                }
                for point in forecast_output.get("forecast", [])
            ]
            if forecast_ok
            else []
        )
        timeline_anomalies = []
        for anomaly in anomalies.values():
            period = str(anomaly.get("period") or "")
            if (
                anomaly.get("metric") != trend.get("measure")
                or anomaly.get("aggregation") != trend.get("aggregation")
                or anomaly.get("granularity") != trend.get("granularity")
                or period not in actual_periods
            ):
                continue
            timeline_anomalies.append(
                {
                    "id": str(anomaly["id"]),
                    "period": period,
                    "label": period,
                    "value": anomaly.get("observed_value"),
                    "severity": {
                        "informational": "info",
                        "warning": "warning",
                        "critical": "critical",
                    }.get(anomaly.get("severity"), "info"),
                    "reason": str(
                        anomaly.get("evidence")
                        or "Specialist anomaly result."
                    ),
                }
            )
            if len(timeline_anomalies) == MAX_DASHBOARD_ANOMALIES:
                break
        timeline = {
            "id": str(trend["id"]),
            "title": str(trend.get("title") or trend["id"]),
            "subtitle": None,
            "granularity": trend.get("granularity", "month"),
            "unit": (
                (prepared.get("dataset_profile") or {}).get("currency")
                or trend.get("measure")
            ),
            "valueFormat": value_format_for_measure(
                str(trend.get("measure") or ""),
                prepared,
            ),
            "actual": actual,
            "anomalies": timeline_anomalies,
            "forecast": forecast,
            "forecastMetadata": {
                "available": bool(forecast),
                "model": forecast_output.get("model") if forecast else None,
                "horizon": len(forecast),
                "horizonUnit": (
                    forecast_output.get("granularity")
                    or trend.get("granularity", "month")
                ),
                "target": forecast_output.get("measure") if forecast else None,
                "confidenceLevel": (
                    forecast_output.get("confidence_level")
                    if forecast
                    else None
                ),
            },
        }

    supporting = _supporting_charts(
        prepared,
        plan.supporting_chart_specs,
        df,
    )
    anomaly_items = [
        {
            "id": item_id,
            "title": (
                f"Anomaly: {anomalies[item_id].get('metric', item_id)}"
            ),
            "description": str(
                anomalies[item_id].get("evidence")
                or "Specialist anomaly result."
            ),
            "severity": {
                "informational": "info",
                "warning": "warning",
                "critical": "critical",
            }.get(anomalies[item_id].get("severity"), "info"),
            "sourceIds": [item_id],
        }
        for item_id in selected_anomalies
    ]
    insight_items = [
        {
            "id": item_id,
            "title": insights[item_id]["title"],
            "description": insights[item_id]["description"],
            "severity": {
                "low": "info",
                "medium": "warning",
                "high": "critical",
            }[insights[item_id]["importance"]],
            "sourceIds": [
                ref["source_id"]
                for ref in insights[item_id].get("evidence", [])
            ],
        }
        for item_id in selected_insights
    ]
    limitations = [
        {
            "id": f"limitation_{index}",
            "title": "Limitation",
            "description": str(value),
            "severity": "info",
            "sourceIds": [],
        }
        for index, value in enumerate(
            synthesis.get("limitations", [])[:6],
            1,
        )
    ]
    sections = []
    seen = set()
    for section in plan.section_order + [
        DashboardSection(id="kpis"),
        DashboardSection(id="timeline"),
        DashboardSection(id="supportingCharts"),
        DashboardSection(id="details"),
    ]:
        if section.id not in seen:
            seen.add(section.id)
            sections.append(
                {
                    "id": section.id,
                    "title": {
                        "kpis": "Key Performance Indicators",
                        "timeline": "Performance Over Time",
                        "supportingCharts": "Supporting Analysis",
                        "details": "Insights and Recommendations",
                    }[section.id],
                    "order": len(sections) + 1,
                    "visible": (
                        section.id != "timeline" or timeline is not None
                    ),
                }
            )

    actions = [
        {
            "id": item_id,
            "title": recommendations[item_id]["title"],
            "description": recommendations[item_id]["description"],
            "priority": recommendations[item_id]["priority"],
            "sourceIds": [
                ref["source_id"]
                for ref in recommendations[item_id].get("evidence", [])
            ],
        }
        for item_id in selected_recommendations
    ]
    if len(actions) < 3:
        existing = {item["id"] for item in actions}
        for action in _fallback_dashboard_actions(
            selected_kpis,
            selected_anomalies,
            forecasting_output,
        ):
            if len(actions) >= 3:
                break
            if action["id"] not in existing:
                actions.append(action)
                existing.add(action["id"])

    api_warnings = []
    if forecasting_output and not forecasting_output.get("forecast"):
        api_warnings.append(
            {
                "code": "FORECAST_UNAVAILABLE",
                "message": "Forecasting was unavailable for this dataset.",
                "component": "forecasting",
                "recoverable": True,
            }
        )
    if anomaly_output and anomaly_output.get("status") == "partial":
        api_warnings.append(
            {
                "code": "ANOMALY_ANALYSIS_PARTIAL",
                "message": "Anomaly analysis completed with limitations.",
                "component": "anomaly_detection",
                "recoverable": True,
            }
        )
    summary = str(
        synthesis.get("executive_summary")
        or "The available specialist evidence has been summarised for review."
    )
    response = {
        "status": "partial",
        "sessionId": str(prepared.get("session_id") or "pending"),
        "dashboard": {
            "title": plan.title,
            "executiveSummary": summary,
            "kpis": dashboard_kpis,
            "timeline": timeline,
            "supportingCharts": supporting,
            "analysis": {
                "businessSummary": summary,
                "keyFindings": [
                    item["description"] for item in insight_items
                ],
            },
            "insights": {
                "criticalAnomalies": [
                    item
                    for item in anomaly_items
                    if item["severity"] == "critical"
                ],
                "warnings": [
                    item
                    for item in anomaly_items
                    if item["severity"] != "critical"
                ],
                "limitations": limitations,
                "opportunities": insight_items,
            },
            "recommendedActions": actions[:MAX_DASHBOARD_RECOMMENDATIONS],
            "datasetSummary": _dataset_summary(prepared),
            "sections": sections,
            "layout": {
                "kpis": {
                    "columns": max(1, min(8, len(dashboard_kpis) or 1)),
                    "maxRows": 2,
                },
                "timeline": {"columnSpan": 12},
                "supportingCharts": {"columns": 2, "maxRows": 2},
                "details": {"columns": 2, "maxRows": 2},
            },
        },
        "warnings": api_warnings,
        "errors": [],
    }
    return DashboardResponse.model_validate(response)


class DashboardGenerationAgent:
    async def run(
        self,
        prepared_dataset: dict[str, Any],
        kpi_trend_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
        synthesis_output: dict[str, Any],
    ) -> DashboardGenerationOutput:
        prepared = (
            prepared_dataset if isinstance(prepared_dataset, dict) else {}
        )
        synthesis = (
            synthesis_output if isinstance(synthesis_output, dict) else {}
        )
        path = Path(str(prepared.get("prepared_file_path") or ""))
        if not path.is_file():
            raise RuntimeError("Prepared dataset is unavailable for dashboard generation.")
        df = pd.read_csv(path, low_memory=False)
        kpis = (kpi_trend_output or {}).get("kpis", [])
        trends = (kpi_trend_output or {}).get("trends", [])
        anomalies = (anomaly_output or {}).get("anomalies", [])
        insights = synthesis.get("key_insights", [])
        recommendations = synthesis.get("recommendations", [])
        fallback = _fallback_plan(
            kpis,
            trends,
            anomalies,
            insights,
            recommendations,
            forecasting_output,
        )
        warning = ""
        payload = {
            "kpis": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in kpis
            ],
            "trends": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in trends
            ],
            "anomalies": [
                {"id": item.get("id"), "severity": item.get("severity")}
                for item in anomalies
            ],
            "insights": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in insights
            ],
            "recommendations": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in recommendations
            ],
            "forecast_available": bool(
                (forecasting_output or {}).get("forecast")
            ),
            "chart_candidates": _chart_candidates(prepared, df),
        }
        try:
            plan = await _request_layout(payload)
        except Exception as exc:
            plan = fallback
            warning = f"Deterministic dashboard layout was used: {exc}"
        plan = _validated_plan(
            plan,
            fallback,
            kpis,
            trends,
            anomalies,
            insights,
            recommendations,
            forecasting_output,
            prepared,
            df,
        )
        dashboard = _build_dashboard(
            prepared,
            plan,
            kpi_trend_output,
            anomaly_output,
            forecasting_output,
            synthesis,
            df,
        )
        return DashboardGenerationOutput(
            status=(
                "complete"
                if dashboard.dashboard and dashboard.dashboard.kpis
                else "partial"
            ),
            layout_plan=plan,
            dashboard=dashboard,
            warnings=[warning] if warning else [],
        )


dashboard_generation_agent = DashboardGenerationAgent()


async def dashboard_generation_node(state: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["session_id"] = state.get(
        "session_id",
        prepared.get("session_id", ""),
    )
    result = await dashboard_generation_agent.run(
        prepared,
        state.get("kpi_trend_output"),
        state.get("anomaly_output"),
        state.get("forecasting_output"),
        state.get("synthesis_output", {}),
    )
    return {
        "dashboard_output": result.dashboard.model_dump(mode="json"),
        "dashboard_layout_plan": result.layout_plan.model_dump(mode="json"),
        "warnings": result.warnings,
        "completed_agents": ["dashboard_generation"],
    }
