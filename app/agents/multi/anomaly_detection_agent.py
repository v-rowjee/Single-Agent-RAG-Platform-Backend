"""Independent anomaly-detection specialist using pandas and numpy."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.services.series import (
    aggregation_for_measure,
    is_numeric_measure,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)
from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.prompts import render_agent_prompts

MIN_TIME_PERIODS = 6
MIN_ROLLING_PERIODS = 6
MAX_GROUP_CARDINALITY = 20
MAX_ANALYSES = 3
MAX_ANOMALIES = 10
SUPPORTED_METHODS = {"z_score", "iqr", "rolling_deviation", "percentage_change"}
SUPPORTED_AGGREGATIONS = {"sum", "mean", "count"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}
SCORE_CRITICAL = 4.0
SCORE_WARNING = 3.0
PERCENT_CRITICAL = 40.0
PERCENT_WARNING = 20.0


class AnomalyDetectionError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnomalyDefinition(StrictModel):
    id: str
    measure: str
    method: str
    aggregation: str = "sum"
    date_column: str | None = None
    granularity: str | None = None
    group_by: str | None = None


class AnomalyPlan(StrictModel):
    analyses: list[AnomalyDefinition] = Field(default_factory=list, max_length=MAX_ANALYSES)
    limitations: list[str] = Field(default_factory=list)


class AnomalyResult(StrictModel):
    id: str
    analysis_id: str
    metric: str
    aggregation: str
    granularity: str | None = None
    period: str | None = None
    observed_value: float
    expected_value: float | None = None
    deviation_percentage: float | None = None
    anomaly_score: float | None = None
    severity: Literal["informational", "warning", "critical"]
    method: str
    evidence: str


class AnomalyDetectionOutput(StrictModel):
    status: Literal["complete", "partial"] = "complete"
    anomalies: list[AnomalyResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _path(prepared: dict[str, Any]) -> Path:
    path = Path(str(prepared.get("prepared_file_path") or ""))
    if not path.is_file(): raise AnomalyDetectionError("prepared_dataset must contain an existing prepared CSV path.")
    return path


def _frequency(granularity: str) -> str:
    return {"day": "D", "week": "W-MON", "month": "M", "quarter": "Q", "year": "Y"}[granularity]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "value"


def _metadata(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {"columns": [{"name": p.get("name"), "type": p.get("inferred_type"), "unique_count": p.get("unique_count")} for p in profile.get("column_profiles", []) if isinstance(p, dict)][:80], "row_count": profile.get("row_count"), "primary_measures": prepared.get("primary_measures") or [], "dimension_candidates": prepared.get("dimension_candidates") or [], "date_column": prepared.get("date_column"), "temporal_profile": prepared.get("temporal_profile") or {"inferred_frequency": prepared.get("time_granularity")}, "time_series_candidates": prepared.get("time_series_candidates") or [], "capability_flags": prepared.get("capability_flags") or {}, "limitations": prepared.get("limitations") or []}


async def _request_plan(prepared: dict[str, Any]) -> AnomalyPlan:
    prompts = render_agent_prompts(
        "multi/anomaly_detection",
        payload=_metadata(prepared),
    )
    return await request_structured(
        policy=agent_model_policy("anomaly_detection"),
        response_model=AnomalyPlan,
        schema_name="anomaly_detection_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _numeric(df: pd.DataFrame, value: str) -> bool:
    return is_numeric_measure(df, value)


def _fallback(prepared: dict[str, Any], df: pd.DataFrame) -> AnomalyPlan:
    measures = ranked_measures(prepared, df)
    if not measures: return AnomalyPlan(limitations=["No numeric measure is available for anomaly detection."])
    date = selected_date_column(prepared, df)
    periods = 0
    if isinstance(date, str) and date in df:
        periods = pd.to_datetime(df[date], errors="coerce").dropna().dt.to_period("M").nunique()
    if isinstance(date, str) and date in df and periods >= MIN_TIME_PERIODS:
        granularity = selected_granularity(prepared)
        definition = AnomalyDefinition(id=f"{granularity}_{_slug(measures[0])}_rolling", measure=measures[0], method="rolling_deviation", aggregation=aggregation_for_measure(measures[0]), date_column=date, granularity=granularity)
    else:
        definition = AnomalyDefinition(id=f"{_slug(measures[0])}_iqr", measure=measures[0], method="iqr")
    return AnomalyPlan(analyses=[definition], limitations=["Anomaly - Deterministic planning was used because LLM planning was unavailable or invalid."])


def _ensure_primary_temporal_analysis(
    analyses: list[AnomalyDefinition],
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[AnomalyDefinition]:
    """Always analyse the dashboard's primary time series at its own grain.

    This prevents a row-level or differently aggregated metric from being
    plotted as though it were an anomaly in the primary timeline.
    """
    primary = select_primary_series(prepared, df)
    if not primary:
        return analyses[:MAX_ANALYSES]

    dates = pd.to_datetime(df[primary.date_column], errors="coerce").dropna()
    periods = dates.dt.to_period(_frequency(primary.granularity)).nunique()
    if periods < MIN_TIME_PERIODS:
        return analyses[:MAX_ANALYSES]

    canonical = AnomalyDefinition(
        id=(
            f"{primary.granularity}_{_slug(primary.measure)}_"
            "rolling_deviation"
        ),
        measure=primary.measure,
        method="rolling_deviation",
        aggregation=primary.aggregation,
        date_column=primary.date_column,
        granularity=primary.granularity,
    )
    supplemental = [
        item
        for item in analyses
        if not (
            item.measure == canonical.measure
            and item.date_column == canonical.date_column
            and item.granularity == canonical.granularity
        )
    ]
    return [canonical, *supplemental][:MAX_ANALYSES]


def _validate(plan: AnomalyPlan, df: pd.DataFrame) -> tuple[list[AnomalyDefinition], list[str]]:
    valid: list[AnomalyDefinition] = []; warnings: list[str] = []; ids: set[str] = set()
    for item in plan.analyses[:MAX_ANALYSES]:
        if item.id in ids or not _numeric(df, item.measure) or item.method not in SUPPORTED_METHODS or item.aggregation not in SUPPORTED_AGGREGATIONS:
            warnings.append(f"Rejected anomaly analysis `{item.id}`."); continue
        temporal = item.date_column is not None or item.granularity is not None
        if temporal and (item.date_column not in df or item.granularity not in SUPPORTED_GRANULARITIES):
            warnings.append(f"Rejected temporal anomaly analysis `{item.id}`."); continue
        if item.group_by and (item.group_by not in df or df[item.group_by].nunique(dropna=True) > MAX_GROUP_CARDINALITY):
            warnings.append(f"Rejected anomaly grouping for `{item.id}`."); continue
        expected = aggregation_for_measure(item.measure)
        if item.aggregation != "count" and item.aggregation != expected:
            warnings.append(
                f"Adjusted anomaly analysis `{item.id}` aggregation "
                f"from `{item.aggregation}` to `{expected}`."
            )
            item = item.model_copy(update={"aggregation": expected})
        ids.add(item.id); valid.append(item)
    return valid, warnings


def _aggregate(series: pd.Series, aggregation: str) -> float:
    if aggregation == "count": return float(series.count())
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum() if aggregation == "sum" else values.mean())


def _series(df: pd.DataFrame, item: AnomalyDefinition) -> list[tuple[str | None, pd.Series]]:
    columns = [item.measure] + ([item.date_column] if item.date_column else []) + ([item.group_by] if item.group_by else [])
    data = df[columns].copy()
    groups = [(None, data)] if not item.group_by else [(str(group), group_data) for group, group_data in data.groupby(item.group_by, observed=True)]
    output: list[tuple[str | None, pd.Series]] = []
    for group, group_data in groups:
        if not item.date_column:
            output.append((group, pd.to_numeric(group_data[item.measure], errors="coerce").dropna().reset_index(drop=True)))
            continue
        group_data[item.date_column] = pd.to_datetime(group_data[item.date_column], errors="coerce")
        group_data = group_data.dropna(subset=[item.date_column]); group_data["period"] = group_data[item.date_column].dt.to_period(_frequency(item.granularity or "month"))
        output.append((group, group_data.groupby("period", observed=True)[item.measure].apply(lambda x: _aggregate(x, item.aggregation)).sort_index()))
    return output


def _severity(score: float | None, percentage: float | None) -> Literal["informational", "warning", "critical"]:
    if percentage is not None:
        return "critical" if abs(percentage) >= PERCENT_CRITICAL else "warning" if abs(percentage) >= PERCENT_WARNING else "informational"
    value = abs(score or 0.0)
    return "critical" if value >= SCORE_CRITICAL else "warning" if value >= SCORE_WARNING else "informational"


def _result(item: AnomalyDefinition, period: Any, observed: float, expected: float | None, score: float | None, percentage: float | None, group: str | None = None) -> AnomalyResult:
    label = str(period) if period is not None else None
    stable_id = "_".join(part for part in [item.granularity or "row", _slug(item.measure), _slug(group or ""), _slug(label or "value"), item.method] if part)
    evidence = (f"{item.group_by}={group}; " if group is not None else "") + f"Observed {observed:.2f}" + (f" versus expected {expected:.2f}" if expected is not None else "")
    return AnomalyResult(id=stable_id, analysis_id=item.id, metric=item.measure, aggregation=item.aggregation, granularity=item.granularity, period=label, observed_value=round(observed, 6), expected_value=round(expected, 6) if expected is not None else None, deviation_percentage=round(percentage, 6) if percentage is not None else None, anomaly_score=round(score, 6) if score is not None else None, severity=_severity(score, percentage), method=item.method, evidence=evidence)


def _detect(item: AnomalyDefinition, values: pd.Series, group: str | None = None) -> list[AnomalyResult]:
    if len(values) < (MIN_TIME_PERIODS if item.date_column else 4): return []
    output: list[AnomalyResult] = []
    numeric = values.astype(float)
    if item.method == "z_score":
        mean, std = float(numeric.mean()), float(numeric.std(ddof=0))
        if std == 0: return []
        for period, value in numeric.items():
            score = (float(value) - mean) / std
            if abs(score) >= SCORE_WARNING: output.append(_result(item, period, float(value), mean, score, None, group))
    elif item.method == "iqr":
        q1, q3 = float(numeric.quantile(.25)), float(numeric.quantile(.75)); iqr = q3 - q1
        if iqr == 0: return []
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for period, value in numeric.items():
            if value < lower or value > upper:
                expected = q3 if value > upper else q1
                score = abs(float(value) - expected) / iqr
                output.append(_result(item, period, float(value), expected, score, None, group))
    elif item.method == "rolling_deviation":
        for position, (period, value) in enumerate(numeric.items()):
            history = numeric.iloc[max(0, position - MIN_ROLLING_PERIODS):position]
            if len(history) < MIN_ROLLING_PERIODS: continue
            expected, std = float(history.mean()), float(history.std(ddof=0))
            if std == 0: continue
            score = (float(value) - expected) / std
            if abs(score) >= SCORE_WARNING: output.append(_result(item, period, float(value), expected, score, None, group))
    else:
        for position, (period, value) in enumerate(numeric.items()):
            if position == 0: continue
            previous = float(numeric.iloc[position - 1])
            if previous == 0: continue
            percentage = (float(value) - previous) / abs(previous) * 100
            if abs(percentage) >= PERCENT_WARNING: output.append(_result(item, period, float(value), previous, None, percentage, group))
    return output


class AnomalyDetectionAgent:
    async def run(self, prepared_dataset: dict[str, Any]) -> AnomalyDetectionOutput:
        if not isinstance(prepared_dataset, dict): raise AnomalyDetectionError("prepared_dataset must be a dictionary.")
        df = pd.read_csv(_path(prepared_dataset), low_memory=False)
        warnings: list[str] = []
        try:
            proposed = await _request_plan(prepared_dataset); analyses, validation = _validate(proposed, df); warnings.extend(validation)
            if not analyses: raise AnomalyDetectionError("LLM plan has no valid analyses.")
            limitations = proposed.limitations
        except Exception as exc:
            warnings.append(str(exc)); fallback = _fallback(prepared_dataset, df); analyses, validation = _validate(fallback, df); warnings.extend(validation); limitations = fallback.limitations
        analyses = _ensure_primary_temporal_analysis(
            analyses,
            prepared_dataset,
            df,
        )
        anomalies: list[AnomalyResult] = []
        for item in analyses:
            for group, values in _series(df, item):
                anomalies.extend(_detect(item, values, group))
        anomalies.sort(key=lambda result: (result.severity != "critical", result.severity != "warning", -(abs(result.anomaly_score or result.deviation_percentage or 0))))
        return AnomalyDetectionOutput(anomalies=anomalies[:MAX_ANOMALIES], warnings=warnings, limitations=[*(prepared_dataset.get("limitations") or []), *limitations])


anomaly_detection_agent = AnomalyDetectionAgent()


async def anomaly_detection_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await anomaly_detection_agent.run(state.get("prepared_dataset", {}))
    except AnomalyDetectionError as exc:
        result = AnomalyDetectionOutput(status="partial", limitations=[str(exc)])
    return {"anomaly_output": result.model_dump(mode="json"), "completed_agents": ["anomaly_detection"]}
