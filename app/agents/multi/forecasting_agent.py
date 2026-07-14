"""Independent TimesFM forecasting specialist."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.timesfm_service import MAX_CONTEXT, timesfm_service

MODEL_NAME = "llama-3.3-70b-versatile"
MIN_FORECAST_PERIODS = 12
DEFAULT_HORIZON = 6
MAX_FORECAST_HORIZON = 24
SUPPORTED_AGGREGATIONS = {"sum", "mean", "count"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class ForecastingError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastDefinition(StrictModel):
    id: str
    title: str
    measure: str
    aggregation: str
    date_column: str
    granularity: str
    horizon: int = Field(default=DEFAULT_HORIZON, ge=1, le=MAX_FORECAST_HORIZON)
    group_by: str | None = None
    group_value: str | int | float | bool | None = None


class ForecastPlan(StrictModel):
    forecast: ForecastDefinition
    limitations: list[str] = Field(default_factory=list)


class HistoricalPoint(StrictModel):
    period: str
    value: float


class ForecastPoint(StrictModel):
    period: str
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class ForecastingOutput(StrictModel):
    status: Literal["complete", "partial"] = "partial"
    series_id: str | None = None
    title: str | None = None
    measure: str | None = None
    aggregation: str | None = None
    granularity: str | None = None
    horizon: int | None = None
    historical: list[HistoricalPoint] = Field(default_factory=list)
    forecast: list[ForecastPoint] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _path(prepared: dict[str, Any]) -> Path:
    path = Path(str(prepared.get("prepared_file_path") or ""))
    if not path.is_file(): raise ForecastingError("prepared_dataset must contain an existing prepared CSV path.")
    return path


def _frequency(granularity: str) -> str:
    return {"day": "D", "week": "W-MON", "month": "M", "quarter": "Q", "year": "Y"}[granularity]


def _granularity(prepared: dict[str, Any]) -> str:
    profile = prepared.get("temporal_profile") or {}
    value = profile.get("inferred_frequency") or prepared.get("time_granularity") or "month"
    return value if value in SUPPORTED_GRANULARITIES else "month"


def _metadata(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {"columns": [{"name": item.get("name"), "type": item.get("inferred_type"), "unique_count": item.get("unique_count")} for item in profile.get("column_profiles", []) if isinstance(item, dict)][:80], "row_count": profile.get("row_count"), "primary_measures": prepared.get("primary_measures") or [], "dimension_candidates": prepared.get("dimension_candidates") or [], "date_column": prepared.get("date_column"), "temporal_profile": prepared.get("temporal_profile") or {"inferred_frequency": prepared.get("time_granularity")}, "time_series_candidates": prepared.get("time_series_candidates") or [], "capability_flags": prepared.get("capability_flags") or {}, "limitations": prepared.get("limitations") or []}


async def _request_groq_plan(prepared: dict[str, Any]) -> ForecastPlan:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key: raise ForecastingError("GROQ_API_KEY is missing.")
    response = await AsyncGroq(api_key=key).chat.completions.create(
        model=MODEL_NAME, temperature=0.1, max_completion_tokens=800, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "Return JSON only: {forecast:{id,title,measure,aggregation,date_column,granularity,horizon,group_by?,group_value?},limitations:[]}. Select one supplied forecastable series using aggregation sum, mean, or count and granularity day, week, month, quarter, or year. Do not calculate forecast values."}, {"role": "user", "content": json.dumps(_metadata(prepared), default=str, separators=(",", ":"))}],
    )
    try: return ForecastPlan.model_validate_json(response.choices[0].message.content or "{}")
    except ValidationError as exc: raise ForecastingError(f"Invalid Groq forecast plan: {exc}") from exc


def _numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df and pd.api.types.is_numeric_dtype(df[column])


def _supports(prepared: dict[str, Any], df: pd.DataFrame) -> str | None:
    flags = prepared.get("capability_flags") or {}
    if flags.get("supports_forecasting") is not True: return "Forecasting is not supported by the prepared dataset capability flags."
    if flags.get("has_temporal_data") is False: return "Forecasting requires temporal data."
    date = prepared.get("date_column") or (prepared.get("temporal_profile") or {}).get("date_column")
    if not isinstance(date, str) or date not in df: return "Forecasting requires a prepared date column."
    if not any(_numeric(df, str(x)) for x in prepared.get("primary_measures") or []) and not any(_numeric(df, str(x)) for x in prepared.get("time_series_candidates") or []): return "Forecasting requires a numeric measure."
    periods = pd.to_datetime(df[date], errors="coerce").dropna().dt.to_period(_frequency(_granularity(prepared))).nunique()
    return None if periods >= MIN_FORECAST_PERIODS else f"Forecasting requires at least {MIN_FORECAST_PERIODS} historical periods."


def _fallback(prepared: dict[str, Any], df: pd.DataFrame) -> ForecastPlan:
    candidates = [str(x) for x in prepared.get("time_series_candidates") or [] if _numeric(df, str(x))]
    candidates += [str(x) for x in prepared.get("primary_measures") or [] if _numeric(df, str(x)) and str(x) not in candidates]
    if not candidates: raise ForecastingError("No forecastable numeric measure is available.")
    date = str(prepared.get("date_column") or (prepared.get("temporal_profile") or {}).get("date_column"))
    measure = candidates[0]
    return ForecastPlan(forecast=ForecastDefinition(id=f"forecast_{re.sub(r'[^a-z0-9]+', '_', measure.lower()).strip('_')}", title=f"Forecast {measure.replace('_', ' ').title()}", measure=measure, aggregation="sum", date_column=date, granularity=_granularity(prepared), horizon=DEFAULT_HORIZON), limitations=["Deterministic planning was used because Groq planning was unavailable or invalid."])


def _validate(plan: ForecastPlan, df: pd.DataFrame) -> ForecastDefinition:
    item = plan.forecast
    if not _numeric(df, item.measure) or item.date_column not in df or item.aggregation not in SUPPORTED_AGGREGATIONS or item.granularity not in SUPPORTED_GRANULARITIES:
        raise ForecastingError("Forecast plan references unsupported columns or options.")
    if item.group_by and (item.group_by not in df or item.group_value is None): raise ForecastingError("Forecast grouping requires an existing group and value.")
    return item


def _prepare(df: pd.DataFrame, item: ForecastDefinition) -> pd.Series:
    columns = [item.date_column, item.measure] + ([item.group_by] if item.group_by else [])
    data = df[columns].copy(); data[item.date_column] = pd.to_datetime(data[item.date_column], errors="coerce")
    data = data.dropna(subset=[item.date_column])
    if item.group_by: data = data[data[item.group_by].astype(str) == str(item.group_value)]
    data["period"] = data[item.date_column].dt.to_period(_frequency(item.granularity))
    grouped = data.groupby("period", observed=True)[item.measure].agg("sum" if item.aggregation == "sum" else "mean" if item.aggregation == "mean" else "count").astype(float).sort_index()
    if grouped.empty: raise ForecastingError("The selected series has no valid values.")
    regular = grouped.reindex(pd.period_range(grouped.index.min(), grouped.index.max(), freq=_frequency(item.granularity)))
    # Small internal gaps are linearly interpolated; remaining edge/long gaps are rejected.
    regular = regular.interpolate(limit=2, limit_area="inside")
    if regular.isna().any(): raise ForecastingError("The selected time series has unfillable gaps.")
    if len(regular) < MIN_FORECAST_PERIODS: raise ForecastingError(f"The selected series has fewer than {MIN_FORECAST_PERIODS} usable periods.")
    return regular.iloc[-MAX_CONTEXT:]


class ForecastingAgent:
    async def run(self, prepared_dataset: dict[str, Any]) -> ForecastingOutput:
        if not isinstance(prepared_dataset, dict): raise ForecastingError("prepared_dataset must be a dictionary.")
        df = pd.read_csv(_path(prepared_dataset), low_memory=False)
        limitation = _supports(prepared_dataset, df)
        if limitation: return ForecastingOutput(limitations=[limitation])
        warnings: list[str] = []
        try:
            proposed = await _request_groq_plan(prepared_dataset); definition = _validate(proposed, df); limitations = proposed.limitations
        except Exception as exc:
            warnings.append(str(exc)); proposed = _fallback(prepared_dataset, df); definition = _validate(proposed, df); limitations = proposed.limitations
        try: series = _prepare(df, definition)
        except Exception as exc:
            return ForecastingOutput(series_id=definition.id, title=definition.title, measure=definition.measure, aggregation=definition.aggregation, granularity=definition.granularity, horizon=definition.horizon, limitations=[*limitations, str(exc)], warnings=warnings)
        historical = [HistoricalPoint(period=str(period), value=round(float(value), 6)) for period, value in series.items()]
        try:
            response = await timesfm_service.forecast([float(value) for value in series.values], definition.horizon)
        except Exception as exc:
            return ForecastingOutput(series_id=definition.id, title=definition.title, measure=definition.measure, aggregation=definition.aggregation, granularity=definition.granularity, horizon=definition.horizon, historical=historical, limitations=[*limitations, f"TimesFM forecast was unavailable: {exc}"], warnings=warnings)
        future = pd.period_range(series.index[-1] + 1, periods=definition.horizon, freq=_frequency(definition.granularity))
        forecast = [ForecastPoint(period=str(period), value=round(float(value), 6), lower_bound=round(float(response.lower_bounds[index]), 6) if response.lower_bounds and index < len(response.lower_bounds) else None, upper_bound=round(float(response.upper_bounds[index]), 6) if response.upper_bounds and index < len(response.upper_bounds) else None) for index, (period, value) in enumerate(zip(future, response.values))]
        return ForecastingOutput(status="complete", series_id=definition.id, title=definition.title, measure=definition.measure, aggregation=definition.aggregation, granularity=definition.granularity, horizon=definition.horizon, historical=historical, forecast=forecast, limitations=limitations, warnings=warnings)


forecasting_agent = ForecastingAgent()


async def forecasting_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await forecasting_agent.run(state.get("prepared_dataset", {}))
    except ForecastingError as exc:
        result = ForecastingOutput(limitations=[str(exc)])
    return {"forecasting_output": result.model_dump(mode="json"), "completed_agents": ["forecasting"]}
