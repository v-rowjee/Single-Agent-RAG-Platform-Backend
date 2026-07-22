"""Independent Chronos-2 forecasting specialist."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.specialists import (
    ForecastDefinition,
    ForecastingOutput,
    ForecastPlan,
    ForecastPoint,
    HistoricalPoint,
)

from app.services.data.series import (
    is_numeric_measure,
    period_frequency,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)
from app.services.forecasting.chronos import MAX_CONTEXT
from app.services.forecasting.service import forecasting_service

# This matches data preparation's capability gate.  Short histories are still
# labelled by the fallback model, while longer series can use Chronos-2.
MIN_FORECAST_PERIODS = 4
DEFAULT_HORIZON = 3
SUPPORTED_AGGREGATIONS = {"sum", "mean", "count"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class ForecastingError(RuntimeError):
    pass


def _path(prepared: dict[str, Any]) -> Path:
    path = Path(str(prepared.get("prepared_file_path") or ""))
    if not path.is_file(): raise ForecastingError("prepared_dataset must contain an existing prepared CSV path.")
    return path


def _frequency(granularity: str) -> str:
    return period_frequency(granularity)


def _granularity(prepared: dict[str, Any]) -> str:
    return selected_granularity(prepared)


def _numeric(df: pd.DataFrame, column: str) -> bool:
    return is_numeric_measure(df, column)


def _supports(prepared: dict[str, Any], df: pd.DataFrame) -> str | None:
    flags = prepared.get("capability_flags") or {}
    if flags.get("supports_forecasting") is not True: return "Forecasting is not supported by the prepared dataset capability flags."
    if flags.get("has_temporal_data") is False: return "Forecasting requires temporal data."
    date = selected_date_column(prepared, df)
    if not isinstance(date, str) or date not in df: return "Forecasting requires a prepared date column."
    if not any(_numeric(df, str(x)) for x in prepared.get("primary_measures") or []) and not any(_numeric(df, str(x)) for x in prepared.get("time_series_candidates") or []): return "Forecasting requires a numeric measure."
    periods = pd.to_datetime(df[date], errors="coerce").dropna().dt.to_period(_frequency(_granularity(prepared))).nunique()
    return None if periods >= MIN_FORECAST_PERIODS else f"Forecasting requires at least {MIN_FORECAST_PERIODS} historical periods."


def _fallback(prepared: dict[str, Any], df: pd.DataFrame) -> ForecastPlan:
    primary = select_primary_series(prepared, df)
    if not primary:
        raise ForecastingError("No forecastable primary series is available.")
    slug = "_".join(
        part for part in primary.measure.lower().replace("-", "_").split("_") if part
    )
    return ForecastPlan(
        forecast=ForecastDefinition(
            id=f"forecast_{slug or 'measure'}",
            title=f"Forecast {primary.measure.replace('_', ' ').title()}",
            measure=primary.measure,
            aggregation=primary.aggregation,
            date_column=primary.date_column,
            granularity=primary.granularity,
            horizon=DEFAULT_HORIZON,
        )
    )


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


def _fallback_forecast(
    series: pd.Series,
    horizon: int,
    granularity: str,
) -> tuple[str, list[float], list[float] | None, list[float] | None]:
    values = np.asarray(series.values, dtype=float)
    seasonal_period = {
        "day": 7,
        "week": 52,
        "month": 12,
        "quarter": 4,
        "year": 0,
    }[granularity]
    non_negative = bool((values >= 0).all())
    if seasonal_period and len(values) >= seasonal_period * 2:
        predictions = [
            float(values[-seasonal_period + (index % seasonal_period)])
            for index in range(horizon)
        ]
        residuals = values[seasonal_period:] - values[:-seasonal_period]
        model = "seasonal_naive"
    else:
        recent = values[-min(12, len(values)) :]
        x = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        predictions = [
            float(slope * (len(recent) - 1 + step) + intercept)
            for step in range(1, horizon + 1)
        ]
        residuals = recent - (slope * x + intercept)
        model = "linear_trend"

    if non_negative:
        predictions = [max(0.0, value) for value in predictions]
    spread = float(np.std(residuals)) * 1.96 if len(residuals) >= 2 else 0.0
    lower = [max(0.0, value - spread) if non_negative else value - spread for value in predictions]
    upper = [value + spread for value in predictions]
    return model, predictions, lower, upper


class ForecastingAgent:
    async def run(self, prepared_dataset: dict[str, Any]) -> ForecastingOutput:
        if not isinstance(prepared_dataset, dict): raise ForecastingError("prepared_dataset must be a dictionary.")
        df = pd.read_csv(_path(prepared_dataset), low_memory=False)
        limitation = _supports(prepared_dataset, df)
        if limitation: return ForecastingOutput(limitations=[limitation])
        warnings: list[str] = []
        proposed = _fallback(prepared_dataset, df)
        definition = _validate(proposed, df)
        limitations = proposed.limitations
        try: series = _prepare(df, definition)
        except Exception as exc:
            return ForecastingOutput(series_id=definition.id, title=definition.title, measure=definition.measure, aggregation=definition.aggregation, granularity=definition.granularity, horizon=definition.horizon, limitations=[*limitations, str(exc)], warnings=warnings)
        historical = [HistoricalPoint(period=str(period), value=round(float(value), 6)) for period, value in series.items()]
        try:
            response = await forecasting_service.forecast(series, definition.horizon)
        except Exception as exc:
            model, values, lower_bounds, upper_bounds = _fallback_forecast(
                series,
                definition.horizon,
                definition.granularity,
            )
            limitations = [
                *limitations,
                f"Chronos-2 was unavailable; {model} fallback was used: {exc}",
            ]
            confidence_level = 0.95
        else:
            model = "Chronos-2"
            values = response.values
            lower_bounds = response.lower_bounds
            upper_bounds = response.upper_bounds
            confidence_level = None
        future = pd.period_range(series.index[-1] + 1, periods=definition.horizon, freq=_frequency(definition.granularity))
        forecast = [ForecastPoint(period=str(period), value=round(float(value), 6), lower_bound=round(float(lower_bounds[index]), 6) if lower_bounds and index < len(lower_bounds) else None, upper_bound=round(float(upper_bounds[index]), 6) if upper_bounds and index < len(upper_bounds) else None) for index, (period, value) in enumerate(zip(future, values))]
        return ForecastingOutput(status="complete", series_id=definition.id, title=definition.title, measure=definition.measure, aggregation=definition.aggregation, granularity=definition.granularity, horizon=definition.horizon, model=model, confidence_level=confidence_level, historical=historical, forecast=forecast, limitations=limitations, warnings=warnings)


forecasting_agent = ForecastingAgent()


async def forecasting_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await forecasting_agent.run(state.get("prepared_dataset", {}))
    except ForecastingError as exc:
        result = ForecastingOutput(limitations=[str(exc)])
    return {"forecasting_output": result.model_dump(mode="json"), "completed_agents": ["forecasting"]}
