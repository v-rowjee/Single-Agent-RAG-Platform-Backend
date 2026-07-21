"""Shared deterministic semantics for KPI, trend, forecast, and chart analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import pandas as pd

from app.core.currency import detect_currency

TimeGranularity: TypeAlias = Literal["day", "week", "month", "quarter", "year"]
SUPPORTED_GRANULARITIES: frozenset[TimeGranularity] = frozenset(
    {"day", "week", "month", "quarter", "year"}
)
TIME_GRANULARITIES: tuple[TimeGranularity, ...] = (
    "day",
    "week",
    "month",
    "quarter",
    "year",
)
TEMPORAL_DIMENSION_NAMES = {
    "date",
    "day",
    "day_of_week",
    "month",
    "month_name",
    "quarter",
    "time",
    "timestamp",
    "week",
    "year",
}
AVERAGE_TOKENS = (
    "average",
    "avg",
    "discount",
    "margin",
    "pct",
    "percent",
    "percentage",
    "price",
    "rate",
    "ratio",
    "score",
)
ADDITIVE_TOKENS = (
    "amount",
    "cost",
    "income",
    "order",
    "profit",
    "quantity",
    "revenue",
    "sales",
    "turnover",
    "unit",
    "volume",
)


@dataclass(frozen=True)
class PrimarySeries:
    measure: str
    aggregation: str
    date_column: str
    granularity: TimeGranularity


def is_numeric_measure(df: pd.DataFrame, column: str) -> bool:
    return (
        column in df
        and pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    )


def aggregation_for_measure(measure: str) -> str:
    lowered = measure.lower()
    if any(token in lowered for token in AVERAGE_TOKENS):
        return "mean"
    if any(token in lowered for token in ADDITIVE_TOKENS):
        return "sum"
    return "sum"


def _measure_score(measure: str) -> int:
    lowered = measure.lower()
    if lowered.startswith("is_") or lowered.endswith("_id") or lowered == "id":
        return -1_000
    priorities = (
        (("net_revenue", "net_sales", "net_income"), 1_000),
        (("profit", "earnings"), 950),
        (("gross_revenue", "revenue", "turnover", "sales"), 900),
        (("quantity", "volume", "orders", "units"), 750),
        (("cost", "amount"), 650),
        (("price",), 400),
        (("discount", "pct", "percent", "rate", "ratio"), 250),
    )
    return max(
        (score for tokens, score in priorities if any(token in lowered for token in tokens)),
        default=500,
    )


def ranked_measures(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[str]:
    candidates: list[str] = []
    for value in [
        *(prepared.get("primary_measures") or []),
        *(prepared.get("time_series_candidates") or []),
        *list(df.columns),
    ]:
        column = str(value)
        if column not in candidates and is_numeric_measure(df, column):
            candidates.append(column)
    positions = {value: index for index, value in enumerate(candidates)}
    return sorted(
        (value for value in candidates if _measure_score(value) >= 0),
        key=lambda value: (-_measure_score(value), positions[value]),
    )


def selected_granularity(prepared: dict[str, Any]) -> TimeGranularity:
    temporal = prepared.get("temporal_profile") or {}
    value = temporal.get("inferred_frequency") or prepared.get("time_granularity")
    return value if value in SUPPORTED_GRANULARITIES else "month"


def selected_date_column(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> str | None:
    temporal = prepared.get("temporal_profile") or {}
    value = prepared.get("date_column") or temporal.get("date_column")
    return str(value) if isinstance(value, str) and value in df else None


def select_primary_series(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> PrimarySeries | None:
    date_column = selected_date_column(prepared, df)
    measures = ranked_measures(prepared, df)
    if not date_column or not measures:
        return None
    measure = measures[0]
    return PrimarySeries(
        measure=measure,
        aggregation=aggregation_for_measure(measure),
        date_column=date_column,
        granularity=selected_granularity(prepared),
    )


def period_frequency(granularity: str) -> str:
    return {
        "day": "D",
        "week": "W-MON",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }[granularity]


def infer_time_granularity(
    values: pd.Series,
    fallback: TimeGranularity | None = None,
) -> TimeGranularity:
    """Choose a readable, sufficiently regular grain for a dated dataset.

    Transaction datasets often contain dates only when a transaction occurred.
    Treating a multi-year transaction history as a daily series can therefore
    create large artificial gaps, even though its monthly history is complete.
    Prefer the finest regular grain with a practical number of displayed points.
    """
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return (
            cast(TimeGranularity, fallback)
            if fallback is not None and fallback in SUPPORTED_GRANULARITIES
            else "month"
        )

    for granularity in TIME_GRANULARITIES:
        frequency = period_frequency(granularity)
        periods = dates.dt.to_period(frequency)
        observed = periods.nunique()
        if observed < 4:
            continue

        calendar_periods = len(
            pd.period_range(periods.min(), periods.max(), freq=frequency)
        )
        coverage = observed / calendar_periods if calendar_periods else 0.0

        # A timeline with more than roughly 18 months of daily points becomes
        # difficult to read; roll it up while preserving the time-series shape.
        if observed <= 180 and coverage >= 0.7:
            return granularity

    # Sparse histories may only become regular at a coarser grain.  Prefer the
    # most complete viable candidate rather than rejecting useful temporal data.
    candidates: list[tuple[float, int, TimeGranularity]] = []
    for granularity in TIME_GRANULARITIES:
        frequency = period_frequency(granularity)
        periods = dates.dt.to_period(frequency)
        observed = periods.nunique()
        if observed < 4:
            continue
        calendar_periods = len(
            pd.period_range(periods.min(), periods.max(), freq=frequency)
        )
        coverage = observed / calendar_periods if calendar_periods else 0.0
        candidates.append((coverage, -observed, granularity))

    if candidates:
        return max(candidates)[2]
    return (
        cast(TimeGranularity, fallback)
        if fallback is not None and fallback in SUPPORTED_GRANULARITIES
        else "month"
    )


def temporal_period_count(values: pd.Series, granularity: str) -> int:
    """Return the number of usable calendar periods at ``granularity``."""
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty or granularity not in SUPPORTED_GRANULARITIES:
        return 0
    return int(dates.dt.to_period(period_frequency(granularity)).nunique())


def is_temporal_dimension(
    column: str,
    prepared: dict[str, Any],
) -> bool:
    lowered = column.lower()
    if column == prepared.get("date_column"):
        return True
    if lowered in TEMPORAL_DIMENSION_NAMES:
        return True
    if any(
        lowered.endswith(suffix)
        for suffix in ("_date", "_day", "_month", "_quarter", "_time", "_week", "_year")
    ):
        return True
    profiles = (prepared.get("dataset_profile") or {}).get("column_profiles") or []
    profile = next(
        (item for item in profiles if isinstance(item, dict) and item.get("name") == column),
        {},
    )
    return profile.get("inferred_type") == "date"


def value_format_for_measure(measure: str, prepared: dict[str, Any]) -> str:
    lowered = measure.lower()
    if any(token in lowered for token in ("pct", "percent", "percentage", "rate")):
        return "percentage"
    if detect_currency([measure]) or any(
        token in lowered
        for token in ("amount", "cost", "gbp", "price", "profit", "revenue", "sales")
    ):
        return "currency"
    return "number"
