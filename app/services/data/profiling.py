"""Deterministic dataset profiling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.core.currency import detect_currency
from app.schemas.data_preparation import ColumnProfile, DatasetProfile
from app.services.data.cleaning import _infer_column_type, _parse_dates_for_column

MAX_PROFILE_COLUMNS = 80
MAX_SAMPLE_LENGTH = 80

def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value

def _short_sample(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return safe[:MAX_SAMPLE_LENGTH]
    return safe

def _profile_dataset(df: pd.DataFrame, business_description: str | None) -> DatasetProfile:
    column_profiles: list[ColumnProfile] = []
    date_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in list(df.columns)[:MAX_PROFILE_COLUMNS]:
        series = df[column]
        inferred = _infer_column_type(series, str(column))
        null_count = int(series.isna().sum())
        samples = [
            _short_sample(value)
            for value in series.dropna().drop_duplicates().head(3).tolist()
        ]

        numeric_minimum = numeric_maximum = numeric_mean = numeric_median = None
        date_success = date_minimum = date_maximum = None
        if inferred == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            numeric_columns.append(str(column))
            if not values.empty:
                numeric_minimum = round(float(values.min()), 6)
                numeric_maximum = round(float(values.max()), 6)
                numeric_mean = round(float(values.mean()), 6)
                numeric_median = round(float(values.median()), 6)
        elif inferred == "date":
            parsed = _parse_dates_for_column(series, str(column))
            success = float(parsed[series.notna()].notna().mean()) if series.notna().any() else 0.0
            date_success = round(success * 100, 2)
            valid_dates = parsed.dropna()
            if not valid_dates.empty:
                date_columns.append(str(column))
                date_minimum = valid_dates.min().isoformat()
                date_maximum = valid_dates.max().isoformat()
        elif inferred == "categorical":
            categorical_columns.append(str(column))

        column_profiles.append(
            ColumnProfile(
                name=str(column),
                inferred_type=inferred,
                null_count=null_count,
                null_percentage=round(float(series.isna().mean() * 100), 2),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=samples,
                numeric_minimum=numeric_minimum,
                numeric_maximum=numeric_maximum,
                numeric_mean=numeric_mean,
                numeric_median=numeric_median,
                date_parse_success_percentage=date_success,
                date_minimum=date_minimum,
                date_maximum=date_maximum,
            )
        )

    profile = DatasetProfile(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        column_profiles=column_profiles,
        candidate_date_columns=date_columns,
        candidate_numeric_columns=numeric_columns,
        candidate_categorical_columns=categorical_columns,
        business_description=(business_description or None),
    )
    return profile.model_copy(
        update={"currency": detect_currency(item.name for item in profile.column_profiles)}
    )
