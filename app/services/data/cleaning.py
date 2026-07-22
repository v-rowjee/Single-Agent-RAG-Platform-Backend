"""Framework-neutral dataset normalization and file cleaning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import DataPreparationError
from app.schemas.data_preparation import GenericCleaningResult, MissingValueSummary

MISSING_MARKERS = {"", " ", "na", "n/a", "null", "none", "missing", "-"}
NUMERIC_CONVERSION_THRESHOLD = 0.9
DATE_CONVERSION_THRESHOLD = 0.75
DATE_CANDIDATE_THRESHOLD = 0.6


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _normalise_column_name(value: Any) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return name or "unnamed"

def _normalise_columns(columns: pd.Index) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        base = _normalise_column_name(column)
        count = counts.get(base, 0)
        output.append(base if count == 0 else f"{base}_{count + 1}")
        counts[base] = count + 1
    return output

def _replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if not pd.api.types.is_object_dtype(result[column]) and not pd.api.types.is_string_dtype(result[column]):
            continue
        text = result[column].astype("string").str.strip()
        missing = text.str.casefold().isin(MISSING_MARKERS)
        result[column] = text.mask(missing, pd.NA)
    return result

def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            continue
        name = str(column).lower()
        if any(token in name for token in ("id", "code", "phone", "postcode", "zip")):
            continue
        text = result[column].astype("string").str.strip()
        cleaned = text.str.replace(r"[$£€¥,%]", "", regex=True).str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        non_null = text.notna() & (text != "")
        ratio = float(numeric[non_null].notna().mean()) if non_null.any() else 0.0
        if ratio >= NUMERIC_CONVERSION_THRESHOLD:
            result[column] = numeric
    return result

def _parse_dates_for_column(series: pd.Series, column: str) -> pd.Series:
    name = column.lower()
    if name == "year" or name.endswith("_year"):
        years = pd.to_numeric(series, errors="coerce")
        years = years.where(years.between(1900, 2200))
        return pd.to_datetime(years.astype("Int64").astype("string"), format="%Y", errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def _is_date_candidate_name(column: str) -> bool:
    """Exclude calendar helper dimensions from destructive date coercion."""
    name = column.lower()
    helper_names = {
        "day",
        "day_name",
        "day_of_week",
        "month",
        "month_name",
        "quarter",
        "week",
        "year",
    }
    if name in helper_names or any(
        name.endswith(suffix)
        for suffix in (
            "_day",
            "_day_name",
            "_month",
            "_month_name",
            "_quarter",
            "_week",
            "_year",
        )
    ):
        return False
    return (
        name in {"date", "datetime", "time", "timestamp"}
        or any(
            token in name
            for token in ("_date", "date_", "_datetime", "_timestamp", "_time")
        )
        or name.endswith("_period")
    )

def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if not _is_date_candidate_name(str(column)):
            continue
        parsed = _parse_dates_for_column(result[column], str(column))
        non_null = result[column].notna()
        ratio = float(parsed[non_null].notna().mean()) if non_null.any() else 0.0
        if ratio >= DATE_CONVERSION_THRESHOLD and parsed.notna().any():
            result[column] = parsed
    return result

def _infer_column_type(series: pd.Series, column: str) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_date_candidate_name(column):
        parsed = _parse_dates_for_column(series, column)
        non_null = series.notna()
        ratio = float(parsed[non_null].notna().mean()) if non_null.any() else 0.0
        if ratio >= DATE_CANDIDATE_THRESHOLD:
            return "date"
    unique = series.nunique(dropna=True)
    if len(series) and unique <= min(50, max(20, int(len(series) * 0.2))):
        return "categorical"
    return "text"

def _generic_clean_csv(uploaded_file_path: str, output_dir: Path) -> tuple[pd.DataFrame, GenericCleaningResult]:
    """Clean a supported uploaded dataset and normalise it to CSV.

    The historical function name is retained because it is an internal import
    used by the orchestration layer.  Multi-agent uploads may arrive as CSV or
    XLSX, while every downstream preparation/specialist node consumes the
    cleaned CSV emitted here.
    """
    path = Path(uploaded_file_path)
    if not path.is_file():
        raise DataPreparationError(f"Uploaded file was not found: {uploaded_file_path}")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise DataPreparationError(
            "The data preparation agent accepts CSV and XLSX files only."
        )

    try:
        original = (
            pd.read_csv(path, low_memory=False)
            if suffix == ".csv"
            else pd.read_excel(path)
        )
    except Exception as exc:
        raise DataPreparationError(
            f"{suffix.removeprefix('.').upper()} could not be read: {exc}"
        ) from exc

    original_rows, original_columns = original.shape
    warnings: list[str] = []
    errors: list[str] = []

    df = original.copy()
    df.columns = _normalise_columns(df.columns)
    df = _replace_missing_markers(df)

    before_empty_rows = len(df)
    df = df.dropna(how="all")
    empty_rows_removed = before_empty_rows - len(df)

    empty_columns = [str(column) for column in df.columns if df[column].isna().all()]
    if empty_columns:
        df = df.drop(columns=empty_columns)

    before_duplicates = len(df)
    df = df.drop_duplicates()
    duplicate_rows_removed = before_duplicates - len(df)

    df = _convert_numeric(df)
    df = _convert_dates(df)

    if df.empty or len(df.columns) == 0:
        raise DataPreparationError("Generic cleaning produced no usable rows or columns.")

    missing_summary = {
        str(column): MissingValueSummary(
            count=int(df[column].isna().sum()),
            percentage=round(float(df[column].isna().mean() * 100), 2),
        )
        for column in df.columns
    }
    inferred_types = {
        str(column): _infer_column_type(df[column], str(column)) for column in df.columns
    }

    cleaned_path = output_dir / "generic_cleaned_dataset.csv"
    try:
        _save_csv(df, cleaned_path)
    except Exception as exc:
        raise DataPreparationError(f"Generic cleaned dataset could not be saved: {exc}") from exc

    report = GenericCleaningResult(
        cleaned_file_path=str(cleaned_path),
        original_row_count=int(original_rows),
        cleaned_row_count=int(len(df)),
        original_column_count=int(original_columns),
        cleaned_column_count=int(len(df.columns)),
        duplicate_rows_removed=int(duplicate_rows_removed),
        empty_rows_removed=int(empty_rows_removed),
        empty_columns_removed=empty_columns,
        missing_value_summary=missing_summary,
        inferred_column_types=inferred_types,
        warnings=warnings,
        errors=errors,
    )
    return df, report
