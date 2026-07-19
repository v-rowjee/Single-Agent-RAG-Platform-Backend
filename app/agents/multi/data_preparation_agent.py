from __future__ import annotations

# 1. Imports and constants

import logging
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.services.series import (
    infer_time_granularity,
    temporal_period_count,
)
from app.core.config import AgentProvider, agent_model_policy
from app.core.llm import provider_display_name, request_structured
from app.core.prompts import render_agent_prompts


SUPPORTED_OPERATIONS = {
    "fill_constant",
    "preserve_missing",
    "exclude_from_measure_analysis",
    "exclude_from_temporal_analysis",
    "drop_rows_with_missing",
    "reconstruct_from_formula",
}
SUPPORTED_FORMULAS = {
    "quantity_times_unit_price",
    "gross_revenue_minus_discount",
}
FORMULA_SOURCE_COLUMNS = {
    "quantity_times_unit_price": ("quantity", "unit_price"),
    "gross_revenue_minus_discount": ("gross_revenue", "discount"),
}
MISSING_MARKERS = {"", " ", "na", "n/a", "null", "none", "missing", "-"}
NUMERIC_CONVERSION_THRESHOLD = 0.9
DATE_CONVERSION_THRESHOLD = 0.75
DATE_CANDIDATE_THRESHOLD = 0.6
MIN_FORECAST_PERIODS = 4
MIN_TREND_PERIODS = 2
MIN_ANOMALY_OBSERVATIONS = 8
MAX_PROFILE_COLUMNS = 80
MAX_SAMPLE_LENGTH = 80

logger = logging.getLogger(__name__)


# 2. Exceptions


class DataPreparationError(RuntimeError):
    pass


# 3. Enums and Pydantic models


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransformationOperation(str, Enum):
    fill_constant = "fill_constant"
    preserve_missing = "preserve_missing"
    exclude_from_measure_analysis = "exclude_from_measure_analysis"
    exclude_from_temporal_analysis = "exclude_from_temporal_analysis"
    drop_rows_with_missing = "drop_rows_with_missing"
    reconstruct_from_formula = "reconstruct_from_formula"


class MissingValueSummary(StrictModel):
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class GenericCleaningResult(StrictModel):
    cleaned_file_path: str
    original_row_count: int = Field(ge=0)
    cleaned_row_count: int = Field(ge=0)
    original_column_count: int = Field(ge=0)
    cleaned_column_count: int = Field(ge=0)
    duplicate_rows_removed: int = Field(ge=0)
    empty_rows_removed: int = Field(ge=0)
    empty_columns_removed: list[str] = Field(default_factory=list)
    missing_value_summary: dict[str, MissingValueSummary] = Field(default_factory=dict)
    inferred_column_types: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ColumnProfile(StrictModel):
    name: str
    inferred_type: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0, le=100)
    unique_count: int = Field(ge=0)
    sample_values: list[Any] = Field(default_factory=list, max_length=3)
    numeric_minimum: float | None = None
    numeric_maximum: float | None = None
    numeric_mean: float | None = None
    numeric_median: float | None = None
    date_parse_success_percentage: float | None = Field(default=None, ge=0, le=100)
    date_minimum: str | None = None
    date_maximum: str | None = None


class DatasetProfile(StrictModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    candidate_date_columns: list[str] = Field(default_factory=list)
    candidate_numeric_columns: list[str] = Field(default_factory=list)
    candidate_categorical_columns: list[str] = Field(default_factory=list)
    business_description: str | None = None


class TemporalProfile(StrictModel):
    date_column: str | None = None
    unique_periods: int = Field(ge=0)
    minimum_date: str | None = None
    maximum_date: str | None = None
    inferred_frequency: Literal["day", "week", "month", "quarter", "year"] | None = None


class ColumnSemanticRole(StrictModel):
    column: str
    role: Literal[
        "date",
        "transaction_id",
        "primary_measure",
        "dimension",
        "category",
        "description",
        "unknown",
    ]
    reason: str | None = None


class PreparationTransformation(StrictModel):
    operation: TransformationOperation
    column: str
    reason: str
    value: str | int | float | bool | None = None
    analysis_types: list[str] = Field(default_factory=list)
    formula_id: str | None = None
    source_columns: list[str] = Field(default_factory=list)


class CapabilityFlags(StrictModel):
    supports_kpis: bool = False
    supports_trends: bool = False
    supports_anomalies: bool = False
    supports_forecasting: bool = False
    has_temporal_data: bool = False


class PreparationPlan(StrictModel):
    semantic_roles: list[ColumnSemanticRole] = Field(default_factory=list)
    date_column: str | None = None
    transaction_id_columns: list[str] = Field(default_factory=list)
    primary_measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    currency: str | None = None
    time_granularity: Literal["day", "week", "month", "quarter", "year"] | None = None
    time_series_candidates: list[str] = Field(default_factory=list)
    transformations: list[PreparationTransformation] = Field(default_factory=list)
    capability_flags: CapabilityFlags = Field(default_factory=CapabilityFlags)
    limitations: list[str] = Field(default_factory=list)


class PreparationReport(StrictModel):
    plan_source: AgentProvider | Literal["fallback"]
    executed_transformations: list[str] = Field(default_factory=list)
    rejected_transformations: list[str] = Field(default_factory=list)
    excluded_from_measure_analysis: dict[str, int] = Field(default_factory=dict)
    excluded_from_temporal_analysis_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


class PreparedDatasetPackage(StrictModel):
    prepared_file_path: str
    file_name: str
    temporal_dataset_path: str | None
    dataset_profile: DatasetProfile
    semantic_column_map: dict[str, str] = Field(default_factory=dict)
    date_column: str | None
    primary_measures: list[str] = Field(default_factory=list)
    dimension_candidates: list[str] = Field(default_factory=list)
    time_series_candidates: list[str] = Field(default_factory=list)
    capability_flags: CapabilityFlags
    temporal_profile: TemporalProfile
    cleaning_report: GenericCleaningResult
    preparation_report: PreparationReport
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# 4. Generic cleaning functions


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


# 5. Dataset profiling functions


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

    return DatasetProfile(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        column_profiles=column_profiles,
        candidate_date_columns=date_columns,
        candidate_numeric_columns=numeric_columns,
        candidate_categorical_columns=categorical_columns,
        business_description=(business_description or None),
    )


# 6. Deterministic fallback planning


def _contains_any(column: str, tokens: tuple[str, ...]) -> bool:
    return any(token in column.lower() for token in tokens)


def _fallback_plan(profile: DatasetProfile, warning: str | None = None) -> PreparationPlan:
    columns = {item.name: item for item in profile.column_profiles}
    numeric = list(profile.candidate_numeric_columns)
    date_column = profile.candidate_date_columns[0] if profile.candidate_date_columns else None
    transaction_ids = [
        name
        for name in columns
        if _contains_any(name, ("transaction_id", "order_id", "invoice_id", "receipt_id"))
    ][:3]
    primary_measures = [
        name
        for name in numeric
        if _contains_any(name, ("revenue", "sales", "amount", "profit", "cost", "value", "quantity", "price"))
    ][:5] or numeric[:3]
    dimensions = [
        item.name
        for item in profile.column_profiles
        if item.name not in primary_measures and item.name != date_column and item.inferred_type in {"categorical", "text"}
    ][:8]

    transformations: list[PreparationTransformation] = []
    for name, item in columns.items():
        if item.null_count <= 0:
            continue
        lowered = name.lower()
        if lowered == "product_name" or ("product" in lowered and "name" in lowered):
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.fill_constant,
                    column=name,
                    value="Unknown Product",
                    reason="Missing product labels are safe to group as unknown.",
                )
            )
        elif "sales_channel" in lowered or "channel" == lowered:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.fill_constant,
                    column=name,
                    value="Unknown",
                    reason="Missing sales channel labels are safe to group as unknown.",
                )
            )
        elif "description" in lowered or "notes" in lowered or "comment" in lowered:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.preserve_missing,
                    column=name,
                    reason="Optional descriptive text should preserve missing values.",
                )
            )
        elif name == date_column:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.exclude_from_temporal_analysis,
                    column=name,
                    analysis_types=["trend", "forecasting"],
                    reason="Rows without valid dates should not be used for temporal analysis.",
                )
            )
        elif name in primary_measures:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.exclude_from_measure_analysis,
                    column=name,
                    analysis_types=["kpi", "trend", "forecasting", "anomaly"],
                    reason="Missing primary measures should not be invented.",
                )
            )
    limitations = []
    if warning:
        limitations.append(warning)

    return PreparationPlan(
        semantic_roles=[
            *([ColumnSemanticRole(column=date_column, role="date", reason="Best date candidate.")] if date_column else []),
            *[
                ColumnSemanticRole(column=column, role="transaction_id", reason="Identifier-like column name.")
                for column in transaction_ids
            ],
            *[
                ColumnSemanticRole(column=column, role="primary_measure", reason="Numeric business measure candidate.")
                for column in primary_measures
            ],
        ],
        date_column=date_column,
        transaction_id_columns=transaction_ids,
        primary_measures=primary_measures,
        dimensions=dimensions,
        categorical_columns=list(profile.candidate_categorical_columns)[:10],
        currency=_detect_currency(profile),
        time_granularity=_guess_time_granularity(profile, date_column),
        time_series_candidates=primary_measures[:3] if date_column else [],
        transformations=transformations,
        capability_flags=CapabilityFlags(
            supports_kpis=bool(primary_measures),
            supports_trends=bool(date_column and primary_measures),
            supports_forecasting=bool(date_column and primary_measures),
            supports_anomalies=bool(primary_measures),
            has_temporal_data=bool(date_column),
        ),
        limitations=limitations,
    )


def _detect_currency(profile: DatasetProfile) -> str | None:
    haystack = " ".join(item.name.lower() for item in profile.column_profiles)
    if "gbp" in haystack or "pound" in haystack:
        return "GBP"
    if "eur" in haystack or "euro" in haystack:
        return "EUR"
    if "usd" in haystack or "dollar" in haystack:
        return "USD"
    return None


def _guess_time_granularity(profile: DatasetProfile, date_column: str | None) -> Literal["day", "week", "month", "quarter", "year"] | None:
    if not date_column:
        return None
    column = date_column.lower()
    if "year" in column:
        return "year"
    if "quarter" in column:
        return "quarter"
    if "month" in column:
        return "month"
    return "day"


# 7. LLM preparation planning


def _compact_profile_payload(profile: DatasetProfile) -> dict[str, Any]:
    return profile.model_dump(mode="json", exclude_none=True)


async def _request_plan(profile: DatasetProfile) -> PreparationPlan:
    prompts = render_agent_prompts(
        "multi/data_preparation",
        supported_operations=sorted(SUPPORTED_OPERATIONS),
        supported_formulas=sorted(SUPPORTED_FORMULAS),
        profile=_compact_profile_payload(profile),
    )
    return await request_structured(
        policy=agent_model_policy("data_preparation"),
        response_model=PreparationPlan,
        schema_name="data_preparation_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


async def _plan_with_provider_or_fallback(
    profile: DatasetProfile,
) -> tuple[PreparationPlan, AgentProvider | Literal["fallback"], list[str]]:
    warnings: list[str] = []
    policy = agent_model_policy("data_preparation")
    try:
        return await _request_plan(profile), policy.provider, warnings
    except Exception as error:
        warnings.append(
            f"{provider_display_name(policy.provider)} preparation planning failed: "
            f"{error}"
        )
        warnings.append("Deterministic fallback preparation plan was used.")
        return _fallback_plan(
            profile,
            "LLM planning failed; deterministic fallback was used.",
        ), "fallback", warnings


# 8. Plan validation


def _profile_map(profile: DatasetProfile) -> dict[str, ColumnProfile]:
    return {item.name: item for item in profile.column_profiles}


def _is_primary_numeric_measure(column: str, plan: PreparationPlan, profile: DatasetProfile) -> bool:
    profile_item = _profile_map(profile).get(column)
    return column in plan.primary_measures and bool(profile_item and profile_item.inferred_type == "numeric")


def _validate_plan(plan: PreparationPlan, profile: DatasetProfile) -> tuple[PreparationPlan, list[str], list[str]]:
    columns = set(_profile_map(profile))
    warnings: list[str] = []
    rejected: list[str] = []

    def known(column: str | None) -> bool:
        return column is None or column in columns

    if not known(plan.date_column):
        warnings.append(f"Rejected unknown date column: {plan.date_column}")
        plan.date_column = None

    if plan.date_column and plan.date_column not in profile.candidate_date_columns:
        warnings.append(f"Date column `{plan.date_column}` was not a plausible date candidate.")
        plan.date_column = None

    plan.transaction_id_columns = [column for column in plan.transaction_id_columns if known(column)]
    plan.primary_measures = [
        column for column in plan.primary_measures if column in profile.candidate_numeric_columns
    ]
    plan.dimensions = [column for column in plan.dimensions if known(column)]
    plan.categorical_columns = [column for column in plan.categorical_columns if known(column)]
    plan.time_series_candidates = [
        column for column in plan.time_series_candidates if column in plan.primary_measures
    ]
    plan.semantic_roles = [role for role in plan.semantic_roles if known(role.column)]

    seen: set[tuple[str, str]] = set()
    column_ops: dict[str, set[TransformationOperation]] = {}
    valid_transformations: list[PreparationTransformation] = []
    for transformation in plan.transformations:
        reason = f"{transformation.operation.value} on `{transformation.column}`"
        if transformation.column not in columns:
            rejected.append(f"{reason}: unknown column")
            continue
        duplicate_key = (transformation.operation.value, transformation.column)
        if duplicate_key in seen:
            rejected.append(f"{reason}: duplicate transformation")
            continue
        seen.add(duplicate_key)

        existing_ops = column_ops.setdefault(transformation.column, set())
        if (
            transformation.operation == TransformationOperation.fill_constant
            and TransformationOperation.preserve_missing in existing_ops
        ) or (
            transformation.operation == TransformationOperation.preserve_missing
            and TransformationOperation.fill_constant in existing_ops
        ):
            rejected.append(f"{reason}: contradictory missing-value operation")
            continue

        if transformation.operation == TransformationOperation.fill_constant:
            if transformation.value is None:
                rejected.append(f"{reason}: fill_constant requires a value")
                continue
            if transformation.column in plan.transaction_id_columns:
                rejected.append(f"{reason}: identifiers cannot be constant-filled")
                continue
            if _is_primary_numeric_measure(transformation.column, plan, profile):
                rejected.append(f"{reason}: primary numeric measures cannot be constant-filled")
                continue
            if isinstance(transformation.value, str) and not transformation.value.strip():
                rejected.append(f"{reason}: fill value cannot be blank")
                continue

        if transformation.operation == TransformationOperation.reconstruct_from_formula:
            if transformation.formula_id not in SUPPORTED_FORMULAS:
                rejected.append(f"{reason}: unsupported formula ID")
                continue
            required = FORMULA_SOURCE_COLUMNS.get(transformation.formula_id or "", ())
            sources = transformation.source_columns or list(required)
            if not all(source in columns for source in sources):
                rejected.append(f"{reason}: unknown reconstruction source column")
                continue
            if transformation.formula_id in FORMULA_SOURCE_COLUMNS and set(sources) != set(required):
                rejected.append(f"{reason}: source columns do not match registered formula")
                continue
            if transformation.column not in columns:
                rejected.append(f"{reason}: target column missing")
                continue
            source_profiles = _profile_map(profile)
            if any(source_profiles[source].inferred_type != "numeric" for source in sources):
                rejected.append(f"{reason}: reconstruction source columns must be numeric")
                continue

        existing_ops.add(transformation.operation)
        valid_transformations.append(transformation)

    plan.transformations = valid_transformations
    plan.capability_flags = _downgrade_capabilities(plan, profile, warnings)
    return plan, warnings, rejected


def _downgrade_capabilities(plan: PreparationPlan, profile: DatasetProfile, warnings: list[str]) -> CapabilityFlags:
    flags = plan.capability_flags
    has_measure = bool(plan.primary_measures)
    has_date = bool(plan.date_column)
    usable_periods = _usable_period_count(profile, plan.date_column)
    rows = profile.row_count

    if flags.supports_kpis and not has_measure:
        warnings.append("KPI analysis disabled because no usable numeric measure exists.")
        flags.supports_kpis = False
    if flags.supports_trends and not (has_date and has_measure and usable_periods >= MIN_TREND_PERIODS):
        warnings.append("Trend analysis disabled because date or numeric measure coverage is insufficient.")
        flags.supports_trends = False
    if flags.supports_forecasting and not (has_date and has_measure and usable_periods >= MIN_FORECAST_PERIODS):
        warnings.append("Forecasting disabled because too few usable time periods exist.")
        flags.supports_forecasting = False
    if flags.supports_anomalies and not (has_measure and rows >= MIN_ANOMALY_OBSERVATIONS):
        warnings.append("Anomaly analysis disabled because there are insufficient observations.")
        flags.supports_anomalies = False
    flags.has_temporal_data = bool(has_date and usable_periods >= MIN_TREND_PERIODS)
    return flags


def _usable_period_count(profile: DatasetProfile, date_column: str | None) -> int:
    if not date_column:
        return 0
    item = _profile_map(profile).get(date_column)
    return int(item.unique_count) if item else 0


def _temporal_profile(
    df: pd.DataFrame,
    date_column: str | None,
    granularity: Literal["day", "week", "month", "quarter", "year"] | None,
) -> TemporalProfile:
    if not date_column or date_column not in df:
        return TemporalProfile(unique_periods=0)
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    return TemporalProfile(
        date_column=date_column,
        unique_periods=temporal_period_count(
            dates,
            granularity or "month",
        ),
        minimum_date=dates.min().date().isoformat() if not dates.empty else None,
        maximum_date=dates.max().date().isoformat() if not dates.empty else None,
        inferred_frequency=granularity,
    )


def _reconcile_temporal_capabilities(
    plan: PreparationPlan,
    prepared: pd.DataFrame,
) -> str | None:
    """Re-evaluate temporal capability after dates have been cleaned.

    Capability flags from an LLM plan are only a proposal.  The cleaned data is
    the authority for its usable periods and prevents a sparse daily transaction
    series from incorrectly suppressing an otherwise valid forecast.
    """
    date_column = plan.date_column
    if not date_column or date_column not in prepared:
        return None

    granularity = infer_time_granularity(
        prepared[date_column],
        plan.time_granularity,
    )
    plan.time_granularity = granularity
    period_count = temporal_period_count(prepared[date_column], granularity)
    measures = [
        column
        for column in plan.primary_measures
        if column in prepared
        and pd.api.types.is_numeric_dtype(prepared[column])
        and prepared[column].notna().any()
    ]
    has_measure = bool(measures)
    has_temporal_data = period_count >= MIN_TREND_PERIODS

    plan.capability_flags.has_temporal_data = has_temporal_data
    plan.capability_flags.supports_kpis = has_measure
    plan.capability_flags.supports_trends = has_measure and has_temporal_data
    plan.capability_flags.supports_forecasting = (
        has_measure and period_count >= MIN_FORECAST_PERIODS
    )
    plan.capability_flags.supports_anomalies = (
        has_measure and len(prepared) >= MIN_ANOMALY_OBSERVATIONS
    )
    if not plan.time_series_candidates and measures:
        plan.time_series_candidates = measures[:3]
    return granularity


# 9. Pandas plan execution


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    output.to_csv(path, index=False)


def _execute_plan(
    df: pd.DataFrame,
    plan: PreparationPlan,
    output_dir: Path,
    plan_source: AgentProvider | Literal["fallback"],
    validation_warnings: list[str],
    rejected_transformations: list[str],
) -> tuple[pd.DataFrame, str | None, PreparationReport]:
    prepared = df.copy()
    temporal_mask = pd.Series(True, index=prepared.index)
    executed: list[str] = []
    warnings = list(validation_warnings)
    measure_exclusions: dict[str, int] = {}

    for transformation in plan.transformations:
        column = transformation.column
        operation = transformation.operation

        if operation == TransformationOperation.fill_constant:
            missing = prepared[column].isna()
            prepared.loc[missing, column] = transformation.value
            executed.append(f"Filled {int(missing.sum())} missing `{column}` values with a constant.")

        elif operation == TransformationOperation.preserve_missing:
            executed.append(f"Preserved missing values in `{column}`.")

        elif operation == TransformationOperation.exclude_from_measure_analysis:
            count = int(prepared[column].isna().sum())
            measure_exclusions[column] = measure_exclusions.get(column, 0) + count
            executed.append(f"Marked {count} rows as excluded from measure analysis for `{column}`.")

        elif operation == TransformationOperation.exclude_from_temporal_analysis:
            invalid = prepared[column].isna()
            if plan.date_column == column:
                parsed = _parse_dates_for_column(prepared[column], column)
                invalid = parsed.isna()
            temporal_mask &= ~invalid
            executed.append(f"Excluded {int(invalid.sum())} rows from temporal analysis using `{column}`.")

        elif operation == TransformationOperation.drop_rows_with_missing:
            before = len(prepared)
            prepared = prepared[prepared[column].notna()].copy()
            temporal_mask = temporal_mask.reindex(prepared.index, fill_value=False)
            executed.append(f"Dropped {before - len(prepared)} rows with missing `{column}`.")

        elif operation == TransformationOperation.reconstruct_from_formula:
            count, formula_warnings = _apply_formula(prepared, transformation)
            warnings.extend(formula_warnings)
            executed.append(f"Reconstructed {count} missing `{column}` values with `{transformation.formula_id}`.")

    prepared_path = output_dir / "prepared_dataset.csv"
    try:
        _save_csv(prepared, prepared_path)
    except Exception as exc:
        raise DataPreparationError(f"Prepared dataset could not be saved: {exc}") from exc

    temporal_path: str | None = None
    temporal_excluded = 0
    if plan.date_column and plan.date_column in prepared.columns:
        parsed_dates = _parse_dates_for_column(prepared[plan.date_column], plan.date_column)
        temporal_mask = temporal_mask.reindex(prepared.index, fill_value=False) & parsed_dates.notna()
        temporal = prepared.loc[temporal_mask].copy()
        temporal_excluded = int(len(prepared) - len(temporal))
        if not temporal.empty:
            temporal[plan.date_column] = parsed_dates.loc[temporal.index]
            temporal_file = output_dir / "prepared_temporal_dataset.csv"
            try:
                _save_csv(temporal, temporal_file)
                temporal_path = str(temporal_file)
            except Exception as exc:
                warnings.append(f"Temporal dataset could not be saved: {exc}")

    report = PreparationReport(
        plan_source=plan_source,
        executed_transformations=executed,
        rejected_transformations=rejected_transformations,
        excluded_from_measure_analysis=measure_exclusions,
        excluded_from_temporal_analysis_rows=temporal_excluded,
        warnings=warnings,
    )
    return prepared, temporal_path, report


def _apply_formula(df: pd.DataFrame, transformation: PreparationTransformation) -> tuple[int, list[str]]:
    warnings: list[str] = []
    target = transformation.column
    formula_id = transformation.formula_id
    sources = transformation.source_columns or list(FORMULA_SOURCE_COLUMNS.get(formula_id or "", ()))
    missing_target = df[target].isna()
    if not missing_target.any():
        return 0, warnings

    source_values = {source: pd.to_numeric(df[source], errors="coerce") for source in sources}
    if formula_id == "quantity_times_unit_price":
        values = source_values["quantity"] * source_values["unit_price"]
    elif formula_id == "gross_revenue_minus_discount":
        values = source_values["gross_revenue"] - source_values["discount"]
    else:
        warnings.append(f"Unsupported formula skipped: {formula_id}")
        return 0, warnings

    valid = missing_target & values.notna() & np.isfinite(values)
    df.loc[valid, target] = values.loc[valid]
    invalid_count = int(missing_target.sum() - valid.sum())
    if invalid_count:
        warnings.append(f"{invalid_count} `{target}` values could not be reconstructed and remain missing.")
    return int(valid.sum()), warnings


# 10. DataPreparationAgent class


class DataPreparationAgent:
    async def run(
        self,
        uploaded_file_path: str,
        session_id: str,
        business_description: str | None = None,
        generic_cleaning_report: GenericCleaningResult | None = None,
        file_name: str | None = None,
        output_dir: Path | None = None,
    ) -> PreparedDatasetPackage:
        logger.info(
            "Data preparation started session_id=%s source_path=%s",
            session_id,
            uploaded_file_path,
        )
        if output_dir is None:
            raise DataPreparationError(
                "A temporary processing workspace is required."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if generic_cleaning_report is None:
            df, cleaning_report = _generic_clean_csv(uploaded_file_path, output_dir)
        else:
            try:
                df = pd.read_csv(uploaded_file_path, low_memory=False)
            except Exception as exc:
                raise DataPreparationError(
                    f"Generic cleaned CSV could not be read: {exc}"
                ) from exc
            cleaning_report = generic_cleaning_report
        logger.info(
            "Generic cleaning completed session_id=%s original_shape=(%s,%s) cleaned_shape=(%s,%s) output=%s",
            session_id,
            cleaning_report.original_row_count,
            cleaning_report.original_column_count,
            cleaning_report.cleaned_row_count,
            cleaning_report.cleaned_column_count,
            cleaning_report.cleaned_file_path,
        )

        profile = _profile_dataset(df, business_description)
        raw_plan, plan_source, planning_warnings = (
            await _plan_with_provider_or_fallback(profile)
        )

        try:
            plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
        except Exception as exc:
            planning_warnings.append(f"Plan validation failed; fallback plan used: {exc}")
            raw_plan = _fallback_plan(profile, "Plan validation failed; deterministic fallback was used.")
            plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
            plan_source = "fallback"

        if plan_source == "fallback":
            logger.info("Data preparation using fallback plan session_id=%s", session_id)
        else:
            logger.info(
                "Data preparation using %s plan session_id=%s",
                provider_display_name(plan_source),
                session_id,
            )

        prepared, temporal_path, preparation_report = _execute_plan(
            df=df,
            plan=plan,
            output_dir=output_dir,
            plan_source=plan_source,
            validation_warnings=[*planning_warnings, *validation_warnings],
            rejected_transformations=rejected,
        )
        prepared_path = str(output_dir / "prepared_dataset.csv")
        prepared_profile = _profile_dataset(prepared, business_description)
        effective_granularity = _reconcile_temporal_capabilities(plan, prepared)

        semantic_map = {role.column: role.role for role in plan.semantic_roles}
        warnings = _dedupe(
            [
                *cleaning_report.warnings,
                *planning_warnings,
                *validation_warnings,
                *preparation_report.warnings,
                *rejected,
            ]
        )
        package = PreparedDatasetPackage(
            prepared_file_path=prepared_path,
            file_name=str(file_name or Path(uploaded_file_path).name),
            temporal_dataset_path=temporal_path,
            dataset_profile=prepared_profile,
            semantic_column_map=semantic_map,
            date_column=plan.date_column,
            primary_measures=plan.primary_measures,
            dimension_candidates=plan.dimensions,
            time_series_candidates=plan.time_series_candidates,
            capability_flags=plan.capability_flags,
            temporal_profile=_temporal_profile(
                prepared,
                plan.date_column,
                effective_granularity,
            ),
            cleaning_report=cleaning_report,
            preparation_report=preparation_report,
            limitations=plan.limitations,
            warnings=warnings,
        )
        logger.info(
            "Data preparation completed session_id=%s prepared_path=%s temporal_path=%s capabilities=%s",
            session_id,
            package.prepared_file_path,
            package.temporal_dataset_path,
            package.capability_flags.model_dump(mode="json"),
        )
        return package


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


# 11. Shared agent instance


data_preparation_agent = DataPreparationAgent()


# 12. LangGraph node function


async def data_preparation_node(state: dict[str, Any]) -> dict[str, Any]:
    session_id = str(state.get("session_id") or state.get("sessionId") or "").strip()
    uploaded_file_path = str(
        state.get("uploaded_file_path")
        or state.get("upload_path")
        or state.get("filePath")
        or ""
    ).strip()
    business_description = state.get("business_description") or state.get("businessDescription")
    file_name = state.get("file_name") or state.get("fileName")
    working_directory = str(state.get("working_directory") or "").strip()

    if not session_id:
        raise DataPreparationError("state.session_id is required.")
    if not uploaded_file_path:
        raise DataPreparationError("state.uploaded_file_path is required.")
    if not working_directory:
        raise DataPreparationError("state.working_directory is required.")

    result = await data_preparation_agent.run(
        uploaded_file_path=uploaded_file_path,
        session_id=session_id,
        business_description=str(business_description) if business_description else None,
        generic_cleaning_report=(
            GenericCleaningResult.model_validate(state["generic_cleaning_report"])
            if state.get("generic_cleaned_file_path")
            and isinstance(state.get("generic_cleaning_report"), dict)
            else None
        ),
        file_name=str(file_name) if file_name else None,
        output_dir=Path(working_directory),
    )

    return {
        "generic_cleaned_file_path": result.cleaning_report.cleaned_file_path,
        "prepared_file_path": result.prepared_file_path,
        "prepared_temporal_file_path": result.temporal_dataset_path,
        "generic_cleaning_report": result.cleaning_report.model_dump(mode="json"),
        "prepared_dataset": result.model_dump(mode="json"),
        "warnings": result.warnings,
        "completed_agents": ["data_preparation"],
    }
