"""Dataset cleaning, profiling, and preparation schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeAlias, get_args

from pydantic import Field, model_validator

from app.core.config import AgentProvider
from app.schemas.common import StrictModel

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
    currency: str | None = None


class TemporalProfile(StrictModel):
    date_column: str | None = None
    unique_periods: int = Field(ge=0)
    minimum_date: str | None = None
    maximum_date: str | None = None
    inferred_frequency: Literal["day", "week", "month", "quarter", "year"] | None = None


SemanticRole: TypeAlias = Literal[
    "date",
    "transaction_id",
    "primary_measure",
    "dimension",
    "category",
    "flag",
    "text",
    "description",
    "unknown",
]
SEMANTIC_ROLE_VALUES = frozenset(get_args(SemanticRole))


class SemanticRoleAssignment(StrictModel):
    column: str
    role: SemanticRole
    reason: str | None = None

class PreparationTransformation(StrictModel):
    operation: TransformationOperation
    column: str
    reason: str | None = None
    value: str | float | bool | None = None
    analysis_types: list[str] = Field(default_factory=list)
    formula_id: str | None = None
    source_columns: list[str] = Field(default_factory=list)


class CapabilityFlags(StrictModel):
    supports_kpis: bool = False
    supports_trends: bool = False
    supports_anomalies: bool = False
    supports_forecasting: bool = False
    has_temporal_data: bool = False


def normalize_semantic_role_assignments(value: Any) -> list[dict[str, Any]]:
    """Return canonical per-column assignments from current and legacy shapes.

    The first valid assignment for a column wins.  This preserves an existing
    per-column assignment when a provider repeats that column in a later
    grouped entry.
    """
    entries: list[Any]
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        # Retain compatibility with the still older ``{role: columns}`` form.
        entries = []
        for role, raw_columns in value.items():
            if isinstance(raw_columns, dict) and isinstance(
                raw_columns.get("column"), str
            ):
                entries.append({"role": role, **raw_columns})
            else:
                entries.append(
                    {
                        "role": role,
                        "values": (
                            [raw_columns]
                            if not isinstance(raw_columns, list)
                            else raw_columns
                        ),
                    }
                )
    else:
        return []

    normalized: list[dict[str, Any]] = []
    seen_columns: set[str] = set()

    def add(column: Any, role: Any, reason: Any = None) -> None:
        if not isinstance(column, str) or not column.strip():
            return
        if not isinstance(role, str) or role not in SEMANTIC_ROLE_VALUES:
            return
        canonical_column = column.strip()
        if canonical_column in seen_columns:
            return
        assignment: dict[str, Any] = {
            "column": canonical_column,
            "role": role,
        }
        if isinstance(reason, str) and reason.strip():
            assignment["reason"] = reason.strip()
        normalized.append(assignment)
        seen_columns.add(canonical_column)

    for entry in entries:
        if isinstance(entry, SemanticRoleAssignment):
            add(entry.column, entry.role, entry.reason)
            continue
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        column = entry.get("column")
        if isinstance(column, str):
            add(column, role, entry.get("reason"))
            continue
        grouped_columns = entry.get("values")
        if not isinstance(grouped_columns, list):
            continue
        for grouped_column in grouped_columns:
            add(grouped_column, role, entry.get("reason"))

    return normalized


class PreparationPlan(StrictModel):
    semantic_roles: list[SemanticRoleAssignment] = Field(default_factory=list)
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

    @model_validator(mode="before")
    @classmethod
    def normalize_common_llm_variants(cls, value: Any) -> Any:
        """Normalize safe, known JSON variants produced by planning models."""
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        payload["semantic_roles"] = normalize_semantic_role_assignments(
            payload.get("semantic_roles")
        )

        granularity = payload.get("time_granularity")
        if isinstance(granularity, str):
            payload["time_granularity"] = {
                "daily": "day",
                "weekly": "week",
                "monthly": "month",
                "quarterly": "quarter",
                "yearly": "year",
            }.get(granularity.strip().lower(), granularity.strip().lower())
        return payload


class PreparationReport(StrictModel):
    plan_source: AgentProvider | Literal["deterministic"]
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
    currency: str | None = None
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
