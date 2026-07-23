"""Deterministic preparation planning, validation, and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.core.config import AgentProvider
from app.core.currency import detect_currency
from app.core.exceptions import DataPreparationError
from app.schemas.data_preparation import (
    CapabilityFlags,
    ColumnProfile,
    DatasetProfile,
    PreparationPlan,
    PreparationReport,
    PreparationTransformation,
    SemanticRoleAssignment,
    TemporalProfile,
    TransformationOperation,
)
from app.services.data.cleaning import (
    DATE_CONVERSION_THRESHOLD,
    _parse_dates_for_column,
)
from app.services.data.series import (
    TimeGranularity,
    infer_time_granularity,
    temporal_period_count,
)

SUPPORTED_FORMULAS = {"quantity_times_unit_price", "gross_revenue_minus_discount"}
FORMULA_SOURCE_COLUMNS = {"quantity_times_unit_price": ("quantity", "unit_price"), "gross_revenue_minus_discount": ("gross_revenue", "discount")}
MIN_FORECAST_PERIODS = 4
MIN_TREND_PERIODS = 2
MIN_ANOMALY_OBSERVATIONS = 8

def _contains_any(column: str, tokens: tuple[str, ...]) -> bool:
    return any(token in column.lower() for token in tokens)

def _semantic_role_for_column(
    item: ColumnProfile,
    profile: DatasetProfile,
) -> SemanticRoleAssignment:
    """Infer one stable semantic role from bounded profile metadata."""
    name = item.name
    lowered = name.casefold()
    non_null_count = max(0, profile.row_count - item.null_count)
    unique_ratio = item.unique_count / non_null_count if non_null_count else 0.0
    mostly_unique = non_null_count > 0 and unique_ratio >= 0.8

    explicit_identifier = (
        lowered == "id"
        or lowered.endswith("_id")
        or _contains_any(
            lowered,
            ("transaction_id", "order_id", "invoice_id", "receipt_id"),
        )
    )
    if explicit_identifier and mostly_unique:
        return SemanticRoleAssignment(
            column=name,
            role="transaction_id",
            reason="Identifier-like name with a mostly unique value distribution.",
        )

    if (
        item.inferred_type == "date"
        or (item.date_parse_success_percentage or 0) >= DATE_CONVERSION_THRESHOLD * 100
    ):
        return SemanticRoleAssignment(
            column=name,
            role="date",
            reason="Values are consistently parseable as dates.",
        )

    boolean_name = (
        lowered.startswith(("is_", "has_", "can_"))
        or lowered.endswith(("_flag", "_active", "_enabled"))
    )
    if item.inferred_type == "boolean" or (boolean_name and item.unique_count <= 2):
        return SemanticRoleAssignment(
            column=name,
            role="flag",
            reason="Boolean or binary indicator column.",
        )

    if item.inferred_type == "numeric":
        if lowered in {"year", "month", "quarter", "week", "day"} or lowered.endswith(
            ("_year", "_month", "_quarter", "_week", "_day")
        ):
            return SemanticRoleAssignment(
                column=name,
                role="dimension",
                reason="Numeric calendar helper used for grouping.",
            )
        if explicit_identifier:
            return SemanticRoleAssignment(
                column=name,
                role="transaction_id",
                reason="Identifier-like numeric column.",
            )
        if _contains_any(
            lowered,
            (
                "revenue",
                "sales",
                "amount",
                "profit",
                "cost",
                "value",
                "quantity",
                "price",
                "margin",
                "discount",
                "rate",
                "percent",
                "pct",
            ),
        ):
            return SemanticRoleAssignment(
                column=name,
                role="primary_measure",
                reason="Numeric business value suitable for aggregation.",
            )
        if item.unique_count <= min(20, max(2, int(non_null_count * 0.05))):
            return SemanticRoleAssignment(
                column=name,
                role="dimension",
                reason="Low-cardinality numeric grouping column.",
            )
        return SemanticRoleAssignment(
            column=name,
            role="primary_measure",
            reason="Numeric distribution suitable for measure analysis.",
        )

    if item.inferred_type == "categorical":
        role: Literal["category", "dimension"] = (
            "category"
            if _contains_any(lowered, ("category", "type", "status", "class"))
            else "dimension"
        )
        return SemanticRoleAssignment(
            column=name,
            role=role,
            reason="Low-cardinality text suitable for grouping.",
        )

    if item.inferred_type == "text":
        return SemanticRoleAssignment(
            column=name,
            role="text",
            reason="High-cardinality free-text column.",
        )

    return SemanticRoleAssignment(
        column=name,
        role="unknown",
        reason="No reliable semantic role could be inferred.",
    )

def _deterministic_plan(
    profile: DatasetProfile,
    limitation: str | None = None,
) -> PreparationPlan:
    """Build the authoritative preparation plan without an LLM call."""
    columns = {item.name: item for item in profile.column_profiles}
    semantic_roles = [
        _semantic_role_for_column(item, profile)
        for item in profile.column_profiles
    ]
    date_columns = [role.column for role in semantic_roles if role.role == "date"]
    date_column = date_columns[0] if date_columns else None
    transaction_ids = [
        role.column for role in semantic_roles if role.role == "transaction_id"
    ][:3]
    primary_measures = [
        role.column for role in semantic_roles if role.role == "primary_measure"
    ][:5]
    dimensions = [
        role.column
        for role in semantic_roles
        if role.role in {"dimension", "category", "flag"}
    ][:10]

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
    limitations = [limitation] if limitation else []
    usable_periods = _usable_period_count(profile, date_column)

    return PreparationPlan(
        semantic_roles=semantic_roles,
        date_column=date_column,
        transaction_id_columns=transaction_ids,
        primary_measures=primary_measures,
        dimensions=dimensions,
        categorical_columns=[
            item.name
            for item in profile.column_profiles
            if item.inferred_type in {"categorical", "boolean"}
        ][:10],
        currency=_detect_currency(profile),
        time_granularity=_guess_time_granularity(profile, date_column),
        time_series_candidates=primary_measures[:3] if date_column else [],
        transformations=transformations,
        capability_flags=CapabilityFlags(
            supports_kpis=bool(primary_measures),
            supports_trends=bool(
                date_column and primary_measures and usable_periods >= MIN_TREND_PERIODS
            ),
            supports_forecasting=bool(
                date_column
                and primary_measures
                and usable_periods >= MIN_FORECAST_PERIODS
            ),
            supports_anomalies=bool(
                primary_measures and profile.row_count >= MIN_ANOMALY_OBSERVATIONS
            ),
            has_temporal_data=bool(
                date_column and usable_periods >= MIN_TREND_PERIODS
            ),
        ),
        limitations=limitations,
    )

def _detect_currency(profile: DatasetProfile) -> str | None:
    return detect_currency(item.name for item in profile.column_profiles)

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

def _merge_plan_enrichment(
    base: PreparationPlan,
    suggestion: PreparationPlan,
    profile: DatasetProfile,
) -> PreparationPlan:
    """Merge safe LLM enrichment without replacing deterministic assignments."""
    known_columns = set(_profile_map(profile))
    roles_by_column = {role.column: role for role in base.semantic_roles}
    for role in suggestion.semantic_roles:
        current = roles_by_column.get(role.column)
        if role.column not in known_columns or (current and current.role != "unknown"):
            continue
        roles_by_column[role.column] = role

    ordered_roles = [
        roles_by_column[item.name]
        for item in profile.column_profiles
        if item.name in roles_by_column
    ]
    date_columns = [role.column for role in ordered_roles if role.role == "date"]
    transaction_ids = [
        role.column for role in ordered_roles if role.role == "transaction_id"
    ][:3]
    primary_measures = [
        role.column
        for role in ordered_roles
        if role.role == "primary_measure"
        and role.column in profile.candidate_numeric_columns
    ][:5]
    dimensions = [
        role.column
        for role in ordered_roles
        if role.role in {"dimension", "category", "flag"}
    ][:10]

    transformations = list(base.transformations)
    seen_transformations = {
        (item.operation, item.column) for item in transformations
    }
    for transformation in suggestion.transformations:
        key = (transformation.operation, transformation.column)
        if key in seen_transformations:
            continue
        transformations.append(transformation)
        seen_transformations.add(key)

    date_column = base.date_column or (date_columns[0] if date_columns else None)
    return base.model_copy(
        update={
            "semantic_roles": ordered_roles,
            "date_column": date_column,
            "transaction_id_columns": transaction_ids,
            "primary_measures": primary_measures,
            "dimensions": dimensions,
            "categorical_columns": _dedupe(
                [*base.categorical_columns, *suggestion.categorical_columns]
            )[:10],
            "currency": base.currency or suggestion.currency,
            "time_granularity": (
                base.time_granularity or suggestion.time_granularity
            ),
            "time_series_candidates": (
                primary_measures[:3] if date_column else []
            ),
            "transformations": transformations,
            # Capability detection remains deterministic metadata logic.
            "capability_flags": base.capability_flags,
            "limitations": _dedupe(
                [*base.limitations, *suggestion.limitations]
            ),
        }
    )

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
    granularity: TimeGranularity | None,
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
) -> TimeGranularity | None:
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
    plan_source: AgentProvider | Literal["deterministic"],
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

def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output
