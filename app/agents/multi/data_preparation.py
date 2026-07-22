from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.core.config import AgentProvider, agent_model_policy
from app.core.exceptions import DataPreparationError
from app.core.llm import provider_display_name, request_structured
from app.core.model_policy import agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.data_preparation import (
    DatasetProfile,
    GenericCleaningResult,
    PreparationPlan,
    PreparedDatasetPackage,
)
from app.services.data.cleaning import _generic_clean_csv
from app.services.data.preparation import (
    _dedupe,
    _deterministic_plan,
    _execute_plan,
    _merge_plan_enrichment,
    _reconcile_temporal_capabilities,
    _temporal_profile,
    _validate_plan,
)
from app.services.data.profiling import _profile_dataset


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

logger = logging.getLogger(__name__)


def _compact_profile_payload(profile: DatasetProfile) -> dict[str, Any]:
    return profile.model_dump(mode="json", exclude_none=True)


async def _request_plan(profile: DatasetProfile) -> PreparationPlan:
    prompts = render_agent_prompts(
        "multi/data_preparation",
        supported_operations=sorted(SUPPORTED_OPERATIONS),
        supported_formulas=sorted(SUPPORTED_FORMULAS),
        profile=_compact_profile_payload(profile),
        output_schema=PreparationPlan.model_json_schema(mode="serialization"),
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


async def _plan_with_optional_enrichment(
    profile: DatasetProfile,
) -> tuple[PreparationPlan, AgentProvider | Literal["deterministic"], list[str]]:
    base_plan = _deterministic_plan(profile)
    warnings: list[str] = []
    policy = agent_model_policy("data_preparation")
    try:
        suggestion = await _request_plan(profile)
        return (
            _merge_plan_enrichment(base_plan, suggestion, profile),
            policy.provider,
            warnings,
        )
    except Exception as error:
        logger.warning(
            "Optional data preparation enrichment failed; deterministic plan retained "
            "provider=%s model=%s error=%s",
            policy.provider,
            policy.model,
            error,
        )
        warnings.append(
            f"{provider_display_name(policy.provider)} preparation enrichment was "
            "unavailable; deterministic preparation was retained."
        )
        return base_plan, "deterministic", warnings


class DataPreparationAgent:
    def __init__(self, *, enable_llm_enrichment: bool = False) -> None:
        self.enable_llm_enrichment = enable_llm_enrichment

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
            raise DataPreparationError("A temporary processing workspace is required.")
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
        if self.enable_llm_enrichment:
            raw_plan, plan_source, planning_warnings = (
                await _plan_with_optional_enrichment(profile)
            )
        else:
            raw_plan = _deterministic_plan(profile)
            plan_source = "deterministic"
            planning_warnings = []

        try:
            plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
        except Exception as exc:
            planning_warnings.append(
                f"Plan enrichment validation failed; deterministic plan retained: {exc}"
            )
            raw_plan = _deterministic_plan(profile)
            plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
            plan_source = "deterministic"

        if plan_source == "deterministic":
            logger.info(
                "Data preparation using deterministic plan session_id=%s", session_id
            )
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
            currency=prepared_profile.currency,
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


data_preparation_agent = DataPreparationAgent(enable_llm_enrichment=True)


async def data_preparation_node(state: dict[str, Any]) -> dict[str, Any]:
    session_id = str(state.get("session_id") or state.get("sessionId") or "").strip()
    uploaded_file_path = str(
        state.get("uploaded_file_path")
        or state.get("upload_path")
        or state.get("filePath")
        or ""
    ).strip()
    business_description = state.get("business_description") or state.get(
        "businessDescription"
    )
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
        business_description=(
            str(business_description) if business_description else None
        ),
        generic_cleaning_report=(
            GenericCleaningResult.model_validate(state["generic_cleaning_report"])
            if state.get("generic_cleaned_file_path")
            and isinstance(state.get("generic_cleaning_report"), dict)
            else None
        ),
        file_name=str(file_name) if file_name else None,
        output_dir=Path(working_directory),
    )

    prepared_dataset = result.model_dump(mode="json")
    dataset_id = str(state.get("dataset_id") or state.get("datasetId") or "").strip()
    if dataset_id:
        prepared_dataset["dataset_id"] = dataset_id
    source_datasets = state.get("source_datasets")
    if isinstance(source_datasets, list):
        prepared_dataset["source_datasets"] = source_datasets

    return {
        "generic_cleaned_file_path": result.cleaning_report.cleaned_file_path,
        "prepared_file_path": result.prepared_file_path,
        "generic_cleaning_report": result.cleaning_report.model_dump(mode="json"),
        "prepared_dataset": prepared_dataset,
        "warnings": result.warnings,
        "completed_agents": ["data_preparation"],
        "model_invocations": [
            agent_model_usage(
                "data_preparation",
                (
                    "fallback"
                    if result.preparation_report.plan_source == "deterministic"
                    else "succeeded"
                ),
            )
        ],
    }
