"""Capability-gated orchestrator for the multi-agent BI workflow."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.prompt_loader import render_agent_prompts
from app.schemas.orchestration import (
    AGENT_ORDER,
    AgentDecision,
    AgentName,
    OrchestrationPlan,
)

MIN_TREND_PERIODS = 2
MIN_FORECAST_PERIODS = 4
MAX_ORCHESTRATION_COLUMNS = 50
MAX_ORCHESTRATION_SAMPLE_VALUES = 5
MAX_ORCHESTRATION_SAMPLE_LENGTH = 80
MAX_ORCHESTRATION_PAYLOAD_BYTES = 100_000

logger = logging.getLogger(__name__)


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator cannot read its required input."""


class OrchestrationPayloadTooLarge(OrchestratorError):
    """Raised before provider invocation when compact messages exceed the limit."""


def _as_positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def detect_analysis_capabilities(
    prepared_dataset: dict[str, Any],
) -> dict[str, bool]:
    """Derive specialist applicability from dataset metadata, not an LLM."""
    profile = prepared_dataset.get("dataset_profile")
    profile = profile if isinstance(profile, dict) else {}
    raw_columns = profile.get("column_profiles")
    raw_columns = raw_columns if isinstance(raw_columns, list) else []

    profiled_numeric = {
        str(column.get("name"))
        for column in raw_columns
        if isinstance(column, dict)
        and column.get("inferred_type") == "numeric"
        and isinstance(column.get("name"), str)
    }
    profiled_temporal = {
        str(column.get("name"))
        for column in raw_columns
        if isinstance(column, dict)
        and column.get("inferred_type") == "date"
        and isinstance(column.get("name"), str)
    }
    primary_measures = {
        item
        for item in prepared_dataset.get("primary_measures") or []
        if isinstance(item, str) and item.strip()
    }
    numeric_columns = profiled_numeric | primary_measures

    temporal_profile = prepared_dataset.get("temporal_profile")
    temporal_profile = temporal_profile if isinstance(temporal_profile, dict) else {}
    date_column = prepared_dataset.get("date_column") or temporal_profile.get(
        "date_column"
    )
    temporal_columns = set(profiled_temporal)
    if isinstance(date_column, str) and date_column.strip():
        temporal_columns.add(date_column)

    unique_periods = _as_positive_int(temporal_profile.get("unique_periods"))
    if unique_periods == 0 and temporal_columns:
        unique_periods = max(
            (
                _as_positive_int(column.get("unique_count"))
                for column in raw_columns
                if isinstance(column, dict)
                and column.get("name") in temporal_columns
            ),
            default=0,
        )

    has_numeric_measure = bool(numeric_columns)
    has_temporal_column = bool(temporal_columns)
    enough_trend_periods = unique_periods == 0 or unique_periods >= MIN_TREND_PERIODS
    enough_forecast_periods = (
        unique_periods == 0 or unique_periods >= MIN_FORECAST_PERIODS
    )
    row_count = _as_positive_int(profile.get("row_count"))
    enough_anomaly_rows = row_count == 0 or row_count >= 8

    return {
        "supports_kpis": has_numeric_measure,
        "supports_trends": (
            has_numeric_measure and has_temporal_column and enough_trend_periods
        ),
        "supports_anomalies": has_numeric_measure and enough_anomaly_rows,
        "supports_forecasting": (
            has_numeric_measure
            and has_temporal_column
            and enough_forecast_periods
        ),
        "has_temporal_data": has_temporal_column and enough_trend_periods,
    }


def _supported_agents(
    capabilities: dict[str, bool],
) -> set[AgentName]:
    supported: set[AgentName] = set()

    if (
        capabilities["supports_kpis"]
        or capabilities["supports_trends"]
    ):
        supported.add("kpi_trend")

    if capabilities["supports_anomalies"]:
        supported.add("anomaly_detection")

    if capabilities["supports_forecasting"]:
        supported.add("forecasting")

    return supported


def _short_text(value: Any, limit: int = MAX_ORCHESTRATION_SAMPLE_LENGTH) -> str:
    return str(value)[:limit]


def _string_list(value: Any, limit: int = MAX_ORCHESTRATION_COLUMNS) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _short_text(item)
        for item in value
        if isinstance(item, str) and item.strip()
    ][:limit]


def _sample_values(value: Any) -> list[str | int | float | bool | None]:
    if not isinstance(value, list):
        return []

    compact: list[str | int | float | bool | None] = []
    for item in value[:MAX_ORCHESTRATION_SAMPLE_VALUES]:
        if item is None or isinstance(item, bool | int | float):
            compact.append(item)
        else:
            compact.append(_short_text(item))
    return compact


def build_orchestration_context(
    prepared_dataset: dict[str, Any],
    capabilities: dict[str, bool],
) -> dict[str, Any]:
    """Build the bounded metadata-only context used for Compound routing."""
    profile = prepared_dataset.get("dataset_profile")
    profile = profile if isinstance(profile, dict) else {}
    raw_columns = profile.get("column_profiles")
    raw_columns = raw_columns if isinstance(raw_columns, list) else []

    columns: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    temporal_columns: list[str] = []
    semantic_column_map = prepared_dataset.get("semantic_column_map")
    semantic_column_map = (
        semantic_column_map if isinstance(semantic_column_map, dict) else {}
    )
    prepared_measure_list = [
        item
        for item in prepared_dataset.get("primary_measures") or []
        if isinstance(item, str) and item.strip()
    ]
    prepared_measures = set(prepared_measure_list)
    prepared_date = prepared_dataset.get("date_column")

    for raw_column in raw_columns[:MAX_ORCHESTRATION_COLUMNS]:
        if not isinstance(raw_column, dict):
            continue
        name = raw_column.get("name")
        inferred_type = raw_column.get("inferred_type")
        if not isinstance(name, str) or not name.strip():
            continue

        compact_name = _short_text(name)
        compact_type = (
            "datetime"
            if inferred_type == "date"
            else _short_text(inferred_type)
            if isinstance(inferred_type, str)
            else "unknown"
        )
        semantic_role = semantic_column_map.get(name)
        if not isinstance(semantic_role, str) or not semantic_role.strip():
            if name == prepared_date or inferred_type == "date":
                semantic_role = "date"
            elif name in prepared_measures:
                semantic_role = "primary_measure"
            elif inferred_type == "numeric":
                semantic_role = "measure"
            elif inferred_type == "boolean":
                semantic_role = "flag"
            elif inferred_type == "categorical":
                semantic_role = "dimension"
            elif inferred_type == "text":
                semantic_role = "text"
            else:
                semantic_role = "unknown"
        columns.append(
            {
                "name": compact_name,
                "dtype": compact_type,
                "semantic_role": _short_text(semantic_role),
                "missing_percentage": raw_column.get("null_percentage"),
                "unique_count": _as_positive_int(raw_column.get("unique_count")),
                "sample_values": _sample_values(raw_column.get("sample_values")),
            }
        )
        if inferred_type == "numeric":
            numeric_columns.append(compact_name)
        elif inferred_type in {"categorical", "boolean"}:
            categorical_columns.append(compact_name)
        elif inferred_type == "date":
            temporal_columns.append(compact_name)

    for measure in prepared_measure_list:
        compact_measure = _short_text(measure)
        if compact_measure not in numeric_columns:
            numeric_columns.append(compact_measure)

    date_column = prepared_dataset.get("date_column")
    if isinstance(date_column, str) and date_column.strip() and date_column not in temporal_columns:
        temporal_columns.append(_short_text(date_column))

    return {
        "dataset_id": _short_text(prepared_dataset.get("dataset_id"), 128)
        if prepared_dataset.get("dataset_id") is not None
        else None,
        "row_count": _as_positive_int(profile.get("row_count")),
        "column_count": _as_positive_int(profile.get("column_count")),
        "columns": columns,
        "numeric_columns": _string_list(numeric_columns),
        "categorical_columns": _string_list(categorical_columns),
        "temporal_columns": _string_list(temporal_columns),
        "available_capabilities": {
            "kpi_analysis": capabilities["supports_kpis"],
            "trend_analysis": capabilities["supports_trends"],
            "anomaly_detection": capabilities["supports_anomalies"],
            "forecasting": capabilities["supports_forecasting"],
        },
    }


def _deterministic_routing_plan(
    supported_agents: set[AgentName],
) -> OrchestrationPlan:
    selected_agents = [
        agent
        for agent in AGENT_ORDER
        if agent in supported_agents
    ]

    decisions = [
        AgentDecision(
            agent=agent,
            selected=agent in supported_agents,
            reason=(
                "Selected because the prepared dataset supports this analysis."
                if agent in supported_agents
                else "Not selected because the prepared dataset does not support this analysis."
            ),
        )
        for agent in AGENT_ORDER
    ]

    return OrchestrationPlan(
        selected_agents=selected_agents,
        decisions=decisions,
    )


Planner: TypeAlias = Callable[
    [dict[str, Any], set[AgentName]],
    Awaitable[OrchestrationPlan],
]


def _max_orchestration_payload_bytes() -> int:
    raw_value = os.getenv("MAX_ORCHESTRATION_PAYLOAD_BYTES")
    if raw_value is None:
        return MAX_ORCHESTRATION_PAYLOAD_BYTES
    try:
        configured = int(raw_value)
    except ValueError:
        return MAX_ORCHESTRATION_PAYLOAD_BYTES
    return configured if configured > 0 else MAX_ORCHESTRATION_PAYLOAD_BYTES


def orchestration_request_size(messages: list[dict[str, str]]) -> int:
    payload_json = json.dumps(messages, ensure_ascii=False, default=str)
    return len(payload_json.encode("utf-8"))


async def _request_plan(
    routing_context: dict[str, Any],
    supported_agents: set[AgentName],
) -> OrchestrationPlan:
    del supported_agents
    prompts = render_agent_prompts(
        "multi/orchestrator",
        payload=routing_context,
    )
    messages = [
        {"role": "system", "content": prompts.system},
        {"role": "user", "content": prompts.user},
    ]
    payload_size_bytes = orchestration_request_size(messages)
    logger.info(
        "Compound orchestration request size: %s bytes",
        payload_size_bytes,
    )
    max_payload_bytes = _max_orchestration_payload_bytes()
    if payload_size_bytes > max_payload_bytes:
        raise OrchestrationPayloadTooLarge(
            f"compact request is {payload_size_bytes} bytes; limit is "
            f"{max_payload_bytes} bytes"
        )
    return await request_structured(
        policy=agent_model_policy("orchestrator"),
        response_model=OrchestrationPlan,
        schema_name="orchestration_plan",
        messages=messages,
    )


def _capability_gated_plan(
    proposed: OrchestrationPlan,
    supported_agents: set[AgentName],
) -> OrchestrationPlan:
    """Keep Compound's routing decision inside deterministic capability gates."""
    proposed_selected = set(proposed.selected_agents)
    proposed_decisions = {
        decision.agent: decision for decision in proposed.decisions
    }
    selected_agents = [
        agent
        for agent in AGENT_ORDER
        if agent in supported_agents and agent in proposed_selected
    ]
    decisions: list[AgentDecision] = []

    for agent in AGENT_ORDER:
        selected = agent in selected_agents
        proposed_decision = proposed_decisions.get(agent)

        if agent not in supported_agents:
            reason = (
                "Not selected because the dataset does not support this analysis."
            )
        elif proposed_decision is not None:
            reason = proposed_decision.reason
        elif selected:
            reason = "Selected by the orchestrator from the eligible specialists."
        else:
            reason = (
                "Not selected because the orchestrator found no additional value "
                "for this dataset."
            )

        decisions.append(
            AgentDecision(
                agent=agent,
                selected=selected,
                reason=reason,
            )
        )

    return OrchestrationPlan(
        selected_agents=selected_agents,
        decisions=decisions,
    )


class OrchestratorAgent:
    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner

    async def run(
        self,
        prepared_dataset: dict[str, Any],
    ) -> OrchestrationPlan:
        if not isinstance(prepared_dataset, dict):
            raise OrchestratorError(
                "prepared_dataset must be a dictionary."
            )

        capabilities = detect_analysis_capabilities(prepared_dataset)
        supported_agents = _supported_agents(capabilities)
        routing_context = build_orchestration_context(
            prepared_dataset,
            capabilities,
        )

        logger.info(
            "Orchestration started with capabilities: %s",
            capabilities,
        )

        if self._planner is None or not supported_agents:
            result = _deterministic_routing_plan(supported_agents)
            logger.info("Deterministic capability routing completed.")
        else:
            try:
                proposed = await self._planner(routing_context, supported_agents)
                result = _capability_gated_plan(proposed, supported_agents)
                logger.info("Compound orchestration completed.")
            except OrchestrationPayloadTooLarge as exc:
                logger.warning(
                    "Compound orchestration skipped because the compact request "
                    "was still too large: %s",
                    exc,
                )
                result = _deterministic_routing_plan(supported_agents)
            except Exception as exc:
                logger.warning(
                    "Compound orchestration failed; using capability routing: %s",
                    exc,
                )
                result = _deterministic_routing_plan(supported_agents)

        logger.info(
            "Selected specialist agents: %s",
            result.selected_agents,
        )

        return result


orchestrator_agent = OrchestratorAgent()


async def orchestrator_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    prepared_dataset = state.get("prepared_dataset")

    if not isinstance(prepared_dataset, dict):
        raise OrchestratorError(
            "state.prepared_dataset is required."
        )

    result = await orchestrator_agent.run(prepared_dataset)

    return {
        "orchestration_plan": result.model_dump(mode="json"),
        "completed_agents": ["orchestrator"],
        "skipped_agents": [
            agent for agent in AGENT_ORDER if agent not in result.selected_agents
        ],
    }
