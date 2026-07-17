"""Central model policy for the multi-agent business-intelligence workflow."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


ReasoningEffort = Literal["none", "low", "medium", "high"] | None


@dataclass(frozen=True)
class AgentModelPolicy:
    model: str
    max_completion_tokens: int
    reasoning_effort: ReasoningEffort = None
    strict_json_schema: bool = False


_DEFAULTS: dict[str, AgentModelPolicy] = {
    "data_preparation": AgentModelPolicy(
        model="openai/gpt-oss-20b",
        max_completion_tokens=800,
        reasoning_effort="low",
        strict_json_schema=True,
    ),
    "kpi_trend": AgentModelPolicy(
        model="openai/gpt-oss-20b",
        max_completion_tokens=600,
        reasoning_effort="low",
        strict_json_schema=True,
    ),
    "anomaly_detection": AgentModelPolicy(
        model="openai/gpt-oss-20b",
        max_completion_tokens=600,
        reasoning_effort="low",
        strict_json_schema=True,
    ),
    "dashboard_generation": AgentModelPolicy(
        model="qwen/qwen3.6-27b",
        max_completion_tokens=500,
        reasoning_effort="none",
        strict_json_schema=False,
    ),
    "insight_synthesis": AgentModelPolicy(
        model="openai/gpt-oss-120b",
        max_completion_tokens=1_000,
        reasoning_effort="low",
        strict_json_schema=True,
    ),
    "chat": AgentModelPolicy(
        model="openai/gpt-oss-120b",
        max_completion_tokens=600,
        reasoning_effort="low",
        strict_json_schema=True,
    ),
}


def agent_model_policy(agent: str) -> AgentModelPolicy:
    """Return an agent policy, allowing a model-only environment override."""
    try:
        default = _DEFAULTS[agent]
    except KeyError as exc:
        raise KeyError(f"Unknown agent model policy: {agent}") from exc
    environment_name = f"GROQ_MODEL_{agent.upper()}"
    model = os.getenv(environment_name, "").strip() or default.model
    return AgentModelPolicy(
        model=model,
        max_completion_tokens=default.max_completion_tokens,
        reasoning_effort=default.reasoning_effort,
        strict_json_schema=(
            default.strict_json_schema and model.startswith("openai/gpt-oss-")
        ),
    )


def configured_agent_models() -> dict[str, str]:
    """Expose resolved assignments for diagnostics without exposing API keys."""
    return {name: agent_model_policy(name).model for name in _DEFAULTS}
