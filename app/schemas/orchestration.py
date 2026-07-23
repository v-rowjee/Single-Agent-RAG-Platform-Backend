"""Orchestration decisions and workflow schemas."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from app.schemas.common import StrictModel

AgentName: TypeAlias = Literal["kpi_trend", "anomaly_detection", "forecasting"]
AGENT_ORDER: tuple[AgentName, ...] = ("kpi_trend", "anomaly_detection", "forecasting")

class AgentDecision(StrictModel):
    agent: AgentName
    selected: bool
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def normalize_run_alias(cls, value: Any) -> Any:
        """Accept Groq Compound's occasional ``run`` Boolean alias."""
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        run = payload.pop("run", None)
        if "selected" not in payload and isinstance(run, bool):
            payload["selected"] = run
        return payload

class OrchestrationPlan(StrictModel):
    selected_agents: list[AgentName] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_keyed_decisions(cls, value: Any) -> Any:
        """Accept Compound's occasional ``{agent: reason}`` decision format."""
        if not isinstance(value, dict) or not isinstance(value.get("decisions"), dict):
            return value

        payload = dict(value)
        selected = {
            str(agent)
            for agent in payload.get("selected_agents") or []
            if isinstance(agent, str)
        }
        decisions: list[dict[str, Any]] = []
        for agent in AGENT_ORDER:
            raw_decision = payload["decisions"].get(agent)
            if raw_decision is None:
                continue
            if isinstance(raw_decision, dict):
                decisions.append({"agent": agent, **raw_decision})
            else:
                reason = str(raw_decision).strip()
                if reason:
                    decisions.append(
                        {
                            "agent": agent,
                            "selected": agent in selected,
                            "reason": reason,
                        }
                    )
        payload["decisions"] = decisions
        return payload

    @field_validator("selected_agents")
    @classmethod
    def validate_unique_agents(
        cls,
        agents: list[AgentName],
    ) -> list[AgentName]:
        if len(agents) != len(set(agents)):
            raise ValueError("selected_agents must not contain duplicates.")

        return agents
