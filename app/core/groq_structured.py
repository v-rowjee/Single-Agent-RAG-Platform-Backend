"""Small validated Groq adapter shared by structured-output agents."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from groq import AsyncGroq
from pydantic import BaseModel

from app.core.config import AgentModelPolicy


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Make a Pydantic schema compatible with Groq strict structured output."""
    schema = deepcopy(model.model_json_schema())

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


async def request_structured(
    *,
    api_key: str,
    policy: AgentModelPolicy,
    response_model: type[StructuredModel],
    schema_name: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> StructuredModel:
    """Request one compact structured completion and validate it once."""
    if policy.provider != "groq":
        raise ValueError(f"No structured-output adapter is registered for {policy.provider!r}.")
    response_format: dict[str, Any]
    if policy.strict_json_schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _strict_schema(response_model),
            },
        }
    else:
        response_format = {"type": "json_object"}

    request: dict[str, Any] = {
        "model": policy.model,
        "temperature": temperature,
        "max_completion_tokens": policy.max_completion_tokens,
        "response_format": response_format,
        "messages": messages,
    }
    if policy.reasoning_effort is not None:
        request["reasoning_effort"] = policy.reasoning_effort

    response = await AsyncGroq(api_key=api_key).chat.completions.create(**request)
    return response_model.model_validate_json(
        response.choices[0].message.content or "{}"
    )
