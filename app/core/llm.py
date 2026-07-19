"""Provider-neutral LLM construction and structured-output requests."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar

from groq import AsyncGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr

from app.core.config import AgentModelPolicy, AgentProvider


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class ProviderConfigurationError(ValueError):
    """Raised when a selected LLM provider cannot be configured."""


class _ProviderAdapter(ABC):
    provider: AgentProvider
    display_name: str
    api_key_environment_variable: str

    def api_key(self) -> str:
        value = os.getenv(self.api_key_environment_variable, "").strip()
        if not value:
            raise ProviderConfigurationError(
                f"{self.api_key_environment_variable} is required when "
                f"provider = {self.provider!r}."
            )
        return value

    @abstractmethod
    def create_chat_model(self, policy: AgentModelPolicy) -> BaseChatModel:
        """Create a LangChain chat model for an unstructured workflow."""

    @abstractmethod
    async def create_structured_completion(
        self,
        *,
        policy: AgentModelPolicy,
        response_format: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> Any:
        """Create one provider-native structured chat completion."""


class _GroqAdapter(_ProviderAdapter):
    provider: AgentProvider = "groq"
    display_name = "Groq"
    api_key_environment_variable = "GROQ_API_KEY"

    def create_chat_model(self, policy: AgentModelPolicy) -> BaseChatModel:
        return ChatGroq(
            model=policy.model,
            api_key=SecretStr(self.api_key()),
            temperature=policy.temperature,
            max_tokens=policy.max_completion_tokens,
            timeout=120,
            max_retries=1,
            reasoning_effort=policy.reasoning_effort,
        )

    async def create_structured_completion(
        self,
        *,
        policy: AgentModelPolicy,
        response_format: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> Any:
        request: dict[str, Any] = {
            "model": policy.model,
            "temperature": policy.temperature,
            "max_completion_tokens": policy.max_completion_tokens,
            "response_format": response_format,
            "messages": messages,
        }
        if policy.reasoning_effort is not None:
            request["reasoning_effort"] = policy.reasoning_effort
        return await AsyncGroq(api_key=self.api_key()).chat.completions.create(
            **request
        )


def _openrouter_extra_body(policy: AgentModelPolicy) -> dict[str, Any]:
    body: dict[str, Any] = {"provider": {"require_parameters": True}}
    if policy.reasoning_effort is not None:
        body["reasoning"] = {"effort": policy.reasoning_effort}
    return body


class _OpenRouterAdapter(_ProviderAdapter):
    provider: AgentProvider = "openrouter"
    display_name = "OpenRouter"
    api_key_environment_variable = "OPENROUTER_API_KEY"

    def create_chat_model(self, policy: AgentModelPolicy) -> BaseChatModel:
        return ChatOpenAI(
            model=policy.model,
            api_key=SecretStr(self.api_key()),
            base_url=OPENROUTER_BASE_URL,
            temperature=policy.temperature,
            max_completion_tokens=policy.max_completion_tokens,
            timeout=120,
            max_retries=1,
            extra_body=_openrouter_extra_body(policy),
        )

    async def create_structured_completion(
        self,
        *,
        policy: AgentModelPolicy,
        response_format: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> Any:
        client = AsyncOpenAI(
            api_key=self.api_key(),
            base_url=OPENROUTER_BASE_URL,
            timeout=120,
            max_retries=1,
        )
        return await client.chat.completions.create(
            model=policy.model,
            temperature=policy.temperature,
            max_tokens=policy.max_completion_tokens,
            response_format=response_format,
            messages=messages,  # type: ignore[arg-type]
            extra_body=_openrouter_extra_body(policy),
        )


_PROVIDERS: dict[AgentProvider, _ProviderAdapter] = {
    adapter.provider: adapter for adapter in (_GroqAdapter(), _OpenRouterAdapter())
}


def _provider(policy: AgentModelPolicy) -> _ProviderAdapter:
    try:
        return _PROVIDERS[policy.provider]
    except KeyError as exc:
        raise ProviderConfigurationError(
            f"No LLM adapter is registered for provider {policy.provider!r}."
        ) from exc


def provider_display_name(provider: AgentProvider) -> str:
    """Return a human-readable provider name for diagnostics."""
    try:
        return _PROVIDERS[provider].display_name
    except KeyError as exc:
        raise ProviderConfigurationError(
            f"No LLM adapter is registered for provider {provider!r}."
        ) from exc


def create_chat_model(policy: AgentModelPolicy) -> BaseChatModel:
    """Create the chat model selected by an agent policy."""
    return _provider(policy).create_chat_model(policy)


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Make a Pydantic schema compatible with strict structured output."""
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
    policy: AgentModelPolicy,
    response_model: type[StructuredModel],
    schema_name: str,
    messages: list[dict[str, str]],
) -> StructuredModel:
    """Request and validate one structured completion from the selected provider."""
    if policy.strict_json_schema:
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _strict_schema(response_model),
            },
        }
    else:
        response_format = {"type": "json_object"}

    response = await _provider(policy).create_structured_completion(
        policy=policy,
        response_format=response_format,
        messages=messages,
    )
    return response_model.model_validate_json(
        response.choices[0].message.content or "{}"
    )
