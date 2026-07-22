"""Provider-neutral LLM construction and structured-output requests."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from json import JSONDecoder, dumps
from time import perf_counter
from typing import Any, TypeVar

from groq import AsyncGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from app.core.config import (
    AgentModelPolicy,
    AgentProvider,
    get_runtime_config,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_STRUCTURED_ATTEMPTS = 3
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
logger = logging.getLogger(__name__)


class ProviderConfigurationError(ValueError):
    """Raised when a selected LLM provider cannot be configured."""


class ProviderRequestError(RuntimeError):
    """A safe provider failure that does not expose prompts or credentials."""

    def __init__(
        self,
        *,
        provider: AgentProvider,
        model: str,
        category: str,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.category = category
        self.status_code = status_code
        status = f" HTTP {status_code}" if status_code is not None else ""
        super().__init__(
            f"{provider_display_name(provider)} request failed for model "
            f"{model!r} ({category}{status})."
        )


class InvalidProviderResponse(ValueError):
    """Raised when a successful HTTP response has no usable assistant content."""


def safe_model_failure_reason(error: Exception) -> str:
    """Return a public failure explanation without exposing provider payloads."""
    if isinstance(error, ProviderRequestError):
        provider = provider_display_name(error.provider)
        if error.status_code == 402:
            return f"{provider} rejected the request because the account has no available paid credit."
        if error.status_code == 429:
            return f"{provider} rate-limited the request. Free-tier models can be temporarily unavailable."
        if error.status_code == 408:
            return f"{provider} timed out before the model produced a response."
        if error.status_code is not None and error.status_code >= 500:
            return f"{provider} was temporarily unavailable (HTTP {error.status_code})."
        if error.category == "invalid_response":
            return "The model response did not match the required output format."
        return f"{provider} could not complete the model request."
    if isinstance(error, (InvalidProviderResponse, ValidationError)):
        return "The model response did not match the required output format."
    if isinstance(error, TimeoutError):
        return "The model request timed out before a response was received."
    return "The model request did not produce a usable response."


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
        response_format: dict[str, Any] | None,
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
        response_format: dict[str, Any] | None,
        messages: list[dict[str, str]],
    ) -> Any:
        request: dict[str, Any] = {
            "model": policy.model,
            "temperature": policy.temperature,
            "max_completion_tokens": policy.max_completion_tokens,
            "messages": messages,
        }
        if response_format is not None:
            request["response_format"] = response_format
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
            timeout=policy.timeout_seconds,
            max_retries=1,
            extra_body=_openrouter_extra_body(policy),
        )

    async def create_structured_completion(
        self,
        *,
        policy: AgentModelPolicy,
        response_format: dict[str, Any] | None,
        messages: list[dict[str, str]],
    ) -> Any:
        client = AsyncOpenAI(
            api_key=self.api_key(),
            base_url=OPENROUTER_BASE_URL,
            timeout=policy.timeout_seconds,
            max_retries=1,
        )
        request: dict[str, Any] = {
            "model": policy.model,
            "temperature": policy.temperature,
            "max_tokens": policy.max_completion_tokens,
            "messages": messages,
            "extra_body": _openrouter_extra_body(policy),
        }
        if response_format is not None:
            request["response_format"] = response_format
            request["extra_body"]["plugins"] = [{"id": "response-healing"}]
        return await client.chat.completions.create(
            **request,
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


def validate_active_provider_credentials() -> None:
    """Fail at startup when an active LLM provider has no credential."""
    runtime = get_runtime_config()
    policy_names = (
        ("single_dashboard", "single_chat")
        if runtime.pipeline_mode == "single"
        else (
            "data_preparation",
            "orchestrator",
            "kpi_trend",
            "anomaly_detection",
            "insight_synthesis",
            "dashboard_generation",
            "chat",
        )
    )
    providers = {runtime.agents[name].provider for name in policy_names}
    for provider in sorted(providers):
        _PROVIDERS[provider].api_key()
    logger.info(
        "Active LLM providers validated pipeline_mode=%s providers=%s",
        runtime.pipeline_mode,
        sorted(providers),
    )


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _usage_values(response: Any) -> tuple[Any, Any, Any]:
    usage = getattr(response, "usage", None)
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


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


def _response_format(
    policy: AgentModelPolicy,
    response_model: type[BaseModel],
    schema_name: str,
) -> dict[str, Any] | None:
    if policy.strict_json_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _strict_schema(response_model),
            },
        }
    if policy.supports_response_format:
        return {"type": "json_object"}
    return None


def _schema_guided_messages(
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    *,
    retry: bool,
) -> list[dict[str, str]]:
    """Add the exact client-side contract without echoing failed output."""
    schema = dumps(
        response_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prefix = (
        "The previous response could not be validated. Correct the structure. "
        if retry
        else ""
    )
    guidance = (
        f"{prefix}Return exactly one JSON object and no markdown or commentary. "
        f"It must validate against this JSON Schema: {schema}"
    )
    guided = [dict(message) for message in messages]
    for message in guided:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content', '')}\n\n{guidance}"
            break
    else:
        guided.insert(0, {"role": "system", "content": guidance})
    return guided


def _parse_structured_content(
    content: str,
    response_model: type[StructuredModel],
) -> StructuredModel:
    try:
        return response_model.model_validate_json(content)
    except ValidationError as original_error:
        decoder = JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
                return response_model.model_validate(value)
            except (ValueError, ValidationError):
                continue
        raise original_error


def _assistant_content(response: Any) -> tuple[str, str | None]:
    """Read common OpenAI-compatible response shapes without assuming choices."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InvalidProviderResponse("Provider response contained no choices.")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise InvalidProviderResponse("Provider response contained no message.")

    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json(), getattr(choice, "finish_reason", None)
    if isinstance(parsed, dict):
        return dumps(parsed, ensure_ascii=False), getattr(choice, "finish_reason", None)

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content, getattr(choice, "finish_reason", None)
    if isinstance(content, dict):
        return dumps(content, ensure_ascii=False), getattr(choice, "finish_reason", None)
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        joined = "".join(text_parts).strip()
        if joined:
            return joined, getattr(choice, "finish_reason", None)
    raise InvalidProviderResponse("Provider response contained no assistant content.")


def _is_retryable_provider_error(status_code: int | None) -> bool:
    return status_code is None or status_code in {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
    }


async def request_structured(
    *,
    policy: AgentModelPolicy,
    response_model: type[StructuredModel],
    schema_name: str,
    messages: list[dict[str, str]],
) -> StructuredModel:
    """Request and validate one structured completion from the selected provider."""
    response_format = _response_format(policy, response_model, schema_name)
    request_messages = messages
    if response_format is None or response_format.get("type") == "json_object":
        request_messages = _schema_guided_messages(
            messages,
            response_model,
            retry=False,
        )

    last_error: Exception | None = None
    for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
        started_at = perf_counter()
        logger.info(
            "LLM request started provider=%s model=%s schema=%s attempt=%s",
            policy.provider,
            policy.model,
            schema_name,
            attempt,
        )
        try:
            response = await _provider(policy).create_structured_completion(
                policy=policy,
                response_format=response_format,
                messages=request_messages,
            )
        except ProviderConfigurationError:
            raise
        except Exception as error:
            last_error = error
            status_code = _status_code(error)
            logger.warning(
                "LLM request failed provider=%s model=%s schema=%s "
                "attempt=%s error_type=%s status_code=%s latency_ms=%.1f",
                policy.provider,
                policy.model,
                schema_name,
                attempt,
                type(error).__name__,
                status_code,
                (perf_counter() - started_at) * 1000,
            )
            strict_schema_rejected = (
                status_code == 400
                and response_format is not None
                and response_format.get("type") == "json_schema"
            )
            if attempt < MAX_STRUCTURED_ATTEMPTS and strict_schema_rejected:
                response_format = {"type": "json_object"}
                request_messages = _schema_guided_messages(
                    messages,
                    response_model,
                    retry=True,
                )
                logger.info(
                    "LLM request retrying with JSON object mode provider=%s "
                    "model=%s schema=%s",
                    policy.provider,
                    policy.model,
                    schema_name,
                )
                continue
            if (
                attempt < MAX_STRUCTURED_ATTEMPTS
                and _is_retryable_provider_error(status_code)
            ):
                continue
            raise ProviderRequestError(
                provider=policy.provider,
                model=policy.model,
                category="provider_error",
                status_code=status_code,
            ) from error

        prompt_tokens, completion_tokens, total_tokens = _usage_values(response)
        logger.info(
            "LLM request completed provider=%s model=%s schema=%s attempt=%s "
            "response_id=%s prompt_tokens=%s completion_tokens=%s "
            "total_tokens=%s latency_ms=%.1f",
            policy.provider,
            policy.model,
            schema_name,
            attempt,
            getattr(response, "id", None),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            (perf_counter() - started_at) * 1000,
        )
        finish_reason: str | None = None
        try:
            content, finish_reason = _assistant_content(response)
            return _parse_structured_content(content, response_model)
        except (InvalidProviderResponse, ValidationError) as error:
            last_error = error
            logger.warning(
                "LLM response validation failed provider=%s model=%s "
                "schema=%s attempt=%s response_id=%s finish_reason=%s "
                "error_type=%s",
                policy.provider,
                policy.model,
                schema_name,
                attempt,
                getattr(response, "id", None),
                finish_reason,
                type(error).__name__,
            )
            if attempt < MAX_STRUCTURED_ATTEMPTS:
                request_messages = _schema_guided_messages(
                    messages,
                    response_model,
                    retry=True,
                )
                continue
            raise ProviderRequestError(
                provider=policy.provider,
                model=policy.model,
                category="invalid_response",
            ) from error

    raise ProviderRequestError(
        provider=policy.provider,
        model=policy.model,
        category="provider_error",
    ) from last_error
