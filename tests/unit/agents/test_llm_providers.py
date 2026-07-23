from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import app.core.llm as llm_module
from app.schemas.data_preparation import PreparationPlan
from app.core.config import AgentModelPolicy
from app.core.llm import (
    OPENROUTER_BASE_URL,
    ProviderConfigurationError,
    ProviderRequestError,
    create_chat_model,
    request_structured,
    validate_active_provider_credentials,
)


class StructuredAnswer(BaseModel):
    answer: str


def _policy(
    provider: str,
    *,
    strict_json_schema: bool = True,
    supports_response_format: bool = True,
) -> AgentModelPolicy:
    return AgentModelPolicy(
        provider=provider,  # type: ignore[arg-type]
        model="provider/model",
        temperature=0.2,
        max_completion_tokens=321,
        reasoning_effort="low",
        strict_json_schema=strict_json_schema,
        supports_response_format=supports_response_format,
    )


def _completion(content: str = '{"answer":"ok"}') -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_groq_structured_requests_keep_the_existing_request_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        async def create(self, **request: Any) -> Any:
            captured["request"] = request
            return _completion()

    class FakeAsyncGroq:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setattr(llm_module, "AsyncGroq", FakeAsyncGroq)

    result = asyncio.run(
        request_structured(
            policy=_policy("groq"),
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Answer"}],
        )
    )

    assert result == StructuredAnswer(answer="ok")
    assert captured["client"] == {"api_key": "groq-secret"}
    request = captured["request"]
    assert request["model"] == "provider/model"
    assert request["temperature"] == 0.2
    assert request["max_completion_tokens"] == 321
    assert request["reasoning_effort"] == "low"
    assert "max_tokens" not in request
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["answer"]
    assert schema["additionalProperties"] is False


def test_data_preparation_strict_schema_has_one_numeric_value_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        async def create(self, **request: Any) -> Any:
            captured["request"] = request
            return _completion("{}")

    class FakeAsyncGroq:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setattr(llm_module, "AsyncGroq", FakeAsyncGroq)

    result = asyncio.run(
        request_structured(
            policy=_policy("groq"),
            response_model=PreparationPlan,
            schema_name="data_preparation_plan",
            messages=[{"role": "user", "content": "Return a plan"}],
        )
    )

    assert result == PreparationPlan()
    schema = captured["request"]["response_format"]["json_schema"]["schema"]
    value_schema = schema["$defs"]["PreparationTransformation"]["properties"]["value"]
    assert {item["type"] for item in value_schema["anyOf"]} == {
        "string",
        "number",
        "boolean",
        "null",
    }


def test_non_strict_data_preparation_accepts_omitted_defaulted_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        async def create(self, **request: Any) -> Any:
            captured["request"] = request
            return _completion('{"primary_measures":["revenue"]}')

    class FakeAsyncGroq:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setattr(llm_module, "AsyncGroq", FakeAsyncGroq)

    result = asyncio.run(
        request_structured(
            policy=_policy("groq", strict_json_schema=False),
            response_model=PreparationPlan,
            schema_name="data_preparation_plan",
            messages=[{"role": "user", "content": "Return a plan"}],
        )
    )

    assert result.primary_measures == ["revenue"]
    assert result.capability_flags.supports_kpis is False
    assert result.limitations == []
    assert captured["request"]["response_format"] == {"type": "json_object"}


def test_openrouter_uses_its_endpoint_and_normalized_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        async def create(self, **request: Any) -> Any:
            captured["request"] = request
            return _completion()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        request_structured(
            policy=_policy("openrouter"),
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Answer"}],
        )
    )

    assert result == StructuredAnswer(answer="ok")
    assert captured["client"] == {
        "api_key": "openrouter-secret",
        "base_url": OPENROUTER_BASE_URL,
        "timeout": 120,
        "max_retries": 1,
    }
    request = captured["request"]
    assert request["max_tokens"] == 321
    assert "max_completion_tokens" not in request
    assert request["extra_body"] == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "low"},
        "plugins": [{"id": "response-healing"}],
    }


def test_prompt_only_structured_request_omits_unsupported_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        async def create(self, **request: Any) -> Any:
            captured["request"] = request
            return _completion('Result:\n```json\n{"answer":"ok"}\n```')

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    policy = _policy(
        "openrouter",
        strict_json_schema=False,
        supports_response_format=False,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        request_structured(
            policy=policy,
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Return JSON only"}],
        )
    )

    assert result == StructuredAnswer(answer="ok")
    assert "response_format" not in captured["request"]
    system_message = captured["request"]["messages"][0]
    assert system_message["role"] == "system"
    assert "JSON Schema" in system_message["content"]


def test_invalid_response_is_retried_with_schema_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **request: Any) -> Any:
            requests.append(request)
            if len(requests) == 1:
                return _completion('{"wrong":"shape"}')
            return _completion('{"answer":"recovered"}')

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        request_structured(
            policy=_policy(
                "openrouter",
                strict_json_schema=False,
                supports_response_format=False,
            ),
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Answer"}],
        )
    )

    assert result == StructuredAnswer(answer="recovered")
    assert len(requests) == 2
    assert "previous response could not be validated" in requests[1]["messages"][0][
        "content"
    ].lower()


def test_missing_provider_choices_are_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **request: Any) -> Any:
            requests.append(request)
            if len(requests) == 1:
                return SimpleNamespace(choices=None, id="empty-response")
            return _completion('{"answer":"recovered"}')

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        request_structured(
            policy=_policy("openrouter"),
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Answer"}],
        )
    )

    assert result == StructuredAnswer(answer="recovered")
    assert len(requests) == 2


def test_rejected_strict_schema_downgrades_to_json_object_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class BadRequestError(Exception):
        status_code = 400

    class Completions:
        async def create(self, **request: Any) -> Any:
            requests.append(request)
            if len(requests) == 1:
                raise BadRequestError("private provider detail")
            return _completion('{"answer":"recovered"}')

    class FakeAsyncGroq:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setattr(llm_module, "AsyncGroq", FakeAsyncGroq)

    result = asyncio.run(
        request_structured(
            policy=_policy("groq"),
            response_model=StructuredAnswer,
            schema_name="structured_answer",
            messages=[{"role": "user", "content": "Answer"}],
        )
    )

    assert result == StructuredAnswer(answer="recovered")
    assert [
        request["response_format"]["type"] for request in requests
    ] == ["json_schema", "json_object"]
    assert "JSON Schema" in requests[1]["messages"][0]["content"]


def test_chat_model_factory_dispatches_from_the_agent_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, Any]] = {}
    groq_model = object()
    openrouter_model = object()

    def fake_groq(**kwargs: Any) -> object:
        captured["groq"] = kwargs
        return groq_model

    def fake_openrouter(**kwargs: Any) -> object:
        captured["openrouter"] = kwargs
        return openrouter_model

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "ChatGroq", fake_groq)
    monkeypatch.setattr(llm_module, "ChatOpenAI", fake_openrouter)

    assert create_chat_model(_policy("groq")) is groq_model
    assert create_chat_model(_policy("openrouter")) is openrouter_model

    assert captured["groq"]["max_tokens"] == 321
    assert captured["groq"]["reasoning_effort"] == "low"
    assert captured["openrouter"]["base_url"] == OPENROUTER_BASE_URL
    assert "default_headers" not in captured["openrouter"]
    assert captured["openrouter"]["max_completion_tokens"] == 321
    assert captured["openrouter"]["extra_body"]["reasoning"] == {"effort": "low"}


def test_only_the_selected_provider_credential_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    with pytest.raises(ProviderConfigurationError, match="OPENROUTER_API_KEY"):
        create_chat_model(_policy("openrouter"))


def test_active_provider_credentials_are_validated_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ProviderConfigurationError, match="OPENROUTER_API_KEY"):
        validate_active_provider_credentials()


def test_provider_errors_are_wrapped_without_exposing_sensitive_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RateLimitError(Exception):
        status_code = 429

    class Completions:
        async def create(self, **request: Any) -> Any:
            del request
            raise RateLimitError("secret-key and private prompt")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            request_structured(
                policy=_policy("openrouter"),
                response_model=StructuredAnswer,
                schema_name="structured_answer",
                messages=[{"role": "user", "content": "private prompt"}],
            )
        )

    assert error.value.provider == "openrouter"
    assert error.value.status_code == 429
    assert "secret-key" not in caplog.text
    assert "private prompt" not in caplog.text


def test_invalid_provider_output_raises_safe_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    class Completions:
        async def create(self, **request: Any) -> Any:
            nonlocal call_count
            del request
            call_count += 1
            return _completion("not-json sensitive output")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            request_structured(
                policy=_policy("openrouter"),
                response_model=StructuredAnswer,
                schema_name="structured_answer",
                messages=[{"role": "user", "content": "Answer"}],
            )
        )

    assert error.value.category == "invalid_response"
    assert call_count == llm_module.MAX_STRUCTURED_ATTEMPTS
    assert "sensitive output" not in str(error.value)


def test_provider_timeout_is_wrapped_as_a_safe_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completions:
        async def create(self, **request: Any) -> Any:
            del request
            raise TimeoutError("private timeout detail")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", FakeAsyncOpenAI)

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(
            request_structured(
                policy=_policy("openrouter"),
                response_model=StructuredAnswer,
                schema_name="structured_answer",
                messages=[{"role": "user", "content": "Answer"}],
            )
        )

    assert error.value.category == "provider_error"
    assert error.value.status_code is None
    assert "private timeout detail" not in str(error.value)
