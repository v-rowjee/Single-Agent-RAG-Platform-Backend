"""Safe loading and rendering of version-controlled TOON agent prompt bundles."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import toons
from jinja2 import StrictUndefined, TemplateSyntaxError
from jinja2.nativetypes import NativeEnvironment


PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
_ENVIRONMENT = NativeEnvironment(
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=False,
)


class PromptTemplateError(ValueError):
    """Raised when a TOON prompt bundle is missing, invalid, or cannot be rendered."""


@dataclass(frozen=True)
class AgentPrompts:
    """Encoded system and user messages for one agent invocation."""

    system: str
    user: str


def _bundle_path(agent_name: str) -> Path:
    path = (PROMPTS_ROOT / f"{agent_name}.toon").resolve()
    if not path.is_relative_to(PROMPTS_ROOT):
        raise PromptTemplateError(f"Invalid prompt bundle name: {agent_name}")
    return path


@lru_cache(maxsize=None)
def _bundle(agent_name: str) -> dict[str, Any]:
    path = _bundle_path(agent_name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptTemplateError(f"Could not read prompt bundle {agent_name}: {exc}") from exc

    try:
        bundle = toons.loads(text, strict=True)
    except Exception as exc:
        raise PromptTemplateError(f"Invalid TOON prompt bundle {agent_name}: {exc}") from exc

    _validate_bundle(agent_name, bundle)
    return bundle


def _validate_bundle(agent_name: str, bundle: Any) -> None:
    if not isinstance(bundle, dict):
        raise PromptTemplateError(f"Prompt bundle {agent_name} must decode to an object")

    if set(bundle) != {"agent", "messages"}:
        raise PromptTemplateError(
            f"Prompt bundle {agent_name} must contain only agent and messages"
        )

    expected_agent = agent_name.rsplit("/", maxsplit=1)[-1]
    if bundle["agent"] != expected_agent:
        raise PromptTemplateError(
            f"Prompt bundle {agent_name} has an invalid agent identifier"
        )

    messages = bundle["messages"]
    if not isinstance(messages, dict) or not messages:
        raise PromptTemplateError(f"Prompt bundle {agent_name} must define message sets")

    for message_set, message_pair in messages.items():
        if not isinstance(message_set, str) or not message_set:
            raise PromptTemplateError(f"Prompt bundle {agent_name} has an invalid message set")
        _validate_message_pair(agent_name, message_set, message_pair)


def _validate_message_pair(agent_name: str, message_set: str, message_pair: Any) -> None:
    label = f"{agent_name}:{message_set}"
    if not isinstance(message_pair, dict) or set(message_pair) != {"system", "user"}:
        raise PromptTemplateError(f"Prompt bundle message set {label} must contain system and user")

    for role in ("system", "user"):
        value = message_pair[role]
        if not isinstance(value, dict) or not value:
            raise PromptTemplateError(f"Prompt bundle message set {label}.{role} must be a non-empty object")
        _validate_template_values(value, f"{label}.{role}")


def _validate_template_values(value: Any, location: str) -> None:
    if isinstance(value, str):
        try:
            _ENVIRONMENT.from_string(value)
        except TemplateSyntaxError as exc:
            raise PromptTemplateError(f"Invalid prompt template at {location}: {exc}") from exc
        return

    if value is None or isinstance(value, bool | int | float):
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_template_values(item, f"{location}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PromptTemplateError(f"Invalid non-string key at {location}")
            _validate_template_values(item, f"{location}.{key}")
        return

    raise PromptTemplateError(f"Unsupported value at {location}")


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _ENVIRONMENT.from_string(value).render(**context)
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, context) for key, item in value.items()}
    return value


def _json_compatible(value: Any) -> Any:
    """Normalise non-JSON Python values consistently before TOON encoding."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError) as exc:
        raise PromptTemplateError(f"Could not normalise prompt data for TOON encoding: {exc}") from exc


def render_agent_prompts(
    agent_name: str,
    message_set: str = "default",
    **context: Any,
) -> AgentPrompts:
    """Render one named message set and encode both roles as TOON."""
    try:
        messages = _bundle(agent_name)["messages"]
        message_pair = messages[message_set]
    except KeyError as exc:
        raise PromptTemplateError(
            f"Prompt bundle {agent_name} does not define message set {message_set}"
        ) from exc
    except Exception as exc:
        if isinstance(exc, PromptTemplateError):
            raise
        raise PromptTemplateError(f"Could not load prompt bundle {agent_name}: {exc}") from exc

    try:
        system = _render_value(message_pair["system"], context)
        user = _render_value(message_pair["user"], context)
        return AgentPrompts(
            system=toons.dumps(_json_compatible(system)).strip(),
            user=toons.dumps(_json_compatible(user)).strip(),
        )
    except Exception as exc:
        if isinstance(exc, PromptTemplateError):
            raise
        raise PromptTemplateError(
            f"Could not render prompt bundle {agent_name}:{message_set}: {exc}"
        ) from exc


def validate_prompt_bundles() -> None:
    """Decode and validate every checked-in TOON bundle at startup."""
    for path in sorted(PROMPTS_ROOT.rglob("*.toon")):
        relative = path.relative_to(PROMPTS_ROOT).as_posix()[:-5]
        _bundle(relative)
