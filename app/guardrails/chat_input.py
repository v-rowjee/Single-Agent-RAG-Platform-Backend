"""Deterministic input validation for the multi-agent chat pipeline."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


BLOCKED_CHAT_ANSWER = (
    "I cannot follow requests to reveal secrets or override the analysis "
    "assistant's instructions. Please ask a question about this dataset's "
    "analysis instead."
)


class ChatInputGuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    query: str = Field(min_length=1)
    blocked_answer: str | None = None


_BLOCKED_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        (
            r"\b(ignore|disregard|override|forget|bypass)\b.{0,80}"
            r"\b(previous|prior|system|developer|assistant|security)\b.{0,40}"
            r"\b(instruction|instructions|prompt|message|rules?)\b"
        ),
        (
            r"\b(reveal|show|print|display|expose|leak|return|give\s+me)\b.{0,100}"
            r"\b(system\s+prompt|developer\s+message|hidden\s+instructions?|"
            r"api[_\s-]?key|service[_\s-]?role|password|access[_\s-]?token|secret)\b"
        ),
        (
            r"\b(system\s+prompt|developer\s+message|hidden\s+instructions?|"
            r"api[_\s-]?key|service[_\s-]?role|password|access[_\s-]?token|secret)\b"
            r".{0,100}\b(reveal|show|print|display|expose|leak|return)\b"
        ),
        r"\b(jailbreak|prompt\s+injection|developer\s+mode)\b",
    )
)


class ChatInputGuardrail:
    def validate(self, query: str) -> ChatInputGuardrailResult:
        cleaned = str(query or "").strip()
        if not cleaned:
            raise ValueError("The chat query cannot be empty.")

        if any(pattern.search(cleaned) for pattern in _BLOCKED_PATTERNS):
            return ChatInputGuardrailResult(
                allowed=False,
                query=cleaned,
                blocked_answer=BLOCKED_CHAT_ANSWER,
            )

        return ChatInputGuardrailResult(allowed=True, query=cleaned)


chat_input_guardrail = ChatInputGuardrail()
