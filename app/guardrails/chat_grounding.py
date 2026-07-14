"""Final deterministic validation for retrieved-evidence chat drafts."""
from __future__ import annotations

import re

from app.agents.multi.chat_agent import GroundedChatDraft, INSUFFICIENT_CONTEXT_ANSWER
from app.rag.models import RetrievedDocument


CAUSAL_TERMS = ("caused", "cause", "because", "responsible", "driver")
CAUSAL_FALLBACK = (
    "The available analysis shows the observed change but does not identify who "
    "caused it."
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?!\w)")


def _source_ids(document: RetrievedDocument) -> list[str]:
    values = [document.metadata.get("source_id")]
    if isinstance(document.metadata.get("source_ids"), list):
        values.extend(document.metadata["source_ids"])
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _ordered_valid_source_ids(
    source_ids: list[str], documents: list[RetrievedDocument]
) -> list[str]:
    requested = {str(value).strip() for value in source_ids if str(value).strip()}
    output: list[str] = []
    for document in documents:
        for source_id in _source_ids(document):
            if source_id in requested and source_id not in output:
                output.append(source_id)
    return output


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in NUMBER_PATTERN.finditer(text)}


class ChatGroundingGuardrail:
    def validate(
        self,
        query: str,
        retrieved_documents: list[RetrievedDocument],
        draft: GroundedChatDraft,
    ) -> GroundedChatDraft:
        if draft.insufficient_context or not retrieved_documents:
            return GroundedChatDraft(
                answer=draft.answer or INSUFFICIENT_CONTEXT_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )

        source_ids = _ordered_valid_source_ids(draft.source_ids, retrieved_documents)
        if not source_ids:
            return GroundedChatDraft(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )

        evidence = "\n".join(document.page_content for document in retrieved_documents)
        query_is_causal = any(term in query.casefold() for term in CAUSAL_TERMS)
        evidence_supports_causation = any(
            term in evidence.casefold() for term in CAUSAL_TERMS
        )
        if query_is_causal and not evidence_supports_causation:
            return GroundedChatDraft(
                answer=CAUSAL_FALLBACK,
                source_ids=source_ids,
                insufficient_context=False,
            )

        unsupported_numbers = _numbers(draft.answer) - _numbers(evidence)
        if unsupported_numbers:
            return GroundedChatDraft(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                source_ids=[],
                insufficient_context=True,
            )
        return draft.model_copy(update={"source_ids": source_ids})


chat_grounding_guardrail = ChatGroundingGuardrail()
