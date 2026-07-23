"""Dataset and session indexing application service."""

from __future__ import annotations

from typing import Any

from app.rag.retrieval.retriever import Retriever, retriever


class IndexingService:
    """Own indexing use cases independently from query retrieval."""

    def __init__(self, retrieval: Retriever | Any | None = None) -> None:
        self._retrieval = retrieval or retriever

    def index_dataset(self, agent_input: Any, profile: dict[str, Any], force: bool = False):
        return self._retrieval.index_dataset(agent_input, profile, force)

    def build_dataset_documents(
        self,
        agent_input: Any,
        profile: dict[str, Any],
        workspace_session_id: str | None = None,
    ):
        return self._retrieval.build_dataset_documents(
            agent_input,
            profile,
            workspace_session_id,
        )

    def index_documents(
        self,
        session_id: str,
        dataset_id: str,
        retrieval_documents: list[dict[str, Any]],
    ):
        return self._retrieval.index_documents(
            session_id,
            dataset_id,
            retrieval_documents,
        )

    def ensure_index(self, agent_input: Any, profile: dict[str, Any]) -> bool:
        return self._retrieval.ensure_index(agent_input, profile)


indexing_service = IndexingService()
