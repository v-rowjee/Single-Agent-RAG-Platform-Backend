"""Business-intelligence chat orchestration and message persistence."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.single import business_intelligence as single_agent_module
from app.core.config import Settings
from app.core.exceptions import SessionNotFoundError
from app.core.model_policy import chat_model_usage
from app.schemas.api import BusinessIntelligenceAgentInput, ChatResponse
from app.services.analysis.files import DatasetFileService
from app.services.analysis.workspaces import WorkspaceService
from app.services.persistence.analysis import DatasetRecord
from app.services.persistence.messages import MessageRecord, MessageRepository

logger = logging.getLogger(__name__)


class BusinessIntelligenceChatService:
    def __init__(
        self,
        *,
        workspaces: WorkspaceService,
        messages: MessageRepository | Any,
        storage: Any,
        retriever: Any,
        chat_graph: Any,
        settings: Settings,
        files: DatasetFileService,
        single_agent: Any | None = None,
    ) -> None:
        self.workspaces = workspaces
        self.messages = messages
        self.storage = storage
        self.retriever = retriever
        self.chat_graph = chat_graph
        self.settings = settings
        self.files = files
        self.single_agent = single_agent

    def chat(self, session_id: str, query: str, user_id: str) -> ChatResponse:
        session, datasets = self.workspaces.load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat is available."
            )
        if session.rag_status == "indexing":
            raise ValueError(
                "The analysis is ready, but its retrieval index is still being "
                "prepared. Try chat again shortly."
            )
        if session.rag_status != "ready":
            raise ValueError(
                "The retrieval index is unavailable. Rebuild the workspace before "
                "using chat."
            )
        if (
            self.settings.bi_pipeline_mode == "multi"
            or len(datasets) > 1
            or datasets[0].session_id is not None
        ):
            return self.chat_with_multi_agent_pipeline(session.id, query, datasets)

        dataset = datasets[0]
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")
        history = self.chat_history(dataset.id)
        self.messages.save_message(
            dataset_id=dataset.id,
            role="user",
            content=cleaned_query,
            sources=[],
        )
        return self.chat_with_single_agent(dataset, cleaned_query, history)

    def chat_with_multi_agent_pipeline(
        self,
        session_id: str,
        query: str,
        datasets: list[DatasetRecord],
    ) -> ChatResponse:
        history = self.chat_history(session_id)
        selected, _ = self.select_chat_datasets(query, datasets)
        scoped = [selected] if selected is not None else datasets
        calculated = self.workspace_calculation_response(query, scoped)
        retrieval_query = (
            f"In dataset `{selected.file_name}`, {query}"
            if selected is not None
            else query
        )
        self.messages.save_message(
            dataset_id=session_id,
            role="user",
            content=query.strip(),
            sources=[],
        )
        if calculated is not None:
            return self.save_chat_response(
                session_id,
                calculated,
                [dataset.id for dataset in scoped],
            )
        result = self.chat_graph.answer(
            session_id,
            retrieval_query,
            history=history,
        )
        return self.save_chat_response(
            session_id,
            result.draft.answer,
            result.draft.source_ids,
        )

    @staticmethod
    def select_chat_datasets(
        query: str,
        datasets: list[DatasetRecord],
    ) -> tuple[DatasetRecord | None, list[DatasetRecord]]:
        if len(datasets) <= 1:
            return None, []
        normalized_query = query.casefold()
        named = [
            dataset
            for dataset in datasets
            if dataset.file_name.casefold() in normalized_query
            or Path(dataset.file_name).stem.casefold() in normalized_query
        ]
        return (named[0], []) if len(named) == 1 else (None, [])

    def workspace_calculation_response(
        self,
        query: str,
        datasets: list[DatasetRecord],
    ) -> str | None:
        if not datasets:
            return None
        try:
            frames: list[pd.DataFrame] = []
            canonical_columns: dict[str, str] = {}
            for dataset in datasets:
                frame = self.files.read_dataframe(
                    dataset.file_name,
                    self.storage.download_file(dataset.storage_path),
                )
                rename: dict[Any, str] = {}
                occupied = {str(column) for column in frame.columns}
                for column in frame.columns:
                    name = str(column)
                    normalized = re.sub(
                        r"[^a-z0-9]+",
                        "_",
                        name.casefold(),
                    ).strip("_")
                    canonical = canonical_columns.setdefault(
                        normalized or name.casefold(),
                        name,
                    )
                    if canonical != name and canonical not in occupied:
                        rename[column] = canonical
                if rename:
                    frame = frame.rename(columns=rename)
                frame["__source_dataset__"] = dataset.file_name
                frames.append(frame)
            if not frames:
                return None
            combined = pd.concat(frames, ignore_index=True, sort=False)
            if combined.empty:
                return None
            numeric_columns = [
                str(column)
                for column in combined.columns
                if column != "__source_dataset__"
                and pd.to_numeric(combined[column], errors="coerce").notna().any()
            ]
            dimensions = [
                str(column)
                for column in combined.columns
                if str(column) not in numeric_columns
            ]
            date_field = next(
                (
                    str(column)
                    for column in combined.columns
                    if any(
                        term in str(column).casefold()
                        for term in ("date", "time", "year", "month", "period")
                    )
                ),
                None,
            )
            profile = {
                "summary": {
                    "measures": numeric_columns,
                    "dimensions": dimensions,
                    "timeField": date_field,
                }
            }
            query_type = self.retriever.route_query(query, profile)
            if query_type not in {
                "calculation",
                "comparison",
                "forecast",
                "mixed",
            }:
                return None
            with tempfile.TemporaryDirectory(prefix="bi_workspace_chat_") as directory:
                combined_path = Path(directory) / "all_uploaded_datasets.csv"
                combined.to_csv(combined_path, index=False)
                agent_input = BusinessIntelligenceAgentInput(
                    sessionId=datasets[0].session_id or datasets[0].id,
                    datasetId=datasets[0].session_id or datasets[0].id,
                    filePath=str(combined_path),
                    fileName="all uploaded datasets",
                    description=datasets[0].description,
                )
                evidence = self.retriever.calculate_evidence(
                    agent_input=agent_input,
                    query=query,
                    query_type=query_type,
                    profile=profile,
                )
            if evidence is None or not evidence.direct_answer:
                return None
            file_names = ", ".join(
                f"`{dataset.file_name}`" for dataset in datasets
            )
            return (
                f"{evidence.direct_answer.rstrip()} "
                f"Dataset scope: {file_names}."
            )
        except Exception:
            logger.exception(
                "Workspace chat calculation failed datasets=%s",
                [dataset.id for dataset in datasets],
            )
            return None

    def chat_with_single_agent(
        self,
        dataset: DatasetRecord,
        query: str,
        history: list[dict[str, str]],
    ) -> ChatResponse:
        try:
            response, source_ids = self.chat_with_agent(
                dataset,
                self.storage.download_file(dataset.storage_path),
                query,
                history,
            )
        except Exception:
            logger.exception(
                "Single-agent chat preparation failed session_id=%s",
                dataset.id,
            )
            response = (
                "The analysis assistant could not answer this question at the moment."
            )
            source_ids = []
        return self.save_chat_response(dataset.id, response, source_ids)

    def chat_with_agent(
        self,
        dataset: DatasetRecord,
        content: bytes,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        try:
            with self.files.temporary_agent_input(dataset, content) as agent_input:
                agent = (
                    self.single_agent
                    or single_agent_module.business_intelligence_agent
                )
                response = agent.chat(
                    agent_input=agent_input,
                    query=query,
                    history=history,
                )
                source_ids = agent.source_ids_for_session(dataset.id)
            return response, source_ids
        except Exception:
            logger.exception(
                "Business intelligence agent failed session_id=%s operation=chat",
                dataset.id,
            )
            return (
                "**Answer:** I cannot answer from the dataset profile because "
                "the AI business intelligence agent is currently unavailable.\n\n"
                f"**Grounding:** Dataset '{dataset.file_name}'; user asked '{query}'.",
                [],
            )

    def chat_history(self, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages.get_recent_messages(session_id, limit=10)
        ]

    def save_chat_response(
        self,
        session_id: str,
        response_text: str,
        source_ids: list[str],
    ) -> ChatResponse:
        answer, grounding = self.split_chat_response(response_text, source_ids)
        self.messages.save_message(
            dataset_id=session_id,
            role="assistant",
            content=f"**Answer:** {answer}\n\n**Grounding:** {grounding}",
            sources=source_ids,
        )
        return ChatResponse(
            answer=answer,
            grounding=grounding,
            agentMetadata=self.chat_model_metadata(),
        )

    @staticmethod
    def split_chat_response(
        response_text: str,
        source_ids: list[str],
    ) -> tuple[str, str]:
        match = re.search(
            r"(?:^|\n)\s*\*\*Grounding:\*\*\s*([\s\S]*)$",
            response_text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            answer = response_text[: match.start()].strip()
            grounding = match.group(1).strip()
        else:
            answer = response_text.strip()
            source_text = ", ".join(str(source_id) for source_id in source_ids)
            grounding = (
                f"Retrieved dataset sources: {source_text}."
                if source_text
                else "No supporting dataset evidence was available."
            )
            if source_ids:
                grounding = "Retrieved dataset sources: " + ", ".join(
                    chr(96) + str(source_id) + chr(96) for source_id in source_ids
                ) + "."
        answer = re.sub(
            r"^\s*\*\*Answer:\*\*\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return (
            answer or "The analysis assistant could not answer this question.",
            grounding or "No supporting dataset evidence was available.",
        )

    def get_chat_history(self, session_id: str, user_id: str) -> dict[str, Any]:
        session, _ = self.workspaces.load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat history is available."
            )
        return {
            "sessionId": session.id,
            "messages": [
                self.chat_message(message)
                for message in self.messages.get_recent_messages(session.id, limit=50)
            ],
        }

    def chat_message(
        self,
        message: MessageRecord,
        grounded: bool | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "grounded": (
                bool(grounded)
                if grounded is not None
                else message.role == "assistant"
            ),
            "createdAt": message.created_at,
        }
        if message.role == "assistant":
            payload["agentMetadata"] = self.chat_model_metadata()
        return payload

    def chat_model_metadata(self) -> dict[str, str]:
        return chat_model_usage(self.settings.bi_pipeline_mode)
