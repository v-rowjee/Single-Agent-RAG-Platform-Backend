"""Supabase chat-message repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.services.persistence.supabase import SupabaseGateway, SupabaseUnavailableError, supabase_gateway

JsonDict = dict[str, object]
MessageRole = Literal["user", "assistant"]

@dataclass(frozen=True)
class MessageRecord:
    id: str
    dataset_id: str
    role: str
    content: str
    sources: list[str]
    created_at: str

class MessageRepository:
    def __init__(self, gateway: SupabaseGateway | None = None) -> None:
        self.gateway = gateway or supabase_gateway

    @property
    def client(self):
        return self.gateway.client

    def save_message(
            self,
            dataset_id: str,
            role: MessageRole,
            content: str,
            sources: list[str] | None = None,
        ) -> MessageRecord:
            payload: JsonDict = {
                "session_id": dataset_id,
                "role": role,
                "content": content,
                "sources": sources or [],
            }
            response = self.client.table("messages").insert(payload).execute()
            rows = list(response.data or [])
            if not rows:
                raise SupabaseUnavailableError("Message insert returned no row.")
            return self._message(rows[0])

    def get_recent_messages(
            self,
            dataset_id: str,
            limit: int = 6,
        ) -> list[MessageRecord]:
            response = (
                self.client.table("messages")
                .select("*")
                .eq("session_id", dataset_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows = list(response.data or [])
            return [self._message(row) for row in reversed(rows)]

    @staticmethod
    def _message(row: JsonDict) -> MessageRecord:
            sources = row.get("sources")
            return MessageRecord(
                id=str(row["id"]),
                dataset_id=str(row.get("session_id") or row.get("dataset_id")),
                role=str(row["role"]),
                content=str(row["content"]),
                sources=[
                    str(source)
                    for source in sources
                    if isinstance(source, str | int | float)
                ]
                if isinstance(sources, list)
                else [],
                created_at=str(row.get("created_at") or MessageRepository._now()),
            )

    @staticmethod
    def _now() -> str:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

message_repository = MessageRepository()
