from __future__ import annotations

from app.services.analysis.chat import BusinessIntelligenceChatService
from app.services.persistence.analysis import DatasetRecord


def _dataset(identifier: str, name: str) -> DatasetRecord:
    return DatasetRecord(
        identifier, "user", name, name, "text/csv", 10, identifier,
        None, "ready", "ready", None, session_id="session",
    )


def test_chat_helpers_select_named_dataset_and_split_grounding() -> None:
    selected, ambiguous = BusinessIntelligenceChatService.select_chat_datasets(
        "Summarise inventory.csv",
        [_dataset("one", "sales.csv"), _dataset("two", "inventory.csv")],
    )
    answer, grounding = BusinessIntelligenceChatService.split_chat_response(
        "**Answer:** Stock is 8.\n\n**Grounding:** Inventory summary.",
        ["inventory_summary"],
    )

    assert selected is not None and selected.id == "two"
    assert ambiguous == []
    assert answer == "Stock is 8."
    assert grounding == "Inventory summary."
