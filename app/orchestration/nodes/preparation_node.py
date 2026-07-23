"""Data-preparation graph adapter."""

from typing import Any

from app.agents.multi.data_preparation import data_preparation_node
from app.orchestration.state import AnalysisState


async def data_preparation_graph_node(
    state: AnalysisState,
) -> dict[str, Any]:
    """Run the existing preparation node against the generic-cleaned dataset."""
    node_state = dict(state)
    node_state["file_name"] = node_state.get("file_name") or str(
        state.get("uploaded_file_path", "")
    ).replace("\\", "/").rsplit("/", 1)[-1]
    node_state["uploaded_file_path"] = state.get(
        "generic_cleaned_file_path",
        state.get("uploaded_file_path", ""),
    )
    update = await data_preparation_node(node_state)
    update.pop("generic_cleaned_file_path", None)
    update.pop("generic_cleaning_report", None)
    return update
