"""Generic-cleaning graph node."""

from pathlib import Path
from typing import Any

from app.orchestration.state import AnalysisState
from app.services.data.cleaning import _generic_clean_csv


async def generic_cleaning_node(
    state: AnalysisState,
) -> dict[str, Any]:
    """Adapt the existing generic cleaner for use as a LangGraph node."""
    uploaded_file_path = str(state.get("uploaded_file_path", "")).strip()
    working_directory = str(state.get("working_directory") or "").strip()
    if not working_directory:
        raise RuntimeError("state.working_directory is required.")

    _, report = _generic_clean_csv(
        uploaded_file_path,
        Path(working_directory),
    )

    return {
        "generic_cleaned_file_path": report.cleaned_file_path,
        "generic_cleaning_report": report.model_dump(mode="json"),
        "warnings": report.warnings,
        "completed_agents": ["generic_cleaning"],
    }
