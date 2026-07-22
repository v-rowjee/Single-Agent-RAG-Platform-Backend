import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AnalysisState(TypedDict, total=False):
    session_id: str
    dataset_id: str
    business_description: str | None
    source_datasets: list[dict[str, Any]]

    uploaded_file_path: str
    working_directory: str
    generic_cleaned_file_path: str
    prepared_file_path: str

    generic_cleaning_report: dict[str, Any]
    prepared_dataset: dict[str, Any]

    orchestration_plan: dict[str, Any]

    kpi_trend_output: dict[str, Any]
    anomaly_output: dict[str, Any]
    forecasting_output: dict[str, Any]

    synthesis_output: dict[str, Any]
    dashboard_output: dict[str, Any]
    retrieval_documents: list[dict[str, Any]]

    workflow_status: str

    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    completed_agents: Annotated[list[str], operator.add]
    failed_agents: Annotated[list[str], operator.add]
    skipped_agents: Annotated[list[str], operator.add]


class ChatState(TypedDict, total=False):
    """State exchanged by the guarded retrieval and chat workflow."""

    session_id: str
    query: str
    history: list[dict[str, Any]]
    retrieval_query: str
    retrieved_documents: list[Any]
    reranked_documents: list[Any]
    draft: Any
    blocked: bool
