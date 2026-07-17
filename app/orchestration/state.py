import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class BusinessIntelligenceState(TypedDict, total=False):
    session_id: str
    dataset_id: str
    business_description: str | None

    uploaded_file_path: str
    working_directory: str
    generic_cleaned_file_path: str
    prepared_file_path: str
    prepared_temporal_file_path: str | None

    generic_cleaning_report: dict[str, Any]
    prepared_dataset: dict[str, Any]

    orchestration_plan: dict[str, Any]

    kpi_trend_output: dict[str, Any]
    anomaly_output: dict[str, Any]
    forecasting_output: dict[str, Any]

    synthesis_output: dict[str, Any]
    dashboard_output: dict[str, Any]
    dashboard_layout_plan: dict[str, Any]
    retrieval_output: dict[str, Any]
    retrieval_documents: list[dict[str, Any]]
    retrieval_indexing_result: dict[str, Any]
    persistence_result: dict[str, Any]

    workflow_status: str

    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    completed_agents: Annotated[list[str], operator.add]
    failed_agents: Annotated[list[str], operator.add]
    skipped_agents: Annotated[list[str], operator.add]
