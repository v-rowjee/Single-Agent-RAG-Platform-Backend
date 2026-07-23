"""Pure and near-pure dashboard assembly operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings
from app.core.model_policy import (
    ModelUsage,
    chat_model_usage,
    multi_dashboard_model_usage,
    single_dashboard_model_usage,
)
from app.schemas.api import ApiMessage, DashboardResponse
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import DatasetInspection, PipelineExecution
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord


class DashboardAssembler:
    def __init__(self, *, settings: Settings, files: DatasetFileService) -> None:
        self.settings = settings
        self.files = files

    def with_model_metadata(
        self,
        response: DashboardResponse,
        selected_agents: list[str] | tuple[str, ...] = (),
        model_invocations: list[ModelUsage] | tuple[ModelUsage, ...] = (),
    ) -> DashboardResponse:
        payload = response.model_dump(mode="json")
        if not response.agentModels and self.settings.bi_pipeline_mode == "single":
            usage = single_dashboard_model_usage()
            payload.update(
                {
                    "pipelineMode": "single",
                    "model": usage[0]["model"],
                    "agentModels": usage,
                }
            )
        elif not response.agentModels:
            payload.update(
                {
                    "pipelineMode": "multi",
                    "agentModels": multi_dashboard_model_usage(
                        selected_agents,
                        model_invocations,
                    ),
                }
            )
        if response.chatAgent is None:
            payload["chatAgent"] = chat_model_usage(self.settings.bi_pipeline_mode)
        return DashboardResponse.model_validate(payload)

    def combine_workspace_executions(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        executions: list[PipelineExecution],
        contents: list[bytes] | None = None,
    ) -> PipelineExecution:
        if len(executions) == 1:
            execution = executions[0]
            if (
                execution.response.status != "failed"
                and execution.response.dashboard is not None
            ):
                return self.with_workspace_dataset_summaries(
                    session,
                    datasets,
                    contents or [],
                    execution,
                )
            return PipelineExecution(
                response=self.with_model_metadata(execution.response),
                workflow=execution.workflow,
                retrieval_documents=execution.retrieval_documents or [],
            )
        if any(item.response.status == "failed" for item in executions):
            return PipelineExecution(
                response=self.with_model_metadata(
                    DashboardResponse(
                        status="failed",
                        sessionId=session.id,
                        dashboard=None,
                        warnings=[
                            warning
                            for item in executions
                            for warning in item.response.warnings
                        ],
                        errors=[
                            *[
                                error
                                for item in executions
                                for error in item.response.errors
                            ],
                            ApiMessage(
                                code="DATASET_PREPARATION_FAILED",
                                message=(
                                    "At least one dataset could not be prepared, so "
                                    "the workspace analysis was stopped."
                                ),
                                component="data_preparation",
                                recoverable=True,
                            ),
                        ],
                    )
                ),
                workflow={"session_id": session.id, "workflow_status": "failed"},
                retrieval_documents=[],
            )
        raise ValueError("Multiple independent executions are no longer produced.")

    def with_workspace_dataset_summaries(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
        execution: PipelineExecution,
    ) -> PipelineExecution:
        payload = execution.response.model_dump(mode="json")
        dashboard = dict(payload["dashboard"] or {})
        summaries: list[dict[str, Any]] = []
        for index, dataset in enumerate(datasets):
            inspection = (
                self.files.inspect_file(dataset.file_name, contents[index])
                if index < len(contents)
                else DatasetInspection(
                    row_count=dataset.row_count or 0,
                    column_count=dataset.column_count or 0,
                    measures=[],
                    dimensions=[],
                    completeness_percent=100.0,
                    missing_value_count=0,
                    duplicate_row_count=0,
                )
            )
            summaries.append(
                {
                    "datasetId": dataset.id,
                    "fileName": dataset.file_name,
                    "rowCount": inspection.row_count,
                    "columnCount": inspection.column_count,
                    "timeField": None,
                    "period": None,
                    "measures": list(inspection.measures),
                    "dimensions": list(inspection.dimensions),
                    "quality": {
                        "completenessPercent": inspection.completeness_percent,
                        "missingValueCount": inspection.missing_value_count,
                        "duplicateRowCount": inspection.duplicate_row_count,
                    },
                    "generatedAt": self.current_timestamp(),
                }
            )
        dashboard["datasetSummaries"] = summaries
        if len(datasets) > 1:
            dashboard["title"] = "Multi-Dataset Business Intelligence Dashboard"
        payload["sessionId"] = session.id
        payload["dashboard"] = dashboard
        workflow = dict(execution.workflow or {})
        workflow.update(
            {
                "session_id": session.id,
                "dataset_ids": [item.id for item in datasets],
                "source_datasets": [
                    {"dataset_id": item.id, "file_name": item.file_name}
                    for item in datasets
                ],
            }
        )
        return PipelineExecution(
            response=DashboardResponse.model_validate(payload),
            workflow=workflow,
            retrieval_documents=self.workspace_retrieval_documents(
                session,
                datasets,
                execution.retrieval_documents or [],
            ),
        )

    @staticmethod
    def workspace_retrieval_documents(
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        retrieval_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        primary_id = datasets[0].id
        output: list[dict[str, Any]] = []
        for document in retrieval_documents:
            item = dict(document)
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "session_id": session.id,
                    "dataset_id": primary_id,
                    "file_name": "all_uploaded_datasets.csv",
                    "dataset_ids": [dataset.id for dataset in datasets],
                    "file_names": [dataset.file_name for dataset in datasets],
                }
            )
            item["metadata"] = metadata
            output.append(item)
        return output

    def build_placeholder_dashboard(
        self,
        dataset: DatasetRecord,
        inspection: DatasetInspection,
    ) -> dict[str, Any]:
        generated_at = self.current_timestamp()
        rows = inspection.row_count
        columns = inspection.column_count
        completeness = inspection.completeness_percent
        file_size_kb = round(dataset.file_size / 1024, 1)
        return {
            "status": "partial",
            "sessionId": dataset.id,
            "dashboard": {
                "title": f"Business Intelligence Dashboard - {dataset.file_name}",
                "executiveSummary": (
                    f"The uploaded dataset contains {rows:,} rows and "
                    f"{columns:,} columns. The AI analysis pipeline was "
                    "unavailable, so this dashboard shows basic dataset metrics."
                ),
                "kpis": [
                    self.kpi("dataset_rows", "Dataset Rows", rows),
                    self.kpi("dataset_columns", "Dataset Columns", columns),
                    self.kpi("data_completeness", "Data Completeness", completeness, "%"),
                    self.kpi("file_size", "File Size", file_size_kb, " KB"),
                ],
                "timeline": None,
                "supportingCharts": [
                    {
                        "id": "dataset_structure",
                        "type": "bar",
                        "title": "Dataset Structure",
                        "subtitle": "Rows and columns detected during upload",
                        "valueFormat": "number",
                        "categories": ["Rows", "Columns"],
                        "series": [
                            {
                                "id": "dataset_structure_values",
                                "name": "Count",
                                "data": [rows, columns],
                            }
                        ],
                        "layout": {"columnSpan": 1, "rowSpan": 1},
                    },
                    {
                        "id": "data_quality",
                        "type": "donut",
                        "title": "Data Completeness",
                        "subtitle": "Complete and missing values",
                        "valueFormat": "percentage",
                        "segments": [
                            {
                                "id": "complete_values",
                                "label": "Complete",
                                "value": completeness,
                            },
                            {
                                "id": "missing_values",
                                "label": "Missing",
                                "value": round(100 - completeness, 2),
                            },
                        ],
                        "layout": {"columnSpan": 1, "rowSpan": 1},
                    },
                ],
                "analysis": {
                    "businessSummary": (
                        "The file was uploaded successfully and basic structure "
                        "checks were completed."
                    ),
                    "keyFindings": [
                        f"The uploaded file is named {dataset.file_name}.",
                        f"The dataset contains {rows:,} rows.",
                        f"The dataset contains {columns:,} columns.",
                        f"{inspection.missing_value_count:,} missing values were detected.",
                    ],
                },
                "insights": {
                    "criticalAnomalies": [],
                    "warnings": [],
                    "limitations": [
                        {
                            "id": "ai_pipeline_unavailable",
                            "title": "AI pipeline unavailable",
                            "description": (
                                "Advanced KPIs, trends, forecasts, anomalies and "
                                "recommendations could not be generated by the AI agent."
                            ),
                            "severity": "info",
                            "sourceIds": [],
                        }
                    ],
                    "opportunities": [],
                },
                "recommendedActions": [
                    {
                        "id": "retry_ai_analysis",
                        "title": "Retry AI analysis",
                        "description": (
                            "Retry the business intelligence agent when the AI "
                            "pipeline is available."
                        ),
                        "priority": "medium",
                        "sourceIds": [],
                    },
                    {
                        "id": "review_dataset_quality",
                        "title": "Review dataset quality",
                        "description": (
                            "Resolve the detected missing values and duplicate records "
                            "before rerunning advanced analysis."
                        ),
                        "priority": "medium",
                        "sourceIds": ["dataset_summary"],
                    },
                    {
                        "id": "confirm_business_context",
                        "title": "Confirm business context",
                        "description": (
                            "Confirm the intended KPI definitions, reporting period, "
                            "and business priorities before the next analysis run."
                        ),
                        "priority": "low",
                        "sourceIds": ["dataset_summary"],
                    },
                ],
                "datasetSummary": {
                    "fileName": dataset.file_name,
                    "rowCount": rows,
                    "columnCount": columns,
                    "timeField": None,
                    "period": None,
                    "measures": inspection.measures,
                    "dimensions": inspection.dimensions,
                    "quality": {
                        "completenessPercent": completeness,
                        "missingValueCount": inspection.missing_value_count,
                        "duplicateRowCount": inspection.duplicate_row_count,
                    },
                    "generatedAt": generated_at,
                },
                "sections": [
                    {"id": "kpis", "title": "Key Performance Indicators", "order": 1, "visible": True},
                    {"id": "timeline", "title": "Performance Over Time", "order": 2, "visible": False},
                    {"id": "supportingCharts", "title": "Supporting Analysis", "order": 3, "visible": True},
                    {"id": "details", "title": "Insights and Recommendations", "order": 4, "visible": True},
                ],
                "layout": {
                    "kpis": {"columns": 4, "maxRows": 2},
                    "timeline": {"columnSpan": 12},
                    "supportingCharts": {"columns": 2, "maxRows": 2},
                    "details": {"columns": 2, "maxRows": 2},
                },
            },
            "warnings": [
                {
                    "code": "AI_PIPELINE_UNAVAILABLE",
                    "message": (
                        "The AI agent could not generate the dashboard, so this "
                        "fallback contains basic dataset information only."
                    ),
                    "component": "business_intelligence_agent",
                    "recoverable": True,
                }
            ],
            "errors": [],
        }

    @staticmethod
    def kpi(
        item_id: str,
        title: str,
        value: int | float,
        suffix: str = "",
    ) -> dict[str, Any]:
        display = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        return {
            "id": item_id,
            "title": title,
            "value": f"{display}{suffix}",
            "rawValue": value,
            "indicator": {"kind": "note", "text": "Detected during upload"},
        }

    @staticmethod
    def current_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
