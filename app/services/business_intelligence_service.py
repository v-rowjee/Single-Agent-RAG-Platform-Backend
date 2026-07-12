from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile


logger = logging.getLogger(__name__)

STORAGE_ROOT = Path("app/storage")
UPLOADS_DIR = STORAGE_ROOT / "uploads"
SESSIONS_DIR = STORAGE_ROOT / "sessions"
RESULTS_DIR = STORAGE_ROOT / "results"

for directory in (UPLOADS_DIR, SESSIONS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class SessionNotFoundError(Exception):
    """Raised when an analysis session does not exist."""


class InvalidUploadError(Exception):
    """Raised when an uploaded file cannot be processed."""


class BusinessIntelligenceService:
    async def create_analysis(
        self,
        file: UploadFile,
        description: str | None = None,
    ) -> dict[str, Any]:
        file_name = Path(file.filename or "uploaded-file").name
        content = await file.read()

        if not content:
            raise InvalidUploadError("The uploaded file is empty.")

        session_id = str(uuid4())
        created_at = self._current_timestamp()

        file_extension = Path(file_name).suffix.lower()
        upload_path = UPLOADS_DIR / f"{session_id}{file_extension}"

        try:
            upload_path.write_bytes(content)

            dataset_info = self._inspect_file(
                file_name=file_name,
                content=content,
            )

            session = {
                "sessionId": session_id,
                "fileName": file_name,
                "contentType": file.content_type,
                "description": description,
                "fileSize": len(content),
                "uploadPath": str(upload_path),
                "createdAt": created_at,
            }

            dashboard_response = self._build_dashboard_with_agent(
                session=session,
                dataset_info=dataset_info,
            )

            self._write_json(
                SESSIONS_DIR / f"{session_id}.json",
                session,
            )

            self._write_json(
                RESULTS_DIR / f"{session_id}.json",
                dashboard_response,
            )

        except Exception:
            upload_path.unlink(missing_ok=True)
            (SESSIONS_DIR / f"{session_id}.json").unlink(missing_ok=True)
            (RESULTS_DIR / f"{session_id}.json").unlink(missing_ok=True)
            raise

        return {
            "status": "success",
            "sessionId": session_id,
            "fileName": file_name,
            "message": "File uploaded and analysis session created successfully.",
        }

    def get_dashboard(self, session_id: str) -> dict[str, Any]:
        self._load_session(session_id)

        result_path = RESULTS_DIR / f"{session_id}.json"

        if not result_path.exists():
            raise SessionNotFoundError(
                f"No dashboard result was found for session '{session_id}'."
            )

        return self._read_json(result_path)

    def chat(
        self,
        session_id: str,
        query: str,
    ) -> dict[str, str]:
        session = self._load_session(session_id)

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")

        response = self._chat_with_agent(
            session=session,
            query=cleaned_query,
        )

        return {"response": response}

    def _load_session(self, session_id: str) -> dict[str, Any]:
        session_path = SESSIONS_DIR / f"{session_id}.json"

        if not session_path.exists():
            raise SessionNotFoundError(
                f"Analysis session '{session_id}' was not found."
            )

        return self._read_json(session_path)

    def _inspect_file(
        self,
        file_name: str,
        content: bytes,
    ) -> dict[str, Any]:
        if not file_name.lower().endswith(".csv"):
            return {
                "rowCount": 0,
                "columnCount": 0,
                "measures": [],
                "dimensions": [],
                "missingValueCount": 0,
                "duplicateRowCount": 0,
                "completenessPercent": 100.0,
            }

        return self._inspect_csv(content)

    def _inspect_csv(self, content: bytes) -> dict[str, Any]:
        try:
            text = content.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text))

            headers = next(reader, [])

            if not headers:
                raise InvalidUploadError(
                    "The uploaded CSV does not contain a header row."
                )

            column_count = len(headers)
            row_count = 0
            missing_value_count = 0
            duplicate_row_count = 0

            seen_rows: set[tuple[str, ...]] = set()

            numeric_columns = [True] * column_count
            columns_with_values = [False] * column_count

            for row in reader:
                normalized_row = [
                    row[index].strip() if index < len(row) else ""
                    for index in range(column_count)
                ]

                if not any(normalized_row):
                    continue

                row_count += 1

                row_key = tuple(normalized_row)

                if row_key in seen_rows:
                    duplicate_row_count += 1
                else:
                    seen_rows.add(row_key)

                for index, value in enumerate(normalized_row):
                    if value == "":
                        missing_value_count += 1
                        continue

                    columns_with_values[index] = True

                    if not self._is_number(value):
                        numeric_columns[index] = False

            measures = [
                headers[index]
                for index in range(column_count)
                if numeric_columns[index] and columns_with_values[index]
            ]

            dimensions = [
                headers[index]
                for index in range(column_count)
                if headers[index] not in measures
            ]

            total_cells = row_count * column_count

            if total_cells == 0:
                completeness_percent = 100.0
            else:
                completeness_percent = round(
                    ((total_cells - missing_value_count) / total_cells) * 100,
                    2,
                )

            return {
                "rowCount": row_count,
                "columnCount": column_count,
                "measures": measures,
                "dimensions": dimensions,
                "missingValueCount": missing_value_count,
                "duplicateRowCount": duplicate_row_count,
                "completenessPercent": completeness_percent,
            }

        except UnicodeDecodeError as error:
            raise InvalidUploadError(
                "The CSV file must use UTF-8 encoding."
            ) from error
        except csv.Error as error:
            raise InvalidUploadError(
                "The uploaded CSV could not be parsed."
            ) from error

    def _build_dashboard_with_agent(
        self,
        session: dict[str, Any],
        dataset_info: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from app.agents.business_intelligence_agent import (
                business_intelligence_agent,
            )

            dashboard_response = business_intelligence_agent.generate_dashboard(
                self._build_agent_input(session)
            )

            return dashboard_response.model_dump(mode="json")

        except Exception:
            logger.exception(
                "Business intelligence agent failed while generating "
                "dashboard. Returning fallback dashboard."
            )

            dashboard_response = self._build_placeholder_dashboard(
                session=session,
                dataset_info=dataset_info,
            )

            return dashboard_response

    def _chat_with_agent(
        self,
        session: dict[str, Any],
        query: str,
    ) -> str:
        try:
            from app.agents.business_intelligence_agent import (
                business_intelligence_agent,
            )

            return business_intelligence_agent.chat(
                agent_input=self._build_agent_input(session),
                query=query,
            )

        except Exception:
            logger.exception(
                "Business intelligence agent failed while answering chat. "
                "Returning fallback response."
            )

            return (
                f'You asked: "{query}". '
                f"The active dataset is {session['fileName']}. "
                "The AI business intelligence agent is currently unavailable."
            )

    @staticmethod
    def _build_agent_input(session: dict[str, Any]) -> Any:
        from app.schemas.business_intelligence import (
            BusinessIntelligenceAgentInput,
        )

        return BusinessIntelligenceAgentInput(
            sessionId=session["sessionId"],
            filePath=session["uploadPath"],
            fileName=session["fileName"],
            description=session.get("description"),
        )

    def _build_placeholder_dashboard(
        self,
        session: dict[str, Any],
        dataset_info: dict[str, Any],
    ) -> dict[str, Any]:
        session_id = session["sessionId"]
        file_name = session["fileName"]

        row_count = dataset_info["rowCount"]
        column_count = dataset_info["columnCount"]
        file_size = session["fileSize"]

        return {
            "status": "partial",
            "sessionId": session_id,
            "dashboard": {
                "title": f"Business Intelligence Dashboard — {file_name}",
                "executiveSummary": (
                    f"The uploaded dataset contains {row_count:,} rows and "
                    f"{column_count:,} columns. The AI analysis pipeline was "
                    "unavailable, so the dashboard currently shows basic "
                    "dataset information."
                ),
                "kpis": [
                    {
                        "id": "dataset_rows",
                        "title": "Dataset Rows",
                        "value": f"{row_count:,}",
                        "rawValue": row_count,
                        "indicator": {
                            "kind": "note",
                            "text": "Rows detected during upload",
                        },
                    },
                    {
                        "id": "dataset_columns",
                        "title": "Dataset Columns",
                        "value": f"{column_count:,}",
                        "rawValue": column_count,
                        "indicator": {
                            "kind": "note",
                            "text": "Columns detected during upload",
                        },
                    },
                    {
                        "id": "dataset_completeness",
                        "title": "Data Completeness",
                        "value": (
                            f"{dataset_info['completenessPercent']:.2f}%"
                        ),
                        "rawValue": dataset_info["completenessPercent"],
                        "indicator": {
                            "kind": "note",
                            "text": "Based on non-empty CSV cells",
                        },
                    },
                    {
                        "id": "uploaded_file_size",
                        "title": "File Size",
                        "value": f"{file_size / 1024:.1f} KB",
                        "rawValue": file_size,
                        "indicator": {
                            "kind": "note",
                            "text": "Uploaded file size",
                        },
                    },
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
                                "data": [row_count, column_count],
                            }
                        ],
                        "layout": {
                            "columnSpan": 1,
                            "rowSpan": 1,
                        },
                    },
                    {
                        "id": "data_quality",
                        "type": "donut",
                        "title": "Data Completeness",
                        "subtitle": "Complete and missing CSV values",
                        "valueFormat": "percentage",
                        "segments": [
                            {
                                "id": "complete_values",
                                "label": "Complete",
                                "value": dataset_info[
                                    "completenessPercent"
                                ],
                            },
                            {
                                "id": "missing_values",
                                "label": "Missing",
                                "value": round(
                                    100
                                    - dataset_info[
                                        "completenessPercent"
                                    ],
                                    2,
                                ),
                            },
                        ],
                        "layout": {
                            "columnSpan": 1,
                            "rowSpan": 1,
                        },
                    },
                ],
                "analysis": {
                    "businessSummary": (
                        "The dataset was uploaded successfully and its basic "
                        "structure was inspected. Business analysis could "
                        "not be generated because the AI agent was "
                        "unavailable."
                    ),
                    "keyFindings": [
                        f"The uploaded file is named {file_name}.",
                        f"The dataset contains {row_count:,} rows.",
                        f"The dataset contains {column_count:,} columns.",
                        (
                            f"{dataset_info['missingValueCount']:,} missing "
                            "values were detected."
                        ),
                        (
                            f"{dataset_info['duplicateRowCount']:,} duplicate "
                            "rows were detected."
                        ),
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
                                "Business KPIs, trends, forecasts, anomalies "
                                "and recommendations could not be generated "
                                "by the AI agent."
                            ),
                            "severity": "info",
                            "sourceIds": [],
                        }
                    ],
                    "opportunities": [
                        {
                            "id": "retry_business_intelligence_agent",
                            "title": "Retry the analysis pipeline",
                            "description": (
                                "The stored dataset can now be passed to the "
                                "business intelligence agent again for full "
                                "analysis."
                            ),
                            "severity": "info",
                            "sourceIds": [],
                        }
                    ],
                },
                "recommendedActions": [
                    {
                        "id": "retry_ai_analysis",
                        "title": "Retry AI analysis",
                        "description": (
                            "Retry the business intelligence agent and replace "
                            "the fallback dataset metrics with calculated "
                            "business KPIs and insights."
                        ),
                        "priority": "medium",
                        "sourceIds": [],
                    }
                ],
                "datasetSummary": {
                    "fileName": file_name,
                    "rowCount": row_count,
                    "columnCount": column_count,
                    "timeField": None,
                    "period": None,
                    "measures": dataset_info["measures"],
                    "dimensions": dataset_info["dimensions"],
                    "quality": {
                        "completenessPercent": dataset_info[
                            "completenessPercent"
                        ],
                        "missingValueCount": dataset_info[
                            "missingValueCount"
                        ],
                        "duplicateRowCount": dataset_info[
                            "duplicateRowCount"
                        ],
                    },
                    "generatedAt": session["createdAt"],
                },
                "sections": [
                    {
                        "id": "kpis",
                        "title": "Key Performance Indicators",
                        "order": 1,
                        "visible": True,
                    },
                    {
                        "id": "timeline",
                        "title": "Performance Over Time",
                        "order": 2,
                        "visible": False,
                    },
                    {
                        "id": "supportingCharts",
                        "title": "Supporting Analysis",
                        "order": 3,
                        "visible": True,
                    },
                    {
                        "id": "details",
                        "title": "Insights and Recommendations",
                        "order": 4,
                        "visible": True,
                    },
                ],
                "layout": {
                    "kpis": {
                        "columns": 4,
                        "maxRows": 2,
                    },
                    "timeline": {
                        "columnSpan": 12,
                    },
                    "supportingCharts": {
                        "columns": 2,
                        "maxRows": 2,
                    },
                    "details": {
                        "columns": 3,
                        "maxRows": 2,
                    },
                },
            },
            "warnings": [
                {
                    "code": "AI_PIPELINE_UNAVAILABLE",
                    "message": (
                        "The AI agent could not generate the dashboard, so "
                        "this fallback contains basic dataset information only."
                    ),
                    "component": "business_intelligence_agent",
                    "recoverable": True,
                }
            ],
            "errors": [],
        }

    @staticmethod
    def _is_number(value: str) -> bool:
        cleaned_value = (
            value.replace(",", "")
            .replace("£", "")
            .replace("$", "")
            .replace("€", "")
            .replace("%", "")
            .strip()
        )

        try:
            float(cleaned_value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _current_timestamp() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _write_json(
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


business_intelligence_service = BusinessIntelligenceService()
