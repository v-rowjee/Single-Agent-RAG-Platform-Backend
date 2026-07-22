"""Dataset parsing, inspection, preview, and temporary-file operations."""

from __future__ import annotations

import io
import math
import re
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.core.exceptions import InvalidUploadError
from app.schemas.api import BusinessIntelligenceAgentInput, UploadCandidate
from app.services.analysis.models import DatasetInspection
from app.services.persistence.analysis import DatasetRecord

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_FILES = 5
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class DatasetFileService:
    async def upload_candidates(
        self,
        files: list[UploadCandidate] | UploadCandidate | Any,
    ) -> list[UploadCandidate]:
        raw_files = list(files) if isinstance(files, (list, tuple)) else [files]
        candidates: list[UploadCandidate] = []
        for item in raw_files:
            if isinstance(item, UploadCandidate):
                candidates.append(item)
                continue
            read = getattr(item, "read", None)
            if not callable(read):
                raise InvalidUploadError("An uploaded file could not be read.")
            candidates.append(
                UploadCandidate(
                    file_name=str(getattr(item, "filename", "") or ""),
                    content_type=str(getattr(item, "content_type", "") or ""),
                    content=await read(),
                )
            )
        return candidates

    def validate_upload_metadata(
        self,
        file_name: str,
        mime_type: str,
        extension: str | None = None,
    ) -> None:
        extension = extension or Path(file_name).suffix.lower()
        if not file_name or file_name in {".csv", ".xlsx"}:
            raise InvalidUploadError("The uploaded file must have a valid name.")
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidUploadError("Only CSV and XLSX files are supported.")
        if mime_type and mime_type not in ALLOWED_MIME_TYPES:
            raise InvalidUploadError("The uploaded file type is not supported.")

    @staticmethod
    def sanitize_file_name(file_name: str) -> str:
        safe_name = Path(file_name or "uploaded-file").name.strip()
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix.lower()
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
        if not stem:
            stem = "uploaded-file"
        return f"{stem[:120]}{suffix}"

    def read_dataframe(
        self,
        file_name: str,
        content: bytes,
        *,
        parse_error: str = "The uploaded dataset could not be parsed.",
    ) -> pd.DataFrame:
        suffix = Path(file_name).suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(io.BytesIO(content), low_memory=False)
            if suffix == ".xlsx":
                return pd.read_excel(io.BytesIO(content))
            raise InvalidUploadError("Only CSV and XLSX files are supported.")
        except UnicodeDecodeError as error:
            raise InvalidUploadError("The CSV file must use UTF-8 encoding.") from error
        except InvalidUploadError:
            raise
        except Exception as error:
            raise InvalidUploadError(parse_error) from error

    def inspect_file(self, file_name: str, content: bytes) -> DatasetInspection:
        frame = self.read_dataframe(
            file_name,
            content,
            parse_error="The uploaded file could not be parsed.",
        )
        row_count = int(len(frame))
        column_count = int(len(frame.columns))
        missing = int(frame.isna().sum().sum())
        duplicates = int(frame.duplicated().sum())
        total_cells = row_count * column_count
        completeness = (
            round(((total_cells - missing) / total_cells) * 100, 2)
            if total_cells
            else 100.0
        )
        measures = [
            str(column)
            for column in frame.select_dtypes(include="number").columns
        ]
        return DatasetInspection(
            row_count=row_count,
            column_count=column_count,
            measures=measures,
            dimensions=[
                str(column) for column in frame.columns if str(column) not in measures
            ],
            missing_value_count=missing,
            duplicate_row_count=duplicates,
            completeness_percent=completeness,
        )

    def read_workspace_dataframe(
        self,
        file_name: str,
        content: bytes,
    ) -> pd.DataFrame:
        frame = self.read_dataframe(file_name, content)
        counts: dict[str, int] = {}
        columns: list[str] = []
        for index, column in enumerate(frame.columns, start=1):
            base = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(column).strip().lower(),
            ).strip("_")
            base = base or f"unnamed_{index}"
            occurrence = counts.get(base, 0)
            columns.append(base if occurrence == 0 else f"{base}_{occurrence + 1}")
            counts[base] = occurrence + 1
        frame.columns = columns
        return frame

    def read_preview_page(
        self,
        file_name: str,
        content: bytes,
        page: int,
        page_size: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        start_row = (page - 1) * page_size
        suffix = Path(file_name).suffix.lower()
        try:
            if suffix == ".csv":
                skiprows = (
                    (lambda row_index: row_index != 0 and row_index <= start_row)
                    if start_row
                    else None
                )
                frame = pd.read_csv(
                    io.BytesIO(content),
                    low_memory=False,
                    skiprows=skiprows,
                    nrows=page_size,
                )
            elif suffix == ".xlsx":
                frame = pd.read_excel(
                    io.BytesIO(content),
                    skiprows=range(1, start_row + 1) if start_row else None,
                    nrows=page_size,
                )
            else:
                raise InvalidUploadError("Only CSV and XLSX files are supported.")
        except UnicodeDecodeError as error:
            raise InvalidUploadError("The CSV file must use UTF-8 encoding.") from error
        except InvalidUploadError:
            raise
        except Exception as error:
            raise InvalidUploadError(
                "The uploaded dataset preview could not be read."
            ) from error

        columns = self.unique_column_names([str(column) for column in frame.columns])
        frame.columns = columns
        return columns, [
            {column: self.json_preview_value(value) for column, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]

    @staticmethod
    def unique_column_names(columns: list[str]) -> list[str]:
        used: dict[str, int] = {}
        result: list[str] = []
        for index, raw_name in enumerate(columns, start=1):
            base_name = raw_name.strip() or f"Column {index}"
            occurrence = used.get(base_name, 0)
            used[base_name] = occurrence + 1
            result.append(
                base_name if occurrence == 0 else f"{base_name} ({occurrence + 1})"
            )
        return result

    @staticmethod
    def json_preview_value(value: Any) -> str | int | float | bool | None:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)):
            return value
        return str(value)

    @contextmanager
    def temporary_agent_input(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> Iterator[BusinessIntelligenceAgentInput]:
        with self.temporary_agent_workspace(dataset, content) as (agent_input, _):
            yield agent_input

    @contextmanager
    def temporary_agent_workspace(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> Iterator[tuple[BusinessIntelligenceAgentInput, Path]]:
        suffix = Path(dataset.file_name).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="bi_dataset_") as directory:
            root = Path(directory)
            path = root / dataset.file_name
            if path.suffix.lower() != suffix:
                path = root / f"dataset{suffix}"
            path.write_bytes(content)
            workspace = root / "processing"
            workspace.mkdir()
            yield (
                BusinessIntelligenceAgentInput(
                    sessionId=dataset.id,
                    datasetId=dataset.id,
                    filePath=str(path),
                    fileName=dataset.file_name,
                    description=dataset.description,
                ),
                workspace,
            )
