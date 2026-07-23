"""Upload validation, persistence, and compensating rollback."""

from __future__ import annotations

import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
)
from app.schemas.api import UploadCandidate
from app.services.analysis.files import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    DatasetFileService,
)
from app.services.analysis.models import InspectedUpload, UploadedWorkspace
from app.services.persistence.analysis import (
    AnalysisRepository,
    AnalysisSessionRecord,
    DatasetRecord,
)
from app.services.persistence.supabase import SupabaseGateway

logger = logging.getLogger(__name__)


class DatasetUploadService:
    def __init__(
        self,
        *,
        analysis: AnalysisRepository | Any,
        storage: SupabaseGateway | Any,
        files: DatasetFileService,
    ) -> None:
        self.analysis = analysis
        self.storage = storage
        self.files = files

    async def prepare_uploads(
        self,
        files: list[UploadCandidate] | UploadCandidate | Any,
        *,
        existing_names: set[str] | None = None,
        existing_hashes: set[str] | None = None,
        maximum_files: int,
    ) -> list[InspectedUpload]:
        candidates = await self.files.upload_candidates(files)
        if not 1 <= len(candidates) <= maximum_files:
            noun = "file" if maximum_files == 1 else "files"
            qualifier = " more" if existing_names or existing_hashes else ""
            raise InvalidUploadError(
                f"Upload between 1 and {maximum_files}{qualifier} {noun}."
            )

        names = set(existing_names or ())
        hashes = set(existing_hashes or ())
        had_existing_names = bool(existing_names)
        had_existing_hashes = bool(existing_hashes)
        inspected: list[InspectedUpload] = []
        for candidate in candidates:
            original_name = Path(candidate.file_name or "").name
            file_name = self.files.sanitize_file_name(original_name)
            mime_type = (candidate.content_type or "").strip()
            self.files.validate_upload_metadata(file_name, mime_type)
            normalized_name = file_name.casefold()
            if normalized_name in names:
                if had_existing_names:
                    raise InvalidUploadError(
                        f"A dataset named '{file_name}' already exists in this workspace."
                    )
                raise InvalidUploadError(
                    f"Duplicate filename '{file_name}' is not allowed."
                )
            content = candidate.content
            if not content:
                raise InvalidUploadError(f"'{file_name}' is empty.")
            if len(content) > MAX_UPLOAD_BYTES:
                raise InvalidUploadError(
                    f"'{file_name}' is larger than the 25 MiB limit."
                )
            file_hash = hashlib.sha256(content).hexdigest()
            if file_hash in hashes:
                if had_existing_hashes:
                    raise InvalidUploadError(
                        f"'{file_name}' duplicates an existing workspace dataset."
                    )
                raise InvalidUploadError(
                    f"'{file_name}' duplicates another file in this batch."
                )
            item = InspectedUpload(
                file_name=file_name,
                mime_type=mime_type,
                content=content,
                file_hash=file_hash,
                inspection=self.files.inspect_file(file_name, content),
            )
            inspected.append(item)
            names.add(normalized_name)
            hashes.add(file_hash)
        return inspected

    async def create_workspace(
        self,
        *,
        files: list[UploadCandidate] | UploadCandidate | Any,
        user_id: str,
        description: str | None,
    ) -> UploadedWorkspace:
        if not user_id:
            raise ValueError("An authenticated user is required.")
        inspected = await self.prepare_uploads(
            files,
            maximum_files=MAX_UPLOAD_FILES,
        )
        get_active_session = getattr(self.analysis, "get_active_session", None)
        active = (
            get_active_session(user_id)
            if callable(get_active_session)
            else self.analysis.get_active_dataset(user_id)
        )
        if active is not None:
            raise DatasetAlreadyExistsError(
                "You already have an active workspace. Use Start Over before uploading another batch."
            )

        session_id = str(uuid4())
        datasets: list[DatasetRecord] = []
        uploaded_paths: list[str] = []
        session_created = False
        try:
            create_session = getattr(self.analysis, "create_session", None)
            if callable(create_session):
                session = create_session(
                    session_id=session_id,
                    user_id=user_id,
                    description=description,
                )
                session_created = True
            else:
                session = AnalysisSessionRecord(
                    id=session_id,
                    user_id=user_id,
                    description=description,
                    status="processing",
                    rag_status="pending",
                    error_message=None,
                )
            for index, item in enumerate(inspected):
                dataset_id = session_id if not session_created and index == 0 else str(uuid4())
                storage_path = (
                    f"{user_id}/{session_id}/{item.file_name}"
                    if not session_created and len(inspected) == 1
                    else f"{user_id}/{session_id}/{dataset_id}/{item.file_name}"
                )
                dataset = self._create_dataset(
                    session_id=session_id,
                    dataset_id=dataset_id,
                    user_id=user_id,
                    description=description,
                    storage_path=storage_path,
                    item=item,
                    legacy=not session_created,
                )
                datasets.append(dataset)
                self.storage.upload_file(
                    storage_path=storage_path,
                    content=item.content,
                    mime_type=item.mime_type,
                )
                uploaded_paths.append(storage_path)
            return UploadedWorkspace(
                session=session,
                datasets=datasets,
                contents=[item.content for item in inspected],
                uploaded_paths=uploaded_paths,
                session_created=session_created,
            )
        except Exception as error:
            self.rollback_new_workspace(
                session_id=session_id,
                user_id=user_id,
                datasets=datasets,
                uploaded_paths=uploaded_paths,
                session_created=session_created,
            )
            if not datasets and self.is_duplicate_dataset_error(error):
                raise DatasetAlreadyExistsError(
                    "You already have an active workspace. Use Start Over before uploading another batch."
                ) from error
            raise

    async def add_to_workspace(
        self,
        *,
        files: list[UploadCandidate] | UploadCandidate | Any,
        session: AnalysisSessionRecord,
        existing_datasets: list[DatasetRecord],
    ) -> UploadedWorkspace:
        available = MAX_UPLOAD_FILES - len(existing_datasets)
        if available <= 0:
            raise InvalidUploadError(
                f"A workspace can contain up to {MAX_UPLOAD_FILES} datasets."
            )
        inspected = await self.prepare_uploads(
            files,
            existing_names={item.file_name.casefold() for item in existing_datasets},
            existing_hashes={item.file_hash for item in existing_datasets},
            maximum_files=available,
        )
        datasets: list[DatasetRecord] = []
        uploaded_paths: list[str] = []
        try:
            for item in inspected:
                dataset_id = str(uuid4())
                storage_path = (
                    f"{session.user_id}/{session.id}/{dataset_id}/{item.file_name}"
                )
                dataset = self._create_dataset(
                    session_id=session.id,
                    dataset_id=dataset_id,
                    user_id=session.user_id,
                    description=session.description,
                    storage_path=storage_path,
                    item=item,
                    legacy=False,
                )
                datasets.append(dataset)
                self.storage.upload_file(
                    storage_path=storage_path,
                    content=item.content,
                    mime_type=item.mime_type,
                )
                uploaded_paths.append(storage_path)
        except Exception as error:
            self.rollback_added_datasets(
                user_id=session.user_id,
                datasets=datasets,
                uploaded_paths=uploaded_paths,
            )
            if self.is_duplicate_dataset_error(error):
                raise InvalidUploadError(
                    "One of these datasets already exists in the workspace."
                ) from error
            raise
        return UploadedWorkspace(
            session=session,
            datasets=datasets,
            contents=[item.content for item in inspected],
            uploaded_paths=uploaded_paths,
            session_created=False,
        )

    def rollback_new_workspace(
        self,
        *,
        session_id: str,
        user_id: str,
        datasets: list[DatasetRecord],
        uploaded_paths: list[str],
        session_created: bool,
    ) -> None:
        self._delete_uploaded_paths(uploaded_paths)
        if session_created:
            self.analysis.delete_session(session_id, user_id)
        else:
            for dataset in datasets:
                self.analysis.delete_dataset(dataset.id, user_id)

    def rollback_added_datasets(
        self,
        *,
        user_id: str,
        datasets: list[DatasetRecord],
        uploaded_paths: list[str],
    ) -> None:
        self._delete_uploaded_paths(uploaded_paths)
        for dataset in datasets:
            self.analysis.delete_dataset(dataset.id, user_id)

    def _delete_uploaded_paths(self, uploaded_paths: list[str]) -> None:
        for storage_path in reversed(uploaded_paths):
            try:
                self.storage.delete_file(storage_path)
            except Exception:
                logger.exception(
                    "Unable to roll back uploaded object path=%s",
                    storage_path,
                )

    def _create_dataset(
        self,
        *,
        session_id: str,
        dataset_id: str,
        user_id: str,
        description: str | None,
        storage_path: str,
        item: InspectedUpload,
        legacy: bool,
    ) -> DatasetRecord:
        values = {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "user_id": user_id,
            "file_name": item.file_name,
            "storage_path": storage_path,
            "mime_type": item.mime_type,
            "file_size": len(item.content),
            "file_hash": item.file_hash,
            "description": description,
            "row_count": item.inspection.row_count,
            "column_count": item.inspection.column_count,
        }
        if legacy:
            parameters = inspect.signature(
                self.analysis.create_dataset
            ).parameters.values()
            accepts_session_id = any(
                parameter.name == "session_id"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if not accepts_session_id:
                values.pop("session_id")
                dataset = self.analysis.create_dataset(**values)
                return DatasetRecord(
                    **{**dataset.__dict__, "session_id": session_id}
                )
        return self.analysis.create_dataset(**values)

    @staticmethod
    def is_duplicate_dataset_error(error: Exception) -> bool:
        code = str(getattr(error, "code", ""))
        message = str(error).lower()
        return code == "23505" or (
            "duplicate" in message and "user" in message and "dataset" in message
        )
