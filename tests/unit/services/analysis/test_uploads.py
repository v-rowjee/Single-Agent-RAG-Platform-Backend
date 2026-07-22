from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import InvalidUploadError
from app.schemas.api import UploadCandidate
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import InspectedUpload
from app.services.analysis.uploads import DatasetUploadService


def test_prepare_uploads_returns_typed_results_and_rejects_duplicate_hashes() -> None:
    service = DatasetUploadService(
        analysis=object(),
        storage=object(),
        files=DatasetFileService(),
    )
    content = b"region,revenue\nNorth,10\n"
    prepared = asyncio.run(
        service.prepare_uploads(
            UploadCandidate(
                file_name="sales.csv",
                content_type="text/csv",
                content=content,
            ),
            maximum_files=5,
        )
    )

    assert isinstance(prepared[0], InspectedUpload)
    assert prepared[0].inspection.row_count == 1

    with pytest.raises(InvalidUploadError, match="existing workspace dataset"):
        asyncio.run(
            service.prepare_uploads(
                UploadCandidate(
                    file_name="copy.csv",
                    content_type="text/csv",
                    content=content,
                ),
                existing_hashes={prepared[0].file_hash},
                maximum_files=4,
            )
        )
