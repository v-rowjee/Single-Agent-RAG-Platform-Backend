from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from app.api.auth import AuthenticatedUser
from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
)
from app.schemas.api import DatasetPreviewResponse, UploadCandidate
from app.services.business_intelligence import (
    business_intelligence_service,
)


router = APIRouter(tags=["analysis"])


async def _upload_candidate(file: UploadFile) -> UploadCandidate:
    return UploadCandidate(
        file_name=file.filename or "",
        content_type=file.content_type or "",
        content=await file.read(),
    )


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    description: str | None = Form(default=None),
) -> dict[str, Any]:
    try:
        legacy_contract = not files and file is not None
        uploaded_files = list(files or [])
        if file is not None:
            uploaded_files.append(file)
        return await business_intelligence_service.create_analysis(
            files=[await _upload_candidate(item) for item in uploaded_files],
            description=description,
            user_id=current_user.id,
            legacy_contract=legacy_contract,
            background_tasks=background_tasks,
        )

    except InvalidUploadError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except DatasetAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Unexpected error during file upload: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file.",
        ) from error


@router.get("/dataset")
def get_active_dataset(
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    try:
        return business_intelligence_service.get_active_dataset_details(
            current_user.id,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dataset.",
        ) from error


@router.post("/dataset")
async def add_datasets(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    try:
        return await business_intelligence_service.add_datasets(
            files=[await _upload_candidate(item) for item in files],
            user_id=current_user.id,
            background_tasks=background_tasks,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Unexpected error while adding datasets: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while adding the datasets.",
        ) from error


@router.delete("/dataset/{dataset_id}", status_code=204)
async def remove_dataset(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> None:
    try:
        await business_intelligence_service.remove_dataset(
            dataset_id=dataset_id,
            user_id=current_user.id,
            background_tasks=background_tasks,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Unexpected error while removing a dataset: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while removing the dataset.",
        ) from error


@router.get("/dataset/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    current_user: AuthenticatedUser,
    dataset_id: str | None = Query(default=None, min_length=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=50),
) -> DatasetPreviewResponse:
    try:
        return DatasetPreviewResponse(
            **business_intelligence_service.get_dataset_preview(
                current_user.id,
                dataset_id,
                page,
                page_size,
            )
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dataset preview.",
        ) from error


@router.post("/dataset/reset", status_code=204)
def reset_dataset(
    current_user: AuthenticatedUser,
) -> None:
    try:
        business_intelligence_service.reset_active_dataset(current_user.id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while resetting the dataset.",
        ) from error


@router.get("/dashboard/{session_id}")
async def get_dashboard(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    try:
        payload = (
            await business_intelligence_service.get_dashboard(
                session_id,
                current_user.id,
                background_tasks=background_tasks,
            )
        ).model_dump(mode="json")
        dashboard = payload.get("dashboard")
        if (
            business_intelligence_service.uses_legacy_contract(session_id)
            and isinstance(dashboard, dict)
            and isinstance(dashboard.get("datasetSummaries"), list)
            and dashboard["datasetSummaries"]
        ):
            legacy_summary = dict(dashboard["datasetSummaries"][0])
            legacy_summary.pop("datasetId", None)
            dashboard["datasetSummary"] = legacy_summary
        return payload

    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dashboard.",
        ) from error


@router.post("/dashboard/{session_id}/rag/rebuild", status_code=202)
def rebuild_dashboard_retrieval(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> dict[str, str]:
    try:
        business_intelligence_service.rebuild_dashboard_retrieval(
            session_id,
            current_user.id,
            background_tasks,
        )
        return {"status": "indexing", "message": "Retrieval index rebuild started."}
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while rebuilding retrieval.",
        ) from error


