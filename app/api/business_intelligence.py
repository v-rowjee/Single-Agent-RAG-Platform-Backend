from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_current_user
from app.schemas.business_intelligence import (
    ActiveDatasetResponse,
    ChatRequest,
    ChatResponse,
    DatasetPreviewResponse,
)

from app.services.business_intelligence_service import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
    business_intelligence_service,
)


router = APIRouter(tags=["business-intelligence"])


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    grounded: bool = False
    createdAt: str


class ChatHistoryResponse(BaseModel):
    sessionId: str
    messages: list[ChatMessage]


@router.post("/upload")
async def upload_file(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    description: str | None = Form(default=None),
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        uploaded_files = list(files or [])
        if file is not None:
            uploaded_files.append(file)
        return await business_intelligence_service.create_analysis(
            files=uploaded_files,
            description=description,
            user_id=current_user.id,
            legacy_contract=not files and file is not None,
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
    current_user: CurrentUser = Depends(get_current_user),
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
    files: list[UploadFile] = File(...),
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return await business_intelligence_service.add_datasets(
            files=files,
            user_id=current_user.id,
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
    current_user: CurrentUser = Depends(get_current_user),
) -> None:
    try:
        await business_intelligence_service.remove_dataset(
            dataset_id=dataset_id,
            user_id=current_user.id,
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
    dataset_id: str | None = Query(default=None, min_length=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=50),
    current_user: CurrentUser = Depends(get_current_user),
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
    current_user: CurrentUser = Depends(get_current_user),
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
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        payload = (
            await business_intelligence_service.get_dashboard(
                session_id,
                current_user.id,
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


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    try:
        return business_intelligence_service.chat(
            session_id=request.sessionId,
            query=request.query,
            user_id=current_user.id,
        )

    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the query.",
        ) from error


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatHistoryResponse:
    try:
        result = business_intelligence_service.get_chat_history(
            session_id,
            current_user.id,
        )

        return ChatHistoryResponse(**result)

    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the chat history.",
        ) from error
