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
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return await business_intelligence_service.create_analysis(
            file=file,
            description=description,
            user_id=current_user.id,
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


@router.get("/dataset", response_model=ActiveDatasetResponse)
def get_active_dataset(
    current_user: CurrentUser = Depends(get_current_user),
) -> ActiveDatasetResponse:
    try:
        return ActiveDatasetResponse(
            **business_intelligence_service.get_active_dataset_details(
                current_user.id,
            )
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


@router.get("/dataset/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=50),
    current_user: CurrentUser = Depends(get_current_user),
) -> DatasetPreviewResponse:
    try:
        return DatasetPreviewResponse(
            **business_intelligence_service.get_dataset_preview(
                current_user.id,
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
        return (
            await business_intelligence_service.get_dashboard(
                session_id,
                current_user.id,
            )
        ).model_dump(mode="json")

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
