from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.schemas.business_intelligence import ChatRequest, ChatResponse

from app.services.business_intelligence_service import (
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
) -> dict[str, Any]:
    try:
        return await business_intelligence_service.create_analysis(
            file=file,
            description=description,
        )

    except InvalidUploadError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Unexpected error during file upload: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file.",
        ) from error


@router.get("/dashboard/{session_id}")
async def get_dashboard(session_id: str) -> dict[str, Any]:
    try:
        return (
            await business_intelligence_service.get_dashboard(session_id)
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
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return business_intelligence_service.chat(
            session_id=request.sessionId,
            query=request.query,
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
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    try:
        result = business_intelligence_service.get_chat_history(session_id)

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
