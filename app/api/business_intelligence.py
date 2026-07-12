from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.business_intelligence_service import (
    InvalidUploadError,
    SessionNotFoundError,
    business_intelligence_service,
)


router = APIRouter(tags=["business-intelligence"])


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1)
    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    response: str


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
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file.",
        ) from error


@router.get("/dashboard/{session_id}")
def get_dashboard(session_id: str) -> dict[str, Any]:
    try:
        return business_intelligence_service.get_dashboard(session_id)

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
        result = business_intelligence_service.chat(
            session_id=request.sessionId,
            query=request.query,
        )

        return ChatResponse(**result)

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