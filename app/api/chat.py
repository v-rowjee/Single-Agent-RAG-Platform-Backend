"""Authenticated chat endpoints."""

from fastapi import APIRouter, HTTPException

from app.api.auth import AuthenticatedUser
from app.core.exceptions import SessionNotFoundError
from app.schemas.api import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.business_intelligence import business_intelligence_service


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, current_user: AuthenticatedUser) -> ChatResponse:
    try:
        return business_intelligence_service.chat(
            session_id=request.sessionId,
            query=request.query,
            user_id=current_user.id,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the query.",
        ) from error


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
    session_id: str,
    current_user: AuthenticatedUser,
) -> ChatHistoryResponse:
    try:
        result = business_intelligence_service.get_chat_history(
            session_id,
            current_user.id,
        )
        return ChatHistoryResponse(**result)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the chat history.",
        ) from error
