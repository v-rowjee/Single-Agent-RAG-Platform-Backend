from fastapi import APIRouter

from app.api.analysis import router as analysis_router
from app.api.chat import router as chat_router


api_router = APIRouter()

api_router.include_router(analysis_router)
api_router.include_router(chat_router)
