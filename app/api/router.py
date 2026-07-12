from fastapi import APIRouter

from app.api.business_intelligence import router as business_intelligence_router


api_router = APIRouter()

api_router.include_router(business_intelligence_router)