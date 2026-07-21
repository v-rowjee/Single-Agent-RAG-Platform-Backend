from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_rag_config, get_runtime_config
from app.core.prompts import validate_prompt_bundles


# Fail at boot rather than during a user request when checked-in assets are invalid.
get_runtime_config()
get_rag_config()
validate_prompt_bundles()


app = FastAPI(
    title="Business Intelligence API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Business Intelligence API is running"}
