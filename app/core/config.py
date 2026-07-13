from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "uploads"


def get_settings() -> Settings:
    return Settings(
        supabase_url=os.environ.get("SUPABASE_URL", "").strip(),
        supabase_service_role_key=os.environ.get(
            "SUPABASE_SERVICE_ROLE_KEY", ""
        ).strip(),
        supabase_storage_bucket=os.environ.get(
            "SUPABASE_STORAGE_BUCKET",
            "uploads",
        ).strip()
        or "uploads",
    )
