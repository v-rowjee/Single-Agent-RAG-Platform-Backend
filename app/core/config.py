from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


PipelineMode = Literal["single", "multi"]


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "uploads"
    # The historical upload workflow uses the single-agent pipeline.
    bi_pipeline_mode: PipelineMode = "single"


def get_settings() -> Settings:
    pipeline_mode = os.environ.get("BI_PIPELINE_MODE", "single").strip().lower()
    if pipeline_mode not in {"single", "multi"}:
        raise ValueError("BI_PIPELINE_MODE must be either 'single' or 'multi'.")
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
        bi_pipeline_mode=cast(PipelineMode, pipeline_mode),
    )
