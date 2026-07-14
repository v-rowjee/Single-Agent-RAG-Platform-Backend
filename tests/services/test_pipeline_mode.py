from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings, get_settings
from app.schemas.business_intelligence import DashboardResponse
from app.services.business_intelligence_service import BusinessIntelligenceService
from app.services.supabase_service import DatasetRecord


SESSION_ID = "15dc222f-bdfa-4e32-9252-19d9f57cc28a"


def session() -> DatasetRecord:
    return DatasetRecord(
        id=SESSION_ID,
        file_name="sales.csv",
        storage_path=f"{SESSION_ID}/sales.csv",
        mime_type="text/csv",
        file_size=10,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )


def dashboard(service: BusinessIntelligenceService) -> DashboardResponse:
    return DashboardResponse.model_validate(
        service._build_placeholder_dashboard(
            dataset=session(),
            dataset_info={
                "rowCount": 2,
                "columnCount": 2,
                "measures": ["Revenue"],
                "dimensions": ["Branch"],
                "missingValueCount": 0,
                "duplicateRowCount": 0,
                "completenessPercent": 100.0,
            },
        )
    )


@pytest.mark.parametrize("mode, selected", [("single", "single"), ("multi", "multi")])
def test_pipeline_mode_selects_only_the_configured_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    selected: str,
) -> None:
    service = BusinessIntelligenceService(
        settings=Settings(
            supabase_url="",
            supabase_service_role_key="",
            bi_pipeline_mode=mode,  # type: ignore[arg-type]
        )
    )
    expected = dashboard(service)
    single = AsyncMock(return_value=expected)
    multi = AsyncMock(return_value=expected)
    monkeypatch.setattr(service, "_run_single_agent_pipeline", single)
    monkeypatch.setattr(service, "_run_multi_agent_pipeline", multi)

    result = asyncio.run(service._run_selected_pipeline(session()))

    assert isinstance(result, DashboardResponse)
    assert DashboardResponse.model_validate(result.model_dump()) == expected
    if selected == "single":
        single.assert_awaited_once()
        multi.assert_not_awaited()
    else:
        multi.assert_awaited_once()
        single.assert_not_awaited()


def test_pipeline_modes_share_the_canonical_dashboard_contract() -> None:
    single_service = BusinessIntelligenceService(
        settings=Settings("", "", bi_pipeline_mode="single")
    )
    multi_service = BusinessIntelligenceService(
        settings=Settings("", "", bi_pipeline_mode="multi")
    )

    single_result = dashboard(single_service)
    multi_result = dashboard(multi_service)

    assert DashboardResponse.model_validate(single_result.model_dump())
    assert DashboardResponse.model_validate(multi_result.model_dump())
    assert type(single_result) is type(multi_result) is DashboardResponse


def test_invalid_pipeline_mode_is_rejected_when_settings_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "random")

    with pytest.raises(ValueError, match="BI_PIPELINE_MODE must be either 'single' or 'multi'"):
        get_settings()
