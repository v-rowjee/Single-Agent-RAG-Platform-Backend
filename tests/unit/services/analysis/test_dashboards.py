from __future__ import annotations

from app.core.config import Settings
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import DatasetInspection
from app.services.persistence.analysis import DatasetRecord


def test_placeholder_dashboard_preserves_public_contract_and_values() -> None:
    assembler = DashboardAssembler(
        settings=Settings("", "", bi_pipeline_mode="multi"),
        files=DatasetFileService(),
    )
    dataset = DatasetRecord(
        id="session",
        user_id="user",
        file_name="sales.csv",
        storage_path="sales.csv",
        mime_type="text/csv",
        file_size=1024,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )
    payload = assembler.build_placeholder_dashboard(
        dataset,
        DatasetInspection(2, 2, ["revenue"], ["region"], 0, 0, 100.0),
    )

    assert payload["sessionId"] == "session"
    assert payload["dashboard"]["datasetSummary"]["rowCount"] == 2
    assert payload["dashboard"]["kpis"][0]["rawValue"] == 2
