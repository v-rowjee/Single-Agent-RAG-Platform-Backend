from __future__ import annotations

from app.core.exceptions import SessionNotFoundError
from app.services.analysis.files import DatasetFileService
from app.services.analysis.workspaces import WorkspaceService
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord

SESSION_ID = "9d719abc-9e09-4c14-b2d6-ed8308a1b85d"


class Repository:
    def __init__(self) -> None:
        self.session = AnalysisSessionRecord(
            SESSION_ID, "owner", None, "ready", "ready", None
        )
        self.dataset = DatasetRecord(
            "dataset", "owner", "sales.csv", "path", "text/csv", 10, "hash",
            None, "ready", "ready", None, session_id=SESSION_ID,
            row_count=1, column_count=2,
        )

    def get_session(self, session_id: str, user_id: str):
        return self.session if (session_id, user_id) == (SESSION_ID, "owner") else None

    def get_session_datasets(self, session_id: str, user_id: str):
        return [self.dataset] if (session_id, user_id) == (SESSION_ID, "owner") else []


def test_load_workspace_enforces_user_ownership() -> None:
    service = WorkspaceService(
        analysis=Repository(),
        storage=object(),
        files=DatasetFileService(),
    )

    session, datasets = service.load_workspace(SESSION_ID, "owner")
    assert session.id == SESSION_ID
    assert datasets[0].id == "dataset"

    try:
        service.load_workspace(SESSION_ID, "intruder")
    except SessionNotFoundError:
        pass
    else:
        raise AssertionError("Workspace ownership must be enforced.")
