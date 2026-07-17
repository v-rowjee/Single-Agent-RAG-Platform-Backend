from types import SimpleNamespace

from app.core.config import Settings
from app.services.supabase_service import SupabaseService


class LegacyDatasetsTable:
    def __init__(self) -> None:
        self.inserts: list[dict[str, object]] = []

    def insert(self, payload: dict[str, object]) -> "LegacyDatasetsTable":
        self.inserts.append(payload)
        return self

    def execute(self) -> SimpleNamespace:
        if len(self.inserts) == 1:
            raise RuntimeError("column datasets.row_count does not exist")

        payload = self.inserts[-1]
        return SimpleNamespace(
            data=[
                {
                    **payload,
                    "status": "processing",
                    "rag_status": "pending",
                    "error_message": None,
                }
            ]
        )


class LegacySchemaClient:
    def __init__(self) -> None:
        self.datasets = LegacyDatasetsTable()

    def table(self, name: str) -> LegacyDatasetsTable:
        assert name == "datasets"
        return self.datasets


def test_create_dataset_falls_back_when_metadata_columns_are_not_migrated() -> None:
    service = SupabaseService(settings=Settings("", ""))
    client = LegacySchemaClient()
    service._client = client

    dataset = service.create_dataset(
        dataset_id="9d719abc-9e09-4c14-b2d6-ed8308a1b85d",
        user_id="59b3d0fc-2d4a-40a0-8bb1-99e19da406ee",
        file_name="sales.csv",
        storage_path="user/session/sales.csv",
        mime_type="text/csv",
        file_size=42,
        file_hash="hash",
        description="Sales data",
        row_count=2,
        column_count=3,
    )

    assert client.datasets.inserts[0]["row_count"] == 2
    assert client.datasets.inserts[0]["column_count"] == 3
    assert "row_count" not in client.datasets.inserts[1]
    assert "column_count" not in client.datasets.inserts[1]
    assert dataset.row_count is None
    assert dataset.column_count is None
