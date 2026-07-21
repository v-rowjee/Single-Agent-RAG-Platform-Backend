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


class RpcRequest:
    def execute(self) -> SimpleNamespace:
        return SimpleNamespace(data=2)


class AtomicIndexClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def rpc(self, name: str, payload: dict[str, object]) -> RpcRequest:
        self.calls.append((name, payload))
        return RpcRequest()


class MissingRpcRequest:
    def execute(self) -> SimpleNamespace:
        raise RuntimeError(
            "PGRST202 Could not find the function public.replace_document_chunks"
        )


class MissingRpcClient:
    def rpc(self, name: str, payload: dict[str, object]) -> MissingRpcRequest:
        return MissingRpcRequest()


class FallbackSupabaseService(SupabaseService):
    def __init__(self) -> None:
        super().__init__(settings=Settings("", ""))
        self.fallback_calls: list[tuple[str, list[dict[str, object]]]] = []

    def _replace_document_chunks_compat(
        self,
        dataset_id: str,
        chunks: list[dict[str, object]],
    ) -> int:
        self.fallback_calls.append((dataset_id, chunks))
        return len(chunks)


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


def test_replace_document_chunks_uses_one_atomic_rpc() -> None:
    service = SupabaseService(settings=Settings("", ""))
    client = AtomicIndexClient()
    service._client = client
    chunks = [
        {"source_id": "one", "embedding": [0.1] * 384},
        {"source_id": "two", "embedding": [0.2] * 384},
    ]

    count = service.replace_document_chunks("dataset-id", chunks)

    assert count == 2
    assert client.calls == [
        (
            "replace_document_chunks",
            {"p_dataset_id": "dataset-id", "p_chunks": chunks},
        )
    ]


def test_replace_document_chunks_keeps_legacy_projects_working() -> None:
    service = FallbackSupabaseService()
    service._client = MissingRpcClient()
    chunks = [{"source_id": "one", "embedding": [0.1] * 384}]

    count = service.replace_document_chunks("dataset-id", chunks)

    assert count == 1
    assert service.fallback_calls == [("dataset-id", chunks)]
