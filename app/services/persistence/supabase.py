"""Low-level Supabase client and object-storage gateway."""

from __future__ import annotations

from app.core.config import Settings, get_settings

class SupabaseUnavailableError(Exception):
    """Raised when the backend cannot use Supabase."""

class SupabaseGateway:
    def __init__(self, settings: Settings | None = None) -> None:
            self.settings = settings or get_settings()
            self._client: object | None = None

    @property
    def client(self):
            if self._client is None:
                if (
                    not self.settings.supabase_url
                    or not self.settings.supabase_service_role_key
                ):
                    raise SupabaseUnavailableError(
                        "Supabase credentials are not configured."
                    )

                from supabase import create_client

                self._client = create_client(
                    self.settings.supabase_url,
                    self.settings.supabase_service_role_key,
                )
            return self._client

    def upload_file(
            self,
            storage_path: str,
            content: bytes,
            mime_type: str,
        ) -> None:
            bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
            file_options = {
                "content-type": mime_type,
                "upsert": "false",
            }
            try:
                bucket.upload(storage_path, content, file_options=file_options)
            except TypeError:
                bucket.upload(storage_path, content, file_options)

    def download_file(self, storage_path: str) -> bytes:
            bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
            payload = bucket.download(storage_path)
            if isinstance(payload, bytes):
                return payload
            if isinstance(payload, bytearray):
                return bytes(payload)
            raise SupabaseUnavailableError("Storage download did not return bytes.")

    def delete_file(self, storage_path: str) -> None:
            bucket = self.client.storage.from_(self.settings.supabase_storage_bucket)
            bucket.remove([storage_path])

supabase_gateway = SupabaseGateway()
