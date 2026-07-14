"""Persist the final BI response through the project's existing Supabase service."""
from __future__ import annotations

from typing import Any

from app.schemas.business_intelligence import DashboardResponse
from app.services.supabase_service import SupabaseService, supabase_service


class BusinessIntelligencePersistenceService:
    def __init__(self, storage: SupabaseService | None = None) -> None:
        self.storage = storage or supabase_service

    def persist_workflow(self, bundle: dict[str, Any]) -> dict[str, Any]:
        session_id = str(bundle.get("session_id") or "")
        dataset_id = str(bundle.get("dataset_id") or session_id)
        try:
            response = DashboardResponse.model_validate(bundle["dashboard_output"])
            # `dashboards.response` is the project's existing JSON persistence
            # location.  The canonical response remains at the top level; the
            # workflow payload is ignored by DashboardResponse when the API
            # subsequently reads this record.
            stored_response = response.model_dump(mode="json")
            stored_response["workflow"] = {
                key: value
                for key, value in bundle.items()
                if key != "dashboard_output"
            }
            stored_response["workflow"]["dashboard_output"] = stored_response.copy()
            stored_response["workflow"]["dashboard_output"].pop("workflow", None)
            self.storage.save_dashboard(
                dataset_id=dataset_id,
                status=response.status,
                response=stored_response,
            )
            self.storage.update_dataset_status(
                dataset_id,
                status="ready" if response.status != "failed" else "failed",
                rag_status=(
                    "ready"
                    if (bundle.get("retrieval_indexing_result") or {}).get("status")
                    == "success"
                    else "failed"
                ),
                error_message=None,
            )
            return {
                "status": "success",
                "session_id": session_id,
                "dataset_id": dataset_id,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "session_id": session_id,
                "dataset_id": dataset_id,
                "message": str(exc),
            }


business_intelligence_persistence_service = BusinessIntelligencePersistenceService()
