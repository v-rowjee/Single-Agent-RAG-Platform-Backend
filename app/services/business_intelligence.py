"""Stable public facade for business-intelligence API callers."""

from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
)
from app.orchestration.graphs.analysis_graph import analysis_graph
from app.services.analysis.facade import BusinessIntelligenceService as _Facade
from app.services.analysis.models import PipelineExecution
from app.services.persistence.analysis import DatasetRecord


class BusinessIntelligenceService(_Facade):
    """Compatibility boundary retaining the established public import path."""

    async def _run_multi_agent_pipeline(
        self,
        dataset: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
        workspace_datasets: list[DatasetRecord] | None = None,
    ) -> PipelineExecution:
        # The graph remains patchable at this public boundary for existing
        # integration extensions. The focused runner owns state construction.
        return await self._pipeline_runner.run_multi_agent(
            dataset,
            content,
            workspace_session_id,
            workspace_datasets,
            graph=analysis_graph,
        )


business_intelligence_service = BusinessIntelligenceService()

__all__ = [
    "BusinessIntelligenceService",
    "DatasetAlreadyExistsError",
    "InvalidUploadError",
    "PipelineExecution",
    "SessionNotFoundError",
    "business_intelligence_service",
]
