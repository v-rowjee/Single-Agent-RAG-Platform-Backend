"""Capability-routed specialist node adapters and failure boundaries."""

import logging
from typing import Any, Awaitable, Callable, Mapping

from app.orchestration.state import AnalysisState

logger = logging.getLogger(__name__)
StateNode = Callable[[AnalysisState], Awaitable[dict[str, Any]]]


def _recoverable_node(
    name: str,
    node: StateNode,
    *,
    empty_update: Mapping[str, Any],
    required: bool = False,
) -> StateNode:
    """Keep optional branch failures in state so graph fan-in can still finish."""

    async def run(state: AnalysisState) -> dict[str, Any]:
        try:
            return await node(state)
        except Exception as exc:
            logger.exception("Multi-agent node failed node=%s", name)
            message = f"{name.replace('_', ' ').title()} failed: {exc}"
            update = dict(empty_update)
            update["failed_agents"] = [name]
            update["errors" if required else "warnings"] = [message]
            return update

    return run
