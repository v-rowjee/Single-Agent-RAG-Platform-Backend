"""Environment settings and version-controlled runtime policies."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from dotenv import load_dotenv

PipelineMode = Literal["single", "multi"]
AgentProvider = Literal["groq", "openrouter"]
ReasoningEffort = Literal["none", "low", "medium", "high"] | None
SUPPORTED_PROVIDERS: frozenset[AgentProvider] = frozenset({"groq", "openrouter"})
REQUIRED_AGENT_KEYS = frozenset(
    {
        "data_preparation",
        "orchestrator",
        "kpi_trend",
        "anomaly_detection",
        "dashboard_generation",
        "insight_synthesis",
        "chat",
        "single_dashboard",
        "single_chat",
    }
)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.toml"
RAG_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag.toml"


# This central module is imported by every runtime boundary, so load secrets once.
load_dotenv(CONFIG_PATH.parent.parent / ".env")


class RuntimeConfigurationError(ValueError):
    """Raised when the checked-in agent configuration is not usable."""


@dataclass(frozen=True)
class AgentModelPolicy:
    provider: AgentProvider
    model: str
    temperature: float
    max_completion_tokens: int
    timeout_seconds: int = 120
    reasoning_effort: ReasoningEffort = None
    strict_json_schema: bool = False
    supports_response_format: bool = True


@dataclass(frozen=True)
class ForecastingPolicy:
    model: str
    max_context: int
    max_horizon: int


@dataclass(frozen=True)
class RuntimeConfiguration:
    pipeline_mode: PipelineMode
    forecasting: ForecastingPolicy
    agents: Mapping[str, AgentModelPolicy]


@dataclass(frozen=True)
class EmbeddingPolicy:
    model: str
    dimensions: int
    batch_size: int


@dataclass(frozen=True)
class RerankingPolicy:
    model: str
    batch_size: int
    limit: int
    max_length: int


@dataclass(frozen=True)
class RetrievalPolicy:
    vector_search_limit: int
    chat_search_limit: int
    match_threshold: float
    max_context_chars: int


@dataclass(frozen=True)
class ChunkingPolicy:
    size: int
    overlap: int
    max_row_batch_documents: int
    rows_per_batch_document: int
    max_columns_per_row_document: int


@dataclass(frozen=True)
class RagConfiguration:
    embedding: EmbeddingPolicy
    reranking: RerankingPolicy
    retrieval: RetrievalPolicy
    chunking: ChunkingPolicy


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "uploads"
    bi_pipeline_mode: PipelineMode = "multi"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigurationError(f"{name} must be a TOML table.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigurationError(f"{name} must be a non-empty string.")
    return value.strip()


def _number(value: Any, name: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigurationError(f"{name} must be a number.")
    if integer:
        if not isinstance(value, int) or value <= 0:
            raise RuntimeConfigurationError(f"{name} must be a positive integer.")
    elif not 0 <= float(value) <= 2:
        raise RuntimeConfigurationError(f"{name} must be between 0 and 2.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeConfigurationError(f"{name} must be a positive integer.")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeConfigurationError(f"{name} must be a non-negative integer.")
    return value


def _ratio(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigurationError(f"{name} must be a number.")
    result = float(value)
    if not 0 <= result <= 1:
        raise RuntimeConfigurationError(f"{name} must be between 0 and 1.")
    return result


def _agent_policy(name: str, values: Any) -> AgentModelPolicy:
    table = _mapping(values, f"agents.{name}")
    provider = _text(table.get("provider"), f"agents.{name}.provider").lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise RuntimeConfigurationError(
            f"agents.{name}.provider must be one of: {supported}."
        )

    reasoning = table.get("reasoning_effort")
    if reasoning is not None and reasoning not in {"none", "low", "medium", "high"}:
        raise RuntimeConfigurationError(
            f"agents.{name}.reasoning_effort must be none, low, medium, high, or omitted."
        )
    strict_json_schema = table.get("strict_json_schema", False)
    if not isinstance(strict_json_schema, bool):
        raise RuntimeConfigurationError(
            f"agents.{name}.strict_json_schema must be true or false."
        )
    supports_response_format = table.get("supports_response_format", True)
    if not isinstance(supports_response_format, bool):
        raise RuntimeConfigurationError(
            f"agents.{name}.supports_response_format must be true or false."
        )
    if strict_json_schema and not supports_response_format:
        raise RuntimeConfigurationError(
            f"agents.{name} cannot enable strict_json_schema when "
            "supports_response_format is false."
        )

    return AgentModelPolicy(
        provider=cast(AgentProvider, provider),
        model=_text(table.get("model"), f"agents.{name}.model"),
        temperature=float(
            _number(table.get("temperature"), f"agents.{name}.temperature")
        ),
        max_completion_tokens=int(
            _number(
                table.get("max_completion_tokens"),
                f"agents.{name}.max_completion_tokens",
                integer=True,
            )
        ),
        timeout_seconds=_positive_integer(
            table.get("timeout_seconds", 120),
            f"agents.{name}.timeout_seconds",
        ),
        reasoning_effort=reasoning,
        strict_json_schema=strict_json_schema,
        supports_response_format=supports_response_format,
    )


def _forecasting_policy(values: Any) -> ForecastingPolicy:
    table = _mapping(values, "forecasting")
    return ForecastingPolicy(
        model=_text(table.get("model"), "forecasting.model"),
        max_context=int(
            _number(table.get("max_context"), "forecasting.max_context", integer=True)
        ),
        max_horizon=int(
            _number(table.get("max_horizon"), "forecasting.max_horizon", integer=True)
        ),
    )


def load_runtime_config(path: Path = CONFIG_PATH) -> RuntimeConfiguration:
    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeConfigurationError(
            f"Agent configuration was not found: {path}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeConfigurationError(f"Invalid TOML in {path}: {exc}") from exc

    pipeline = _mapping(raw.get("pipeline"), "pipeline")
    mode = _text(pipeline.get("mode"), "pipeline.mode").lower()
    if mode not in {"single", "multi"}:
        raise RuntimeConfigurationError(
            "pipeline.mode must be either 'single' or 'multi'."
        )

    raw_agents = _mapping(raw.get("agents"), "agents")
    missing = REQUIRED_AGENT_KEYS - raw_agents.keys()
    if missing:
        raise RuntimeConfigurationError(
            "agents is missing required entries: " + ", ".join(sorted(missing))
        )
    agents = {
        name: _agent_policy(name, raw_agents[name]) for name in REQUIRED_AGENT_KEYS
    }
    return RuntimeConfiguration(
        pipeline_mode=mode,
        forecasting=_forecasting_policy(raw.get("forecasting")),
        agents=agents,
    )  # type: ignore[arg-type]


def load_rag_config(path: Path = RAG_CONFIG_PATH) -> RagConfiguration:
    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeConfigurationError(
            f"RAG configuration was not found: {path}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeConfigurationError(f"Invalid TOML in {path}: {exc}") from exc

    embedding = _mapping(raw.get("embedding"), "embedding")
    reranking = _mapping(raw.get("reranking"), "reranking")
    retrieval = _mapping(raw.get("retrieval"), "retrieval")
    chunking = _mapping(raw.get("chunking"), "chunking")

    chunk_size = _positive_integer(chunking.get("size"), "chunking.size")
    chunk_overlap = _non_negative_integer(
        chunking.get("overlap"),
        "chunking.overlap",
    )
    if chunk_overlap >= chunk_size:
        raise RuntimeConfigurationError(
            "chunking.overlap must be smaller than chunking.size."
        )

    vector_search_limit = _positive_integer(
        retrieval.get("vector_search_limit"),
        "retrieval.vector_search_limit",
    )
    chat_search_limit = _positive_integer(
        retrieval.get("chat_search_limit"),
        "retrieval.chat_search_limit",
    )
    rerank_limit = _positive_integer(reranking.get("limit"), "reranking.limit")
    if chat_search_limit > vector_search_limit:
        raise RuntimeConfigurationError(
            "retrieval.chat_search_limit cannot exceed "
            "retrieval.vector_search_limit."
        )
    if rerank_limit > vector_search_limit:
        raise RuntimeConfigurationError(
            "reranking.limit cannot exceed retrieval.vector_search_limit."
        )

    return RagConfiguration(
        embedding=EmbeddingPolicy(
            model=_text(embedding.get("model"), "embedding.model"),
            dimensions=_positive_integer(
                embedding.get("dimensions"),
                "embedding.dimensions",
            ),
            batch_size=_positive_integer(
                embedding.get("batch_size"),
                "embedding.batch_size",
            ),
        ),
        reranking=RerankingPolicy(
            model=_text(reranking.get("model"), "reranking.model"),
            batch_size=_positive_integer(
                reranking.get("batch_size"),
                "reranking.batch_size",
            ),
            limit=rerank_limit,
            max_length=_positive_integer(
                reranking.get("max_length"),
                "reranking.max_length",
            ),
        ),
        retrieval=RetrievalPolicy(
            vector_search_limit=vector_search_limit,
            chat_search_limit=chat_search_limit,
            match_threshold=_ratio(
                retrieval.get("match_threshold"),
                "retrieval.match_threshold",
            ),
            max_context_chars=_positive_integer(
                retrieval.get("max_context_chars"),
                "retrieval.max_context_chars",
            ),
        ),
        chunking=ChunkingPolicy(
            size=chunk_size,
            overlap=chunk_overlap,
            max_row_batch_documents=_positive_integer(
                chunking.get("max_row_batch_documents"),
                "chunking.max_row_batch_documents",
            ),
            rows_per_batch_document=_positive_integer(
                chunking.get("rows_per_batch_document"),
                "chunking.rows_per_batch_document",
            ),
            max_columns_per_row_document=_positive_integer(
                chunking.get("max_columns_per_row_document"),
                "chunking.max_columns_per_row_document",
            ),
        ),
    )


@lru_cache(maxsize=1)
def get_runtime_config() -> RuntimeConfiguration:
    return load_runtime_config()


@lru_cache(maxsize=1)
def get_rag_config() -> RagConfiguration:
    return load_rag_config()


def agent_model_policy(agent: str) -> AgentModelPolicy:
    """Return the version-controlled policy for one LLM invocation."""
    try:
        return get_runtime_config().agents[agent]
    except KeyError as exc:
        raise KeyError(f"Unknown agent model policy: {agent}") from exc


def configured_agent_models() -> dict[str, str]:
    """Expose resolved assignments for diagnostics without exposing API keys."""
    return {
        name: policy.model for name, policy in get_runtime_config().agents.items()
    }


def get_settings() -> Settings:
    runtime = get_runtime_config()
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
        bi_pipeline_mode=runtime.pipeline_mode,
    )
