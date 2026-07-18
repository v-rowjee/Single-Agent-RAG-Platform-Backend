from __future__ import annotations

from pathlib import Path

import pytest
import toons

import app.core.prompts as prompt_module
from app.core.config import (
    CONFIG_PATH,
    RAG_CONFIG_PATH,
    RuntimeConfigurationError,
    get_runtime_config,
    get_settings,
    load_rag_config,
    load_runtime_config,
)
from app.core.prompts import (
    PROMPTS_ROOT,
    PromptTemplateError,
    render_agent_prompts,
    validate_prompt_bundles,
)


def test_checked_in_configuration_is_multi_agent_and_groq_only() -> None:
    config = load_runtime_config()

    assert config.pipeline_mode == "multi"
    assert {policy.provider for policy in config.agents.values()} == {"groq"}
    assert config.agents["data_preparation"].model == "openai/gpt-oss-20b"
    assert config.agents["insight_synthesis"].model == "openai/gpt-oss-120b"
    assert config.forecasting.model == "google/timesfm-2.5-200m-pytorch"


def test_environment_does_not_override_versioned_agent_configuration(monkeypatch) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "single")
    monkeypatch.setenv("GROQ_MODEL_DATA_PREPARATION", "not-a-configured-model")
    get_runtime_config.cache_clear()

    assert get_settings().bi_pipeline_mode == "multi"
    assert get_runtime_config().agents["data_preparation"].model == "openai/gpt-oss-20b"

    get_runtime_config.cache_clear()


def test_checked_in_rag_configuration() -> None:
    config = load_rag_config()

    assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding.batch_size == 64
    assert config.reranking.limit == 5
    assert config.retrieval.vector_search_limit == 12
    assert config.retrieval.chat_search_limit == 6
    assert config.retrieval.match_threshold == 0.2
    assert config.chunking.size == 800
    assert config.chunking.overlap == 100


def test_invalid_rag_configuration_is_rejected(tmp_path: Path) -> None:
    content = RAG_CONFIG_PATH.read_text(encoding="utf-8")
    config_path = tmp_path / "rag.toml"
    config_path.write_text(
        content.replace("overlap = 100", "overlap = 800"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError, match="chunking.overlap"):
        load_rag_config(config_path)

    config_path.write_text(
        content.replace("match_threshold = 0.2", "match_threshold = 1.2"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="retrieval.match_threshold"):
        load_rag_config(config_path)


def test_invalid_pipeline_mode_is_rejected(tmp_path: Path) -> None:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace('mode = "multi"', 'mode = "invalid"')
    config_path = tmp_path / "agents.toml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match="pipeline.mode"):
        load_runtime_config(config_path)


def test_invalid_agent_provider_and_model_are_rejected(tmp_path: Path) -> None:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'provider = "groq"', 'provider = "unsupported"', 1
    )
    config_path = tmp_path / "agents.toml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match="provider"):
        load_runtime_config(config_path)

    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            'model = "openai/gpt-oss-20b"', 'model = ""', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="model"):
        load_runtime_config(config_path)


def test_prompt_bundles_validate_and_render_structured_toon() -> None:
    expected_bundles = {
        "multi/anomaly_detection.toon",
        "multi/chat.toon",
        "multi/dashboard_generation.toon",
        "multi/data_preparation.toon",
        "multi/insight_synthesis.toon",
        "multi/kpi_trend.toon",
        "single/business_intelligence.toon",
    }
    actual_bundles = {
        path.relative_to(PROMPTS_ROOT).as_posix()
        for path in PROMPTS_ROOT.rglob("*.toon")
    }
    assert actual_bundles == expected_bundles

    validate_prompt_bundles()
    prompts = render_agent_prompts(
        "multi/data_preparation",
        supported_operations=["preserve_missing"],
        supported_formulas=["quantity_times_unit_price"],
        profile={"row_count": 12, "columns": ["date", "revenue"]},
    )

    system = toons.loads(prompts.system, strict=True)
    user = toons.loads(prompts.user, strict=True)
    assert system["supported_operations"] == ["preserve_missing"]
    assert system["supported_formulas"] == ["quantity_times_unit_price"]
    assert user["profile"]["row_count"] == 12

    with pytest.raises(PromptTemplateError, match="supported_formulas"):
        render_agent_prompts(
            "multi/data_preparation",
            supported_operations=["preserve_missing"],
            profile={},
        )


def test_single_agent_message_sets_render_structured_toon() -> None:
    prompts = render_agent_prompts(
        "single/business_intelligence",
        "profile_chat",
        profile={"summary": {"rowCount": 5}},
        history=[{"role": "user", "content": "What is in this data?"}],
        query="What is in this data?",
    )

    system = toons.loads(prompts.system, strict=True)
    user = toons.loads(prompts.user, strict=True)
    assert system["output"]["format"] == "Markdown under 90 words"
    assert user["profile"]["summary"]["rowCount"] == 5


@pytest.fixture
def isolated_prompt_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    root.mkdir()
    monkeypatch.setattr(prompt_module, "PROMPTS_ROOT", root)
    prompt_module._bundle.cache_clear()
    yield root
    prompt_module._bundle.cache_clear()


def test_malformed_toon_bundle_is_rejected(isolated_prompt_root: Path) -> None:
    (isolated_prompt_root / "broken.toon").write_text(
        "agent: broken\nmessages[2]: only_one\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="Invalid TOON"):
        validate_prompt_bundles()


def test_invalid_bundle_shape_is_rejected(isolated_prompt_root: Path) -> None:
    (isolated_prompt_root / "broken.toon").write_text(
        """agent: broken
messages:
  default:
    system: not_an_object
    user:
      input: valid
""",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="system"):
        validate_prompt_bundles()


def test_invalid_message_set_and_path_are_rejected() -> None:
    with pytest.raises(PromptTemplateError, match="does not define message set"):
        render_agent_prompts("multi/chat", "missing")

    with pytest.raises(PromptTemplateError, match="Invalid prompt bundle name"):
        render_agent_prompts("../outside")
