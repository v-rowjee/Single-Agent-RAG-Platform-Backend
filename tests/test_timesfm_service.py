from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from app.services.timesfm_service import TIMESFM_MODEL_ID, TimesFMService


def test_timesfm_disables_torch_compile_for_cpu_friendly_cold_start(
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeModel:
        def compile(self, config: object) -> None:
            calls["forecast_config"] = config

    class FakeTimesFMModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return FakeModel()

    class FakeForecastConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.values = kwargs

    fake_timesfm = ModuleType("timesfm")
    fake_timesfm.TimesFM_2p5_200M_torch = FakeTimesFMModel  # type: ignore[attr-defined]
    fake_timesfm.ForecastConfig = FakeForecastConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm", fake_timesfm)

    model = TimesFMService()._load_model()

    assert isinstance(model, FakeModel)
    assert calls["model_id"] == TIMESFM_MODEL_ID
    assert calls["model_kwargs"] == {"torch_compile": False}
    assert isinstance(calls["forecast_config"], FakeForecastConfig)
