from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd

from app.services.chronos_service import CHRONOS_MODEL_ID, ChronosService


def _install_fake_dependencies(monkeypatch, *, cuda_available: bool) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    class FakeModel:
        def predict_df(self, context: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
            calls["context"] = context.copy()
            calls["predict_kwargs"] = kwargs
            return pd.DataFrame(
                {
                    "item_id": ["primary_series", "primary_series"],
                    "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
                    "predictions": [120.0, 125.0],
                    "0.025": [100.0, 103.0],
                    "0.5": [120.0, 125.0],
                    "0.975": [140.0, 147.0],
                }
            )

    class FakeChronos2Pipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return FakeModel()

    fake_chronos = ModuleType("chronos")
    fake_chronos.Chronos2Pipeline = FakeChronos2Pipeline  # type: ignore[attr-defined]
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def test_chronos_selects_the_available_device(monkeypatch) -> None:
    calls = _install_fake_dependencies(monkeypatch, cuda_available=False)

    ChronosService()._load_model()

    assert calls["model_id"] == CHRONOS_MODEL_ID
    assert calls["model_kwargs"] == {"device_map": "cpu"}


def test_chronos_maps_dataframe_quantiles_to_the_forecast_contract(monkeypatch) -> None:
    calls = _install_fake_dependencies(monkeypatch, cuda_available=True)
    series = pd.Series(
        [90.0, 100.0, 110.0],
        index=pd.period_range("2024-10", periods=3, freq="M"),
    )

    result = ChronosService()._forecast_sync(series, horizon=2)

    assert calls["model_kwargs"] == {"device_map": "cuda"}
    assert calls["context"].to_dict("list") == {
        "item_id": ["primary_series"] * 3,
        "timestamp": list(pd.to_datetime(["2024-10-01", "2024-11-01", "2024-12-01"])),
        "target": [90.0, 100.0, 110.0],
    }
    assert calls["predict_kwargs"] == {
        "prediction_length": 2,
        "quantile_levels": [0.025, 0.5, 0.975],
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "target": "target",
    }
    assert result.values == [120.0, 125.0]
    assert result.lower_bounds == [100.0, 103.0]
    assert result.upper_bounds == [140.0, 147.0]
