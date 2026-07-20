"""Small lazy, cached adapter for the official TimesFM 2.5 Torch model."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import get_runtime_config

_FORECASTING_POLICY = get_runtime_config().forecasting
TIMESFM_MODEL_ID = _FORECASTING_POLICY.model
MAX_CONTEXT = _FORECASTING_POLICY.max_context
MAX_HORIZON = _FORECASTING_POLICY.max_horizon


class TimesFMServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimesFMForecast:
    values: list[float]
    lower_bounds: list[float] | None = None
    upper_bounds: list[float] | None = None


class TimesFMService:
    def __init__(self) -> None:
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import timesfm
        except ImportError as exc:
            raise TimesFMServiceError("TimesFM is not installed; install timesfm[torch] to enable forecasting.") from exc
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                TIMESFM_MODEL_ID,
                torch_compile=False,
            )
            model.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=MAX_HORIZON, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        except Exception as exc:
            raise TimesFMServiceError(f"TimesFM model could not be loaded: {exc}") from exc
        self._model = model
        return model

    def _forecast_sync(self, values: list[float], horizon: int) -> TimesFMForecast:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise TimesFMServiceError("TimesFM expects a non-empty one-dimensional finite numeric series.")
        if not 1 <= horizon <= MAX_HORIZON:
            raise TimesFMServiceError(f"Forecast horizon must be between 1 and {MAX_HORIZON}.")
        point, quantiles = self._load_model().forecast(horizon=horizon, inputs=[array[-MAX_CONTEXT:]])
        point_values = np.asarray(point, dtype=float)[0, :horizon].tolist()
        lower: list[float] | None = None; upper: list[float] | None = None
        if quantiles is not None:
            quantile_values = np.asarray(quantiles, dtype=float)
            if quantile_values.ndim >= 3 and quantile_values.shape[1] >= horizon:
                # TimesFM exposes quantile channels. The minimum/maximum channel is a safe optional interval.
                lower = quantile_values[0, :horizon, :].min(axis=-1).tolist()
                upper = quantile_values[0, :horizon, :].max(axis=-1).tolist()
        return TimesFMForecast(values=[float(value) for value in point_values], lower_bounds=lower, upper_bounds=upper)

    async def forecast(self, values: list[float], horizon: int) -> TimesFMForecast:
        return await asyncio.to_thread(self._forecast_sync, values, horizon)


timesfm_service = TimesFMService()
