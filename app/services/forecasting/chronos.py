"""Lazy, cached adapter for the self-hosted Chronos-2 forecasting model."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_runtime_config

_FORECASTING_POLICY = get_runtime_config().forecasting
CHRONOS_MODEL_ID = _FORECASTING_POLICY.model
MAX_CONTEXT = _FORECASTING_POLICY.max_context
MAX_HORIZON = _FORECASTING_POLICY.max_horizon
QUANTILE_LEVELS = [0.025, 0.5, 0.975]


class ChronosServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChronosForecast:
    values: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]


class ChronosService:
    def __init__(self) -> None:
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from chronos import Chronos2Pipeline
            import torch
        except ImportError as exc:
            raise ChronosServiceError(
                "Chronos-2 is not installed; install chronos-forecasting>=2.0 to enable forecasting."
            ) from exc

        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = Chronos2Pipeline.from_pretrained(
                CHRONOS_MODEL_ID,
                device_map=device_map,
            )
        except Exception as exc:
            raise ChronosServiceError(f"Chronos-2 model could not be loaded: {exc}") from exc
        return self._model

    @staticmethod
    def _timestamp(value: object) -> pd.Timestamp:
        if isinstance(value, pd.Period):
            return value.start_time
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ChronosServiceError("Chronos-2 requires valid timestamps for every historical value.")
        return timestamp

    @staticmethod
    def _prediction_column(predictions: pd.DataFrame, quantile: float) -> str:
        candidates = (str(quantile), quantile)
        for candidate in candidates:
            if candidate in predictions.columns:
                return candidate  # type: ignore[return-value]
        raise ChronosServiceError(f"Chronos-2 response did not include quantile {quantile}.")

    def _forecast_sync(self, series: pd.Series, horizon: int) -> ChronosForecast:
        if not isinstance(series, pd.Series) or series.empty:
            raise ChronosServiceError("Chronos-2 expects a non-empty pandas Series.")
        values = np.asarray(series.values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ChronosServiceError("Chronos-2 expects a one-dimensional finite numeric series.")
        if not isinstance(series.index, pd.PeriodIndex | pd.DatetimeIndex):
            raise ChronosServiceError("Chronos-2 expects a period or datetime index.")
        if not 1 <= horizon <= MAX_HORIZON:
            raise ChronosServiceError(f"Forecast horizon must be between 1 and {MAX_HORIZON}.")

        context = pd.DataFrame(
            {
                "item_id": "primary_series",
                "timestamp": [self._timestamp(index) for index in series.index[-MAX_CONTEXT:]],
                "target": values[-MAX_CONTEXT:],
            }
        )
        try:
            predictions = self._load_model().predict_df(
                context,
                prediction_length=horizon,
                quantile_levels=QUANTILE_LEVELS,
                id_column="item_id",
                timestamp_column="timestamp",
                target="target",
            )
        except ChronosServiceError:
            raise
        except Exception as exc:
            raise ChronosServiceError(f"Chronos-2 forecasting failed: {exc}") from exc

        if not isinstance(predictions, pd.DataFrame) or len(predictions) < horizon:
            raise ChronosServiceError("Chronos-2 returned fewer prediction points than requested.")
        predictions = predictions.iloc[:horizon]
        point_column = "predictions"
        if point_column not in predictions:
            point_column = self._prediction_column(predictions, 0.5)
        lower_column = self._prediction_column(predictions, 0.025)
        upper_column = self._prediction_column(predictions, 0.975)
        point_values = predictions[point_column].to_numpy(dtype=float)
        lower_bounds = predictions[lower_column].to_numpy(dtype=float)
        upper_bounds = predictions[upper_column].to_numpy(dtype=float)
        if not (
            np.isfinite(point_values).all()
            and np.isfinite(lower_bounds).all()
            and np.isfinite(upper_bounds).all()
        ):
            raise ChronosServiceError("Chronos-2 returned non-finite forecast values.")

        return ChronosForecast(
            values=[float(value) for value in point_values],
            lower_bounds=[float(value) for value in lower_bounds],
            upper_bounds=[float(value) for value in upper_bounds],
        )

    async def forecast(self, series: pd.Series, horizon: int) -> ChronosForecast:
        return await asyncio.to_thread(self._forecast_sync, series, horizon)


chronos_service = ChronosService()
