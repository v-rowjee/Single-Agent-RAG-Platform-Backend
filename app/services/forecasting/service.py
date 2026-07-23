"""Forecasting interface used by specialist agents."""

from __future__ import annotations

from typing import Protocol

from app.services.forecasting.chronos import ChronosForecast, chronos_service


class ForecastingService(Protocol):
    async def forecast(
        self,
        values: list[float],
        horizon: int,
    ) -> ChronosForecast: ...


forecasting_service: ForecastingService = chronos_service
