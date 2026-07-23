"""Shared Pydantic primitives and domain aliases."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

DashboardStatus = Literal["success", "partial", "failed"]
ValueFormat = Literal["number", "currency", "percentage", "decimal", "text"]
IndicatorKind = Literal["increase", "decrease", "note"]
Severity = Literal["info", "warning", "critical"]
Priority = Literal["low", "medium", "high", "critical"]
Granularity = Literal["day", "week", "month", "quarter", "year"]

class StrictModel(BaseModel):
    """Forbid undeclared fields at internal structured-output boundaries."""

    model_config = ConfigDict(extra="forbid")
