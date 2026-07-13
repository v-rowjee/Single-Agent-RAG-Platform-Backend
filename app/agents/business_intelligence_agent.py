from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, SecretStr

from app.schemas.business_intelligence import (
    BusinessIntelligenceAgentInput,
    DashboardResponse,
)
from app.rag.models import RerankedDocument, RetrievedDocument
from app.rag.rag_service import compact_profile_for_chat, rag_service

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
MODEL_NAME = "llama-3.3-70b-versatile"
CHAT_MAX_TOKENS = 220
logger = logging.getLogger(__name__)


class DraftAction(BaseModel):
    title: str
    description: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class Narrative(BaseModel):
    title: str = Field(min_length=1)
    executiveSummary: str
    businessSummary: str
    keyFindings: list[str] = Field(default_factory=list, max_length=5)
    opportunities: list[str] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=3)
    actions: list[DraftAction] = Field(default_factory=list, max_length=3)


class AgentState(TypedDict, total=False):
    mode: Literal["dashboard", "chat"]
    agent_input: BusinessIntelligenceAgentInput
    query: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    query_type: str
    calculated_evidence: str | None
    direct_answer: str | None
    retrieved_documents: list[RetrievedDocument]
    reranked_documents: list[RerankedDocument]
    retrieved_context: str
    dashboard_response: DashboardResponse
    chat_response: str


class BusinessIntelligenceAgent:
    """Compact BI pipeline with deterministic final schema construction."""

    def __init__(self) -> None:
        self._dashboard_chain: Any | None = None
        self._rag_chat_chain: Any | None = None
        self._profile_chat_chain: Any | None = None
        self._profiles: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[dict[str, str]]] = {}
        self._last_source_ids: dict[str, list[str]] = {}
        self.graph = self._build_graph()

    def run(self, agent_input: BusinessIntelligenceAgentInput) -> DashboardResponse:
        return self.graph.invoke({"mode": "dashboard", "agent_input": agent_input})[
            "dashboard_response"
        ]

    def generate_dashboard(
        self, agent_input: BusinessIntelligenceAgentInput
    ) -> DashboardResponse:
        return self.run(agent_input)

    def profile_for_session(
        self,
        agent_input: BusinessIntelligenceAgentInput,
    ) -> dict[str, Any]:
        profile = self._profiles.get(agent_input.sessionId)
        if profile is None:
            profile = self._profile(agent_input)
            self._profiles[agent_input.sessionId] = profile
        return profile

    def chat(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        query = query.strip()
        if not query:
            raise ValueError("The query cannot be empty.")

        conversation_history = (
            history
            if history is not None
            else self._history.get(agent_input.sessionId, [])
        )
        response = self.graph.invoke(
            {
                "mode": "chat",
                "agent_input": agent_input,
                "query": query,
                "history": conversation_history,
            }
        )["chat_response"]

        self._history[agent_input.sessionId] = [
            *conversation_history,
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ][-12:]
        return response

    def source_ids_for_session(self, session_id: str) -> list[str]:
        return list(self._last_source_ids.get(session_id, []))

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("dashboard", self._dashboard)
        graph.add_node("route_chat_query", self._route_chat_query)
        graph.add_node("calculate_evidence", self._calculate_evidence)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("rerank_documents", self._rerank_documents)
        graph.add_node("answer_chat", self._answer_chat)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges(
            "prepare",
            lambda state: state["mode"],
            {"dashboard": "dashboard", "chat": "route_chat_query"},
        )
        graph.add_edge("dashboard", END)
        graph.add_edge("route_chat_query", "calculate_evidence")
        graph.add_edge("calculate_evidence", "retrieve_documents")
        graph.add_edge("retrieve_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "answer_chat")
        graph.add_edge("answer_chat", END)
        return graph.compile()

    def _prepare(self, state: AgentState) -> dict[str, Any]:
        agent_input = state["agent_input"]
        profile = self._profiles.get(agent_input.sessionId)
        if profile is None:
            profile = self._profile(agent_input)
            self._profiles[agent_input.sessionId] = profile
        return {"profile": profile}

    def _dashboard(self, state: AgentState) -> dict[str, DashboardResponse]:
        self._create_chains()
        agent_input = state["agent_input"]
        profile = state["profile"]

        try:
            narrative = self._dashboard_chain.invoke(
                {
                    "description": agent_input.description or "Not provided",
                    "profile": self._json(profile),
                }
            )
            if not isinstance(narrative, Narrative):
                narrative = Narrative.model_validate(narrative)
        except Exception:
            narrative = self._fallback(agent_input, profile)

        return {"dashboard_response": self._response(agent_input, profile, narrative)}

    def _route_chat_query(self, state: AgentState) -> dict[str, str]:
        query_type = rag_service.route_query(state["query"], state["profile"])
        return {"query_type": query_type}

    def _calculate_evidence(self, state: AgentState) -> dict[str, str | None]:
        evidence = rag_service.calculate_evidence(
            agent_input=state["agent_input"],
            query=state["query"],
            query_type=state["query_type"],
            profile=state["profile"],
        )
        if evidence is None:
            return {"calculated_evidence": None, "direct_answer": None}
        return {
            "calculated_evidence": evidence.text,
            "direct_answer": evidence.direct_answer,
        }

    def _retrieve_documents(
        self,
        state: AgentState,
    ) -> dict[str, list[RetrievedDocument]]:
        agent_input = state["agent_input"]
        if not rag_service.ensure_index(agent_input, state["profile"]):
            return {"retrieved_documents": []}
        documents = rag_service.retrieve(agent_input=agent_input, query=state["query"])
        return {"retrieved_documents": documents}

    def _rerank_documents(
        self,
        state: AgentState,
    ) -> dict[str, list[RerankedDocument] | str]:
        retrieved = state.get("retrieved_documents", [])
        reranked = rag_service.rerank(state["query"], retrieved)
        context_documents = reranked if reranked else retrieved
        context = rag_service.build_context(
            context_documents,
            calculated_evidence=state.get("calculated_evidence"),
        )
        return {
            "reranked_documents": reranked,
            "retrieved_context": context,
        }

    def _answer_chat(self, state: AgentState) -> dict[str, str]:
        context = state.get("retrieved_context", "").strip()
        calculated_evidence = state.get("calculated_evidence")
        direct_answer = state.get("direct_answer")
        source_ids = self._source_ids(
            state.get("reranked_documents") or state.get("retrieved_documents", [])
        )
        self._last_source_ids[state["agent_input"].sessionId] = source_ids

        if not context and direct_answer:
            return {"chat_response": direct_answer}

        if context:
            try:
                self._create_chains()
                response = self._rag_chat_chain.invoke(
                    {
                        "history": self._history_text(state.get("history", [])),
                        "context": context,
                        "query": state["query"],
                    }
                )
                logger.info(
                    "RAG grounded answer generated session_id=%s sources=%s calculated=%s",
                    state["agent_input"].sessionId,
                    source_ids,
                    bool(calculated_evidence),
                )
                return {"chat_response": response.strip()}
            except Exception:
                logger.exception(
                    "Groq RAG answer generation failed session_id=%s",
                    state["agent_input"].sessionId,
                )
                if direct_answer:
                    return {"chat_response": direct_answer}

        fallback = self._profile_based_chat_fallback(state)
        if fallback:
            return {"chat_response": fallback}

        closest = ", ".join(f"`{source_id}`" for source_id in source_ids) or "none"
        return {
            "chat_response": (
                "**Answer:** The indexed dataset evidence is not sufficient to "
                "answer this question reliably.\n\n"
                f"**Grounding:** The closest retrieved sources were {closest}."
            )
        }

    def _deterministic_chat_response(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
    ) -> str | None:
        lowered = query.casefold()
        asks_for_forecast = any(
            word in lowered for word in ("forecast", "predict", "project")
        )
        if not asks_for_forecast or "revenue" not in lowered:
            return None

        try:
            return self._forecast_revenue_response(agent_input, query)
        except Exception:
            return None

    def _forecast_revenue_response(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
    ) -> str | None:
        df = self._read(agent_input.filePath)
        year_column = self._column(df, "Year")
        price_column = self._column(df, "Price_USD")
        volume_column = self._column(df, "Sales_Volume")
        if not year_column or not price_column or not volume_column:
            return None

        match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if not match:
            return None
        target_year = int(match.group(1))

        working = df[[year_column, price_column, volume_column]].copy()
        region_column = self._column(df, "Region")
        region = (
            self._query_category(df, region_column, query)
            if region_column
            else None
        )
        if region_column and region:
            working[region_column] = df[region_column]
            working = working[
                working[region_column].astype(str).str.casefold()
                == region.casefold()
            ]

        working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
        working[price_column] = pd.to_numeric(working[price_column], errors="coerce")
        working[volume_column] = pd.to_numeric(working[volume_column], errors="coerce")
        working = working.dropna(subset=[year_column, price_column, volume_column])
        if working.empty:
            return None

        working["revenue"] = working[price_column] * working[volume_column]
        grouped = working.groupby(year_column)["revenue"].sum().sort_index()
        grouped = grouped[grouped.index.astype(int) == grouped.index]
        if len(grouped) < 4:
            return None

        years = [int(year) for year in grouped.index]
        last_year = max(years)
        if target_year <= last_year:
            return None

        values = grouped.astype(float).to_numpy()
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        steps_ahead = target_year - last_year
        prediction = float(slope * (len(values) - 1 + steps_ahead) + intercept)

        region_text = f" for {region}" if region else ""
        filter_text = f", filtered to `Region = {region}`" if region else ""
        return (
            f"**Answer:** The forecasted total revenue{region_text} in "
            f"{target_year} is **{self._display_currency(prediction)}**, using "
            "a linear trend on annual revenue.\n\n"
            f"**Grounding:** Dataset `{agent_input.fileName}`; revenue = "
            f"`{price_column} * {volume_column}`{filter_text}, using "
            f"`{year_column}` {min(years)}-{last_year}."
        )

    def _create_chains(self) -> None:
        if self._dashboard_chain is not None:
            return

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from the environment.")

        llm = ChatGroq(
            model=MODEL_NAME,
            api_key=SecretStr(api_key),
            temperature=0,
            max_tokens=1000,
            timeout=120,
            max_retries=1,
        )
        chat_llm = ChatGroq(
            model=MODEL_NAME,
            api_key=SecretStr(api_key),
            temperature=0,
            max_tokens=CHAT_MAX_TOKENS,
            timeout=120,
            max_retries=1,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Use only the supplied dataset profile. Never invent values,
trends, anomalies or forecasts. Return concise business narrative only.""",
                ),
                (
                    "human",
                    """Description: {description}
Profile: {profile}

Return a title, executive summary, business summary, up to five findings,
three opportunities, three limitations and three recommended actions.""",
                ),
            ]
        )
        self._dashboard_chain = prompt | llm.with_structured_output(
            Narrative, method="function_calling"
        )

        rag_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer only from the supplied calculated evidence and retrieved dataset context.
Do not invent values, rows, trends, forecasts or categories.
Deterministic calculated evidence takes priority over retrieved evidence.
If evidence is insufficient, say so directly.
Do not use outside knowledge to make claims about the uploaded dataset.
Keep the response concise.
Cite retrieved sources using their source IDs.
Distinguish historical values from forecast values.
Do not claim that correlation proves causation.

Format exactly:
**Answer:** Direct answer in one to four sentences.

**Grounding:** Mention the calculation fields and/or retrieved source IDs that support the answer.""",
                ),
                (
                    "human",
                    "History: {history}\nEvidence:\n{context}\n\nQuestion: {query}",
                ),
            ]
        )
        self._rag_chat_chain = rag_chat_prompt | chat_llm | StrOutputParser()

        profile_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer only from the supplied compact dataset profile.
Return short Markdown under 90 words.

Format exactly:
**Answer:** 1-2 direct sentences.

**Grounding:** Mention the compact profile fields that support the answer.

If the compact profile does not contain enough evidence, say that directly.
Do not include raw JSON, long lists, tables, code blocks, or unsupported recommendations.""",
                ),
                (
                    "human",
                    "Compact profile: {profile}\nHistory: {history}\nQuestion: {query}",
                ),
            ]
        )
        self._profile_chat_chain = profile_chat_prompt | chat_llm | StrOutputParser()

    def _profile(self, agent_input: BusinessIntelligenceAgentInput) -> dict[str, Any]:
        df = self._read(agent_input.filePath)
        date_field, dates = self._date_field(df)
        if date_field and dates is not None:
            df = df.copy()
            df[date_field] = dates

        measures = [
            str(column)
            for column in df.select_dtypes(include="number").columns
            if not self._identifier(df, str(column))
        ][:8]
        dimensions = [
            str(column)
            for column in df.columns
            if str(column) not in measures and str(column) != date_field
        ][:10]

        missing = int(df.isna().sum().sum())
        cells = len(df) * len(df.columns)
        return {
            "summary": {
                "fileName": agent_input.fileName,
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "timeField": date_field,
                "period": self._period(df, date_field),
                "measures": measures,
                "dimensions": dimensions,
                "quality": {
                    "completenessPercent": (
                        round((cells - missing) / cells * 100, 2) if cells else 100.0
                    ),
                    "missingValueCount": missing,
                    "duplicateRowCount": int(df.duplicated().sum()),
                },
                "generatedAt": self._now(),
            },
            "metrics": self._metrics(df, measures, date_field),
            "bar": self._bar_data(df, dimensions, measures),
            "donut": self._donut_data(df, dimensions),
            "timeline": self._timeline_data(df, date_field, measures),
        }

    def _response(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
        narrative: Narrative,
    ) -> DashboardResponse:
        kpis = self._kpis(profile["metrics"])
        charts = self._charts(profile)
        timeline = self._timeline(profile["timeline"])
        success = len(kpis) >= 4 and len(charts) >= 2
        source_ids = [item["id"] for item in [*kpis, *charts]][:3]

        limitations = list(narrative.limitations)
        warnings = []
        if not success:
            message = (
                "Not enough valid measures or grouped data for a complete dashboard."
            )
            limitations.append(message)
            warnings.append(
                {
                    "code": "PARTIAL_DASHBOARD",
                    "message": message,
                    "component": "dashboard",
                    "recoverable": True,
                }
            )

        result = {
            "status": "success" if success else "partial",
            "sessionId": agent_input.sessionId,
            "warnings": warnings,
            "errors": [],
            "dashboard": {
                "title": narrative.title,
                "executiveSummary": narrative.executiveSummary,
                "kpis": kpis,
                "timeline": timeline,
                "supportingCharts": charts,
                "analysis": {
                    "businessSummary": narrative.businessSummary,
                    "keyFindings": narrative.keyFindings,
                },
                "insights": {
                    "criticalAnomalies": [],
                    "warnings": [],
                    "limitations": self._insights(
                        limitations, "limitation", "warning", source_ids
                    ),
                    "opportunities": self._insights(
                        narrative.opportunities,
                        "opportunity",
                        "info",
                        source_ids,
                    ),
                },
                "recommendedActions": [
                    {
                        "id": f"action_{index}",
                        "title": action.title,
                        "description": action.description,
                        "priority": action.priority,
                        "sourceIds": source_ids,
                    }
                    for index, action in enumerate(narrative.actions, 1)
                ],
                "datasetSummary": profile["summary"],
                "sections": [
                    {
                        "id": "kpis",
                        "title": "Key Performance Indicators",
                        "order": 1,
                        "visible": True,
                    },
                    {
                        "id": "timeline",
                        "title": "Timeline",
                        "order": 2,
                        "visible": timeline is not None,
                    },
                    {
                        "id": "supportingCharts",
                        "title": "Supporting Charts",
                        "order": 3,
                        "visible": bool(charts),
                    },
                    {
                        "id": "details",
                        "title": "Business Details",
                        "order": 4,
                        "visible": True,
                    },
                ],
                "layout": {
                    "kpis": {"columns": min(max(len(kpis), 1), 4), "maxRows": 2},
                    "timeline": {"columnSpan": 12},
                    "supportingCharts": {"columns": 2, "maxRows": 2},
                    "details": {"columns": 2, "maxRows": 2},
                },
            },
        }
        return DashboardResponse.model_validate(result)

    def _metrics(
        self, df: pd.DataFrame, measures: list[str], date_field: str | None
    ) -> list[dict[str, Any]]:
        output = []
        for name in measures:
            values = pd.to_numeric(df[name], errors="coerce").dropna()
            if values.empty:
                continue

            change = None
            if date_field:
                working = df[[date_field, name]].dropna().copy()
                if not working.empty:
                    _, code = self._grain(working[date_field])
                    working["period"] = working[date_field].dt.to_period(code)
                    aggregation = "mean" if self._average(name) else "sum"
                    grouped = working.groupby("period")[name].agg(aggregation)
                    if len(grouped) >= 2 and float(grouped.iloc[-2]) != 0:
                        change = round(
                            (float(grouped.iloc[-1]) - float(grouped.iloc[-2]))
                            / abs(float(grouped.iloc[-2]))
                            * 100,
                            2,
                        )

            output.append(
                {
                    "name": name,
                    "sum": round(float(values.sum()), 2),
                    "average": round(float(values.mean()), 2),
                    "change": change,
                }
            )
        return output

    def _bar_data(
        self, df: pd.DataFrame, dimensions: list[str], measures: list[str]
    ) -> dict[str, Any] | None:
        for dimension in dimensions:
            if not 2 <= df[dimension].nunique(dropna=True) <= 30:
                continue
            for measure in measures:
                aggregation = "mean" if self._average(measure) else "sum"
                grouped = (
                    df.dropna(subset=[dimension, measure])
                    .groupby(dimension)[measure]
                    .agg(aggregation)
                    .sort_values(ascending=False)
                    .head(6)
                )
                if not grouped.empty:
                    return {
                        "dimension": dimension,
                        "measure": measure,
                        "aggregation": aggregation,
                        "values": [
                            {"label": str(label), "value": round(float(value), 2)}
                            for label, value in grouped.items()
                        ],
                    }
        return None

    def _donut_data(
        self, df: pd.DataFrame, dimensions: list[str]
    ) -> dict[str, Any] | None:
        for dimension in dimensions:
            if 2 <= df[dimension].nunique(dropna=True) <= 12:
                counts = (
                    df[dimension].fillna("Missing").astype(str).value_counts().head(6)
                )
                return {
                    "dimension": dimension,
                    "values": [
                        {"label": str(label), "value": int(value)}
                        for label, value in counts.items()
                    ],
                }
        return None

    def _timeline_data(
        self,
        df: pd.DataFrame,
        date_field: str | None,
        measures: list[str],
    ) -> dict[str, Any] | None:
        if not date_field or not measures:
            return None
        dates = df[date_field].dropna()
        if dates.empty:
            return None

        granularity, code = self._grain(dates)
        measure = measures[0]
        aggregation = "mean" if self._average(measure) else "sum"
        working = df[[date_field, measure]].dropna().copy()
        working["period"] = working[date_field].dt.to_period(code)
        grouped = working.groupby("period")[measure].agg(aggregation).tail(18)
        if grouped.empty:
            return None

        values = grouped.astype(float).to_numpy()
        anomalies = []
        standard_deviation = float(values.std())
        if len(values) >= 4 and standard_deviation > 0:
            mean = float(values.mean())
            for index, (period, value) in enumerate(grouped.items(), 1):
                score = abs((float(value) - mean) / standard_deviation)
                if score >= 2:
                    anomalies.append(
                        {
                            "id": f"anomaly_{index}",
                            "period": str(period),
                            "label": str(period),
                            "value": round(float(value), 2),
                            "severity": "critical" if score >= 3 else "warning",
                            "reason": f"Value is {score:.1f} standard deviations from the timeline mean.",
                        }
                    )

        forecast = []
        if len(values) >= 4:
            x = np.arange(len(values), dtype=float)
            slope, intercept = np.polyfit(x, values, 1)
            residual_spread = float(np.std(values - (slope * x + intercept))) * 1.96
            for step in range(1, 4):
                prediction = float(slope * (len(values) - 1 + step) + intercept)
                period = str(grouped.index[-1] + step)
                forecast.append(
                    {
                        "period": period,
                        "label": period,
                        "value": round(prediction, 2),
                        "lowerBound": round(prediction - residual_spread, 2),
                        "upperBound": round(prediction + residual_spread, 2),
                    }
                )

        return {
            "measure": measure,
            "aggregation": aggregation,
            "granularity": granularity,
            "points": [
                {
                    "period": str(period),
                    "label": str(period),
                    "value": round(float(value), 2),
                }
                for period, value in grouped.items()
            ],
            "anomalies": anomalies,
            "forecast": forecast,
        }

    def _kpis(self, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for metric in metrics[:8]:
            average = self._average(metric["name"])
            value = metric["average"] if average else metric["sum"]
            change = metric.get("change")
            kind = "note" if change in (None, 0) else "increase" if change > 0 else "decrease"
            text = (
                "No previous-period comparison"
                if change is None
                else "No change from previous period"
                if change == 0
                else f"{abs(change):.1f}% vs previous period"
            )
            output.append(
                {
                    "id": f"kpi_{self._slug(metric['name'])}",
                    "title": f"{'Average' if average else 'Total'} {self._title(metric['name'])}",
                    "value": self._display(metric["name"], value),
                    "rawValue": value,
                    "indicator": {"kind": kind, "text": text},
                }
            )
        return output

    def _charts(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        charts = []
        bar = profile["bar"]
        if bar:
            charts.append(
                {
                    "id": "chart_bar",
                    "type": "bar",
                    "title": f"{self._title(bar['measure'])} by {self._title(bar['dimension'])}",
                    "subtitle": f"{bar['aggregation'].title()} aggregation",
                    "valueFormat": self._format(bar["measure"]),
                    "categories": [item["label"] for item in bar["values"]],
                    "series": [
                        {
                            "id": "series_bar",
                            "name": self._title(bar["measure"]),
                            "data": [item["value"] for item in bar["values"]],
                        }
                    ],
                    "layout": {"columnSpan": 1, "rowSpan": 1},
                }
            )

        donut = profile["donut"]
        if donut:
            charts.append(
                {
                    "id": "chart_donut",
                    "type": "donut",
                    "title": f"Distribution by {self._title(donut['dimension'])}",
                    "subtitle": None,
                    "valueFormat": "number",
                    "segments": [
                        {
                            "id": f"segment_{index}",
                            "label": item["label"],
                            "value": item["value"],
                        }
                        for index, item in enumerate(donut["values"], 1)
                    ],
                    "layout": {"columnSpan": 1, "rowSpan": 1},
                }
            )
        return charts

    def _timeline(self, item: dict[str, Any] | None) -> dict[str, Any] | None:
        if not item:
            return None
        forecast = item["forecast"]
        return {
            "id": "timeline_main",
            "title": f"{self._title(item['measure'])} over time",
            "subtitle": f"{item['aggregation'].title()} by {item['granularity']}",
            "granularity": item["granularity"],
            "unit": None,
            "valueFormat": self._format(item["measure"]),
            "actual": item["points"],
            "anomalies": item["anomalies"],
            "forecast": forecast,
            "forecastMetadata": {
                "available": bool(forecast),
                "model": "linear_trend" if forecast else None,
                "horizon": len(forecast),
                "horizonUnit": item["granularity"],
                "target": item["measure"],
                "confidenceLevel": 0.95 if forecast else None,
            },
        }

    def _fallback(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
    ) -> Narrative:
        summary = profile["summary"]
        return Narrative(
            title=f"Business Intelligence Dashboard — {agent_input.fileName}",
            executiveSummary=f"The dataset contains {summary['rowCount']:,} rows and {summary['columnCount']} columns.",
            businessSummary="The dashboard summarises the main measures, categories and time-based patterns in the dataset.",
            keyFindings=[
                f"{len(summary['measures'])} numerical measures were detected.",
                f"Data completeness is {summary['quality']['completenessPercent']:.2f}%.",
            ],
            limitations=[
                "The AI narrative failed, so a deterministic summary was used."
            ],
        )

    @staticmethod
    def _insights(
        values: list[str],
        prefix: str,
        severity: Literal["info", "warning", "critical"],
        source_ids: list[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": f"{prefix}_{index}",
                "title": value[:80],
                "description": value,
                "severity": severity,
                "sourceIds": source_ids,
            }
            for index, value in enumerate(values[:4], 1)
            if value.strip()
        ]

    @staticmethod
    def _read(file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        raise ValueError("Only CSV, Excel and JSON files are supported.")

    @staticmethod
    def _date_field(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
        for column in df.columns:
            name = str(column)
            if not any(
                word in name.lower()
                for word in ("date", "time", "year", "month", "period")
            ):
                continue
            source = df[column].astype(str)
            if "year" in name.lower():
                source = source + "-01-01"
            parsed = pd.to_datetime(source, errors="coerce")
            if len(parsed) and parsed.notna().mean() >= 0.6:
                return name, parsed
        return None, None

    @staticmethod
    def _period(df: pd.DataFrame, date_field: str | None) -> dict[str, str] | None:
        if not date_field:
            return None
        values = df[date_field].dropna()
        if values.empty:
            return None
        start, end = values.min(), values.max()
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": f"{start:%Y-%m-%d} – {end:%Y-%m-%d}",
        }

    @staticmethod
    def _identifier(df: pd.DataFrame, column: str) -> bool:
        name = column.lower()
        looks_like_id = (
            name == "id"
            or name.endswith("_id")
            or any(word in name for word in ("code", "reference", "number"))
        )
        return looks_like_id and (
            len(df) == 0 or df[column].nunique(dropna=True) / len(df) >= 0.5
        )

    @staticmethod
    def _grain(dates: pd.Series) -> tuple[str, str]:
        days = int((dates.max() - dates.min()).days)
        if days > 730:
            return "year", "Y"
        if days > 120:
            return "month", "M"
        if days > 30:
            return "week", "W"
        return "day", "D"

    @classmethod
    def _display(cls, name: str, value: float) -> str:
        absolute = abs(value)
        divisor, suffix = (
            (1_000_000_000, "B")
            if absolute >= 1_000_000_000
            else (1_000_000, "M")
            if absolute >= 1_000_000
            else (1_000, "K")
            if absolute >= 1_000
            else (1, "")
        )
        text = f"{value / divisor:,.2f}{suffix}"
        lowered = name.lower()
        if cls._format(name) == "currency":
            symbol = "£" if "gbp" in lowered else "€" if "eur" in lowered else "$" if "usd" in lowered else ""
            return f"{symbol}{text}"
        return f"{text}%" if cls._format(name) == "percentage" else text

    @classmethod
    def _display_currency(cls, value: float) -> str:
        return cls._display("revenue_usd", value)

    @staticmethod
    def _column(df: pd.DataFrame, name: str) -> str | None:
        expected = name.casefold()
        for column in df.columns:
            if str(column).casefold() == expected:
                return str(column)
        return None

    @staticmethod
    def _query_category(
        df: pd.DataFrame,
        column: str | None,
        query: str,
    ) -> str | None:
        if not column:
            return None

        lowered = query.casefold()
        values = sorted(
            (str(value) for value in df[column].dropna().unique()),
            key=len,
            reverse=True,
        )
        for value in values:
            if re.search(rf"\b{re.escape(value.casefold())}\b", lowered):
                return value
        return None

    @staticmethod
    def _average(name: str) -> bool:
        return any(
            word in name.lower()
            for word in (
                "price",
                "rate",
                "percent",
                "margin",
                "average",
                "avg",
                "score",
            )
        )

    @staticmethod
    def _format(name: str) -> str:
        value = name.lower()
        if any(word in value for word in ("percent", "rate", "margin")):
            return "percentage"
        if any(
            word in value for word in ("price", "revenue", "cost", "profit", "amount")
        ):
            return "currency"
        return "number"

    @staticmethod
    def _title(value: str) -> str:
        return value.replace("_", " ").replace("-", " ").strip().title()

    @staticmethod
    def _slug(value: str) -> str:
        return "_".join(value.lower().replace("-", " ").split())

    @staticmethod
    def _history_text(history: list[dict[str, str]]) -> str:
        return (
            "None"
            if not history
            else "\n".join(
                f"{item['role']}: {item['content']}" for item in history[-6:]
            )
        )

    def _profile_based_chat_fallback(self, state: AgentState) -> str | None:
        try:
            self._create_chains()
            response = self._profile_chat_chain.invoke(
                {
                    "profile": compact_profile_for_chat(state["profile"]),
                    "history": self._history_text(state.get("history", [])),
                    "query": state["query"],
                }
            )
            return response.strip()
        except Exception:
            logger.exception(
                "Compact profile fallback failed session_id=%s",
                state["agent_input"].sessionId,
            )
            return None

    @staticmethod
    def _source_ids(
        documents: list[RetrievedDocument] | list[RerankedDocument],
    ) -> list[str]:
        output: list[str] = []
        for document in documents[:5]:
            source_id = document.metadata.get("source_id")
            if isinstance(source_id, str) and source_id not in output:
                output.append(source_id)
        return output

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


business_intelligence_agent = BusinessIntelligenceAgent()
