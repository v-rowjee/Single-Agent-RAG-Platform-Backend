# Business Intelligence Backend

## Run

```powershell
uvicorn app.main:app --reload
```

## Pipeline mode

Set `BI_PIPELINE_MODE` in `.env`:

```dotenv
# Existing single-agent dashboard and chat workflow
BI_PIPELINE_MODE=single

# Multi-agent analysis and session-scoped retrieval chat
BI_PIPELINE_MODE=multi
```

`single` remains the default when the variable is omitted. The two modes expose
the same upload, dashboard, chat, and chat-history API contracts.

The multi-agent analysis flow is:

```text
Upload -> Generic Cleaning -> Data Preparation -> Orchestrator
       -> capability-gated KPI/Trend, Anomaly, and Forecast specialists
       -> Specialist Join -> Insight Synthesis
       -> Dashboard Generation --------------------\
       -> Retrieval Preparation -> Retrieval Indexing
                                                    -> Output Join
                                                    -> Persistence -> END
```

Dashboard generation and retrieval indexing must both report completion or
failure before the output join runs. Optional specialist and retrieval failures
produce a partial dashboard with warnings; cleaning, preparation, dashboard, or
persistence failures produce a failed result.

Multi-agent chat uses a separate pipeline:

```text
Session Validation -> Input Guardrail -> Session-Filtered Retrieval
                   -> Chat Agent -> Output Grounding Guardrail -> Chat Response
```

## Tests

```powershell
pytest -q
```
