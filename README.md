# Business Intelligence Backend

## Run

```powershell
uvicorn app.main:app --reload
```

Copy `.env.sample` to `.env` and set the Supabase service-role key. Apply
`scripts/db.sql` in the Supabase SQL editor before starting the API. The script
recreates application data tables, creates a profile for each Supabase Auth
account, and leaves the Supabase-managed `auth.users` records intact.

All `/api/upload`, `/api/dashboard/{session_id}`, `/api/chat`, and
`/api/chat/{session_id}/history` requests require an `Authorization: Bearer
<Supabase access JWT>` header. The backend validates the JWT and returns data
only when the session belongs to its authenticated user.

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
