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

Set the pipeline mode in `config/agents.toml`:

```toml
[pipeline]
# Change this to "single" for the existing single-agent dashboard and chat workflow.
mode = "multi"
```

`multi` is the checked-in default. The two modes expose
the same upload, dashboard, chat, and chat-history API contracts.

Each `agents.<name>` section selects the provider, model, generation limits,
and reasoning effort for one LLM invocation. Each LLM agent has one versioned
TOON bundle in `app/prompts/`; the backend validates the bundle at startup and
serializes its structured system and user context as TOON before invocation.
Mode and model settings are deliberately not read from `.env`.
The `[forecasting]` table configures the TimesFM model and its limits.
Keep `GROQ_API_KEY`, Supabase credentials, and other secrets in `.env` only.

The multi-agent analysis flow is:

```text
Upload -> Generic Cleaning -> Data Preparation -> Orchestrator
       -> capability-gated KPI/Trend, Anomaly, and Forecast specialists
       -> Specialist Join -> Insight Synthesis
       -> Dashboard Generation ----\
       -> Retrieval Preparation -----> Output Join
                                      -> Service-owned Retrieval Indexing
                                      -> Service-owned Persistence -> END
```

RAG model assignments, embedding and reranking limits, retrieval thresholds,
and document chunking settings live in `config/rag.toml`. Both checked-in TOML
files are validated when the API starts, so invalid settings fail early with a
configuration error.

Dashboard generation and retrieval preparation must both report completion or
failure before the graph's output join runs. The top-level business intelligence
service then owns retrieval indexing, dashboard/workflow persistence, and the
final dataset status update for both pipeline modes. Optional specialist and
retrieval failures produce a partial dashboard with warnings; cleaning,
preparation, dashboard, or persistence failures produce a failed result.

Multi-agent chat uses a separate pipeline:

```text
Session Validation -> Input Guardrail -> Session-Filtered Retrieval
                   -> Chat Agent -> Output Grounding Guardrail -> Chat Response
```

## Tests

```powershell
pytest -q
```

## Orchestration

USER UPLOAD
    │
    ▼
Generic Cleaning Service
    │
    ▼
Data Preparation Agent
    │
    ▼
Orchestrator Agent
    │
    ├──────────────┬──────────────────┐
    ▼              ▼                  ▼
KPI & Trend     Anomaly Detection   Forecasting
Agent           Agent               Agent
    │              │                  │
    └──────────────┴──────────────────┘
                   │
                   ▼
             Specialist Join
                   │
                   ▼
         Insight Synthesis Agent
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
Dashboard Generation   Retrieval Preparation
Agent                  Agent
         │                    │
         ▼                    ▼
 Dashboard JSON       RAG Documents / Chunks
         │                    │
         └─────────┬──────────┘
                   ▼
          Supabase Persistence
