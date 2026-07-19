# Business Intelligence Backend

## Run

```powershell
uvicorn app.main:app --reload
```

Copy `.env.sample` to `.env` and set the Supabase service-role key. Apply
`scripts/db.sql` in the Supabase SQL editor before starting the API. The script
recreates application data tables, creates a profile for each Supabase Auth
account, and leaves the Supabase-managed `auth.users` records intact.

For an existing single-dataset installation, apply
`scripts/migrate_multi_dataset_sessions.sql` instead. It retains existing
records and stored files, marks those legacy workspaces as reset-required, and
lets each owner remove them through Start Over before uploading a new batch.

For an existing database created with an earlier version, apply
`scripts/migrate_atomic_rag_index.sql` once. It installs the transactional
vector-index replacement function without deleting existing application data.

All `/api/upload`, `/api/dashboard/{session_id}`, `/api/chat`, and
`/api/chat/{session_id}/history` requests require an `Authorization: Bearer
<Supabase access JWT>` header. The backend validates the JWT and returns data
only when the session belongs to its authenticated user.

`POST /api/upload` accepts one to five repeated multipart `files` fields.
Different schemas are prepared independently and synthesized into one
session-scoped dashboard and retrieval index. `GET /api/dataset` returns the
workspace plus its `datasets[]`; previews use
`GET /api/dataset/preview?dataset_id=<uuid>&page=1&page_size=50`.
Additional files can be appended with `POST /api/dataset` using the same
repeated `files` fields, up to five total datasets. Remove one with
`DELETE /api/dataset/{dataset_id}`. Both operations rebuild the dashboard and
retrieval index, and clear chat history that was grounded in the previous file
set. Removing the last dataset deletes the workspace.
Chat questions use every dataset in the active workspace by default; naming a
file explicitly narrows the answer to that dataset.

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
Keep API keys, Supabase credentials, and other secrets in `.env` only.

## LLM providers

Groq remains the checked-in default. Every LLM agent can independently select
`groq` or `openrouter` in `config/agents.toml`, so the pipeline can use one
provider throughout or mix providers by workload:

```toml
[agents.chat]
provider = "openrouter"
model = "openai/gpt-oss-120b"
temperature = 0.1
max_completion_tokens = 600
reasoning_effort = "low"
strict_json_schema = true
```

Use a model identifier available from the selected provider. Configure only
the credentials needed by the active policies:

```dotenv
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

Changing `provider` does not change the agent prompts, response schemas,
deterministic validation, fallback behavior, or API contracts.

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
Session Validation -> Input Guardrail -> History-Aware Session Retrieval
                   -> Cross-Encoder Reranking -> Chat Agent
                   -> Output Grounding Guardrail -> Chat Response
```

Both pipeline modes use the same configured chunker and transactional Supabase
index replacement. Multi-agent retrieval combines compact analytical findings
with bounded prepared-row batches so detailed lookups are not limited to
dashboard summaries. Recent conversation history is used to resolve follow-up
references, but only retrieved documents are accepted as factual evidence.

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
