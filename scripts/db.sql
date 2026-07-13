-- =========================================================
-- Extensions
-- =========================================================

create extension if not exists vector
with schema extensions;


-- =========================================================
-- 1. DATASETS
-- Stores uploaded-file metadata and processing state.
-- The dataset ID is also the existing sessionId.
-- =========================================================

create table public.datasets (
    id uuid primary key default gen_random_uuid(),

    file_name text not null,
    storage_path text not null unique,
    mime_type text not null,
    file_size bigint not null
        check (file_size >= 0),

    file_hash text not null,
    description text,

    status text not null default 'processing'
        check (
            status in (
                'processing',
                'ready',
                'failed'
            )
        ),

    rag_status text not null default 'pending'
        check (
            rag_status in (
                'pending',
                'indexing',
                'ready',
                'failed'
            )
        ),

    error_message text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);


-- =========================================================
-- 2. DASHBOARDS
-- Stores the complete validated DashboardResponse.
-- One current dashboard is stored per dataset.
-- =========================================================

create table public.dashboards (
    id uuid primary key default gen_random_uuid(),

    dataset_id uuid not null unique
        references public.datasets(id)
        on delete cascade,

    status text not null
        check (
            status in (
                'success',
                'partial',
                'failed'
            )
        ),

    response jsonb not null,

    generated_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);


-- =========================================================
-- 3. MESSAGES
-- Stores one chat history per dataset.
-- No conversations table is needed yet.
-- =========================================================

create table public.messages (
    id uuid primary key default gen_random_uuid(),

    dataset_id uuid not null
        references public.datasets(id)
        on delete cascade,

    role text not null
        check (
            role in (
                'user',
                'assistant'
            )
        ),

    content text not null
        check (
            length(trim(content)) > 0
        ),

    -- Optional RAG source IDs used to generate the response.
    sources jsonb not null default '[]'::jsonb,

    created_at timestamptz not null default now()
);


-- =========================================================
-- 4. DOCUMENT CHUNKS
-- Stores RAG text, metadata and vector embeddings.
-- =========================================================

create table public.document_chunks (
    id uuid primary key default gen_random_uuid(),

    dataset_id uuid not null
        references public.datasets(id)
        on delete cascade,

    source_id text not null,
    document_type text not null,

    chunk_index integer not null default 0
        check (chunk_index >= 0),

    content text not null
        check (
            length(trim(content)) > 0
        ),

    metadata jsonb not null default '{}'::jsonb,

    -- all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    embedding extensions.vector(384) not null,

    created_at timestamptz not null default now(),

    unique (
        dataset_id,
        source_id,
        chunk_index
    )
);


-- =========================================================
-- Normal indexes
-- =========================================================

create index messages_dataset_created_idx
on public.messages (
    dataset_id,
    created_at desc
);

create index document_chunks_dataset_idx
on public.document_chunks (
    dataset_id
);

create index document_chunks_type_idx
on public.document_chunks (
    dataset_id,
    document_type
);


-- =========================================================
-- Vector similarity index
-- Uses cosine distance.
-- =========================================================

create index document_chunks_embedding_hnsw_idx
on public.document_chunks
using hnsw (
    embedding extensions.vector_cosine_ops
);


-- =========================================================
-- Automatically maintain updated_at fields.
-- =========================================================

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;


create trigger datasets_set_updated_at
before update on public.datasets
for each row
execute function public.set_updated_at();


create trigger dashboards_set_updated_at
before update on public.dashboards
for each row
execute function public.set_updated_at();


-- =========================================================
-- RAG similarity-search function
-- Call this from FastAPI through supabase.rpc(...).
-- =========================================================

create or replace function public.match_document_chunks(
    p_dataset_id uuid,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 5,
    p_match_threshold double precision default 0
)
returns table (
    id uuid,
    source_id text,
    document_type text,
    content text,
    metadata jsonb,
    similarity double precision
)
language sql
stable
set search_path = public, extensions
as $$
    select
        dc.id,
        dc.source_id,
        dc.document_type,
        dc.content,
        dc.metadata,
        1 - (
            dc.embedding <=> p_query_embedding
        ) as similarity
    from public.document_chunks dc
    where dc.dataset_id = p_dataset_id
      and (
          1 - (
              dc.embedding <=> p_query_embedding
          )
      ) >= p_match_threshold
    order by
        dc.embedding <=> p_query_embedding
    limit least(
        greatest(p_match_count, 1),
        50
    );
$$;


-- =========================================================
-- Security until authentication is added.
--
-- No public RLS policies are created.
-- Only the trusted FastAPI backend using the Supabase
-- service-role key should access these tables.
-- =========================================================

alter table public.datasets
enable row level security;

alter table public.dashboards
enable row level security;

alter table public.messages
enable row level security;

alter table public.document_chunks
enable row level security;
