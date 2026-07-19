-- Clean-install Supabase schema for multi-dataset analysis workspaces.
-- This recreates application tables but leaves auth.users and Storage intact.

create extension if not exists vector with schema extensions;

drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.match_session_document_chunks(
    uuid, extensions.vector, integer, double precision
);
drop function if exists public.replace_session_document_chunks(uuid, jsonb);
drop function if exists public.match_document_chunks(
    uuid, extensions.vector, integer, double precision
);
drop function if exists public.replace_document_chunks(uuid, jsonb);
drop function if exists public.create_profile_for_new_user();
drop function if exists public.set_updated_at();

drop table if exists public.session_processing cascade;
drop table if exists public.document_chunks cascade;
drop table if exists public.messages cascade;
drop table if exists public.dashboards cascade;
drop table if exists public.datasets cascade;
drop table if exists public.analysis_sessions cascade;
drop table if exists public.profiles cascade;

create table public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    email text not null check (length(trim(email)) > 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.analysis_sessions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null unique references public.profiles(id) on delete cascade,
    description text,
    status text not null default 'processing'
        check (status in ('processing', 'ready', 'failed')),
    rag_status text not null default 'pending'
        check (rag_status in ('pending', 'indexing', 'ready', 'failed')),
    error_message text,
    requires_reset boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.datasets (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.analysis_sessions(id) on delete cascade,
    user_id uuid not null references public.profiles(id) on delete cascade,
    file_name text not null,
    storage_path text not null unique,
    mime_type text not null,
    file_size bigint not null check (file_size >= 0),
    file_hash text not null,
    description text,
    row_count integer not null check (row_count >= 0),
    column_count integer not null check (column_count >= 0),
    -- Retained for compatibility; analysis_sessions is authoritative.
    status text not null default 'processing'
        check (status in ('processing', 'ready', 'failed')),
    rag_status text not null default 'pending'
        check (rag_status in ('pending', 'indexing', 'ready', 'failed')),
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (session_id, file_hash)
);

create table public.dashboards (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null unique
        references public.analysis_sessions(id) on delete cascade,
    status text not null check (status in ('success', 'partial', 'failed')),
    response jsonb not null,
    generated_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.messages (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.analysis_sessions(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null check (length(trim(content)) > 0),
    sources jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create table public.document_chunks (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.analysis_sessions(id) on delete cascade,
    dataset_id uuid not null references public.datasets(id) on delete cascade,
    source_id text not null,
    document_type text not null,
    chunk_index integer not null default 0 check (chunk_index >= 0),
    content text not null check (length(trim(content)) > 0),
    metadata jsonb not null default '{}'::jsonb,
    embedding extensions.vector(384) not null,
    created_at timestamptz not null default now(),
    unique (session_id, source_id, chunk_index)
);

create table public.session_processing (
    session_id uuid primary key references public.analysis_sessions(id) on delete cascade,
    workflow_status text not null check (workflow_status in ('success', 'partial', 'failed')),
    generic_cleaning_report jsonb not null default '{}'::jsonb,
    prepared_dataset jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index datasets_session_created_idx
    on public.datasets (session_id, created_at);
create index datasets_user_created_idx
    on public.datasets (user_id, created_at desc);
create index messages_session_created_idx
    on public.messages (session_id, created_at desc);
create index document_chunks_session_idx
    on public.document_chunks (session_id);
create index document_chunks_dataset_idx
    on public.document_chunks (dataset_id);
create index document_chunks_type_idx
    on public.document_chunks (session_id, document_type);
create index document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger profiles_set_updated_at before update on public.profiles
for each row execute function public.set_updated_at();
create trigger analysis_sessions_set_updated_at before update on public.analysis_sessions
for each row execute function public.set_updated_at();
create trigger datasets_set_updated_at before update on public.datasets
for each row execute function public.set_updated_at();
create trigger dashboards_set_updated_at before update on public.dashboards
for each row execute function public.set_updated_at();
create trigger session_processing_set_updated_at before update on public.session_processing
for each row execute function public.set_updated_at();

create or replace function public.create_profile_for_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
    insert into public.profiles (id, email)
    values (new.id, coalesce(new.email, 'unknown@example.invalid'))
    on conflict (id) do update set email = excluded.email;
    return new;
end;
$$;

create trigger on_auth_user_created after insert on auth.users
for each row execute procedure public.create_profile_for_new_user();

insert into public.profiles (id, email)
select id, coalesce(email, 'unknown@example.invalid')
from auth.users
on conflict (id) do update set email = excluded.email;

alter table public.profiles enable row level security;
alter table public.analysis_sessions enable row level security;
alter table public.datasets enable row level security;
alter table public.dashboards enable row level security;
alter table public.messages enable row level security;
alter table public.document_chunks enable row level security;
alter table public.session_processing enable row level security;

create policy profiles_select_own on public.profiles
for select to authenticated using (auth.uid() = id);
create policy analysis_sessions_owner_access on public.analysis_sessions
for all to authenticated using (auth.uid() = user_id)
with check (auth.uid() = user_id);
create policy datasets_owner_access on public.datasets
for all to authenticated using (auth.uid() = user_id)
with check (
    auth.uid() = user_id
    and exists (
        select 1 from public.analysis_sessions s
        where s.id = datasets.session_id and s.user_id = auth.uid()
    )
);
create policy dashboards_owner_access on public.dashboards
for all to authenticated using (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = dashboards.session_id and s.user_id = auth.uid()
    )
) with check (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = dashboards.session_id and s.user_id = auth.uid()
    )
);
create policy messages_owner_access on public.messages
for all to authenticated using (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = messages.session_id and s.user_id = auth.uid()
    )
) with check (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = messages.session_id and s.user_id = auth.uid()
    )
);
create policy document_chunks_owner_access on public.document_chunks
for all to authenticated using (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = document_chunks.session_id and s.user_id = auth.uid()
    )
) with check (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = document_chunks.session_id and s.user_id = auth.uid()
    )
);
create policy session_processing_owner_access on public.session_processing
for all to authenticated using (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = session_processing.session_id and s.user_id = auth.uid()
    )
) with check (
    exists (
        select 1 from public.analysis_sessions s
        where s.id = session_processing.session_id and s.user_id = auth.uid()
    )
);

create or replace function public.replace_session_document_chunks(
    p_session_id uuid,
    p_chunks jsonb
)
returns integer
language plpgsql volatile security invoker
set search_path = public, extensions as $$
declare inserted_count integer;
begin
    if jsonb_typeof(p_chunks) <> 'array' then
        raise exception 'p_chunks must be a JSON array';
    end if;
    delete from public.document_chunks where session_id = p_session_id;
    insert into public.document_chunks (
        session_id, dataset_id, source_id, document_type, chunk_index,
        content, metadata, embedding
    )
    select
        p_session_id,
        (chunk ->> 'dataset_id')::uuid,
        chunk ->> 'source_id',
        chunk ->> 'document_type',
        coalesce((chunk ->> 'chunk_index')::integer, 0),
        chunk ->> 'content',
        coalesce(chunk -> 'metadata', '{}'::jsonb),
        (chunk -> 'embedding')::text::extensions.vector
    from jsonb_array_elements(p_chunks) as chunk;
    get diagnostics inserted_count = row_count;
    return inserted_count;
end;
$$;

create or replace function public.match_session_document_chunks(
    p_session_id uuid,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 5,
    p_match_threshold double precision default 0
)
returns table (
    id uuid, session_id uuid, dataset_id uuid, source_id text,
    document_type text, content text, metadata jsonb, similarity double precision
)
language sql stable security invoker
set search_path = public, extensions as $$
    select
        dc.id, dc.session_id, dc.dataset_id, dc.source_id, dc.document_type,
        dc.content, dc.metadata,
        1 - (dc.embedding <=> p_query_embedding) as similarity
    from public.document_chunks dc
    where dc.session_id = p_session_id
      and 1 - (dc.embedding <=> p_query_embedding) >= p_match_threshold
    order by dc.embedding <=> p_query_embedding
    limit least(greatest(p_match_count, 1), 50);
$$;

grant usage on schema public to service_role;
grant all privileges on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
grant execute on all functions in schema public to service_role;
alter default privileges in schema public grant all privileges on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;
alter default privileges in schema public grant execute on functions to service_role;
