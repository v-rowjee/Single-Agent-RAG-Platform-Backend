-- Non-destructive migration from one dataset per user to one workspace with
-- many datasets. Existing workspaces are retained only so their owners can
-- remove them through Start Over; requires_reset blocks analysis endpoints.

begin;

create table if not exists public.analysis_sessions (
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

drop trigger if exists analysis_sessions_set_updated_at
on public.analysis_sessions;
create trigger analysis_sessions_set_updated_at
before update on public.analysis_sessions
for each row execute function public.set_updated_at();

insert into public.analysis_sessions (
    id, user_id, description, status, rag_status, error_message,
    requires_reset, created_at, updated_at
)
select
    id, user_id, description, status, rag_status, error_message,
    true, created_at, updated_at
from public.datasets
on conflict (id) do nothing;

alter table public.datasets add column if not exists session_id uuid;
update public.datasets set session_id = id where session_id is null;
alter table public.datasets alter column session_id set not null;
alter table public.datasets
    drop constraint if exists datasets_user_id_key;
alter table public.datasets
    drop constraint if exists datasets_session_id_fkey;
alter table public.datasets
    add constraint datasets_session_id_fkey
    foreign key (session_id) references public.analysis_sessions(id) on delete cascade;
alter table public.datasets
    drop constraint if exists datasets_session_file_hash_key;
alter table public.datasets
    add constraint datasets_session_file_hash_key unique (session_id, file_hash);
create index if not exists datasets_session_created_idx
    on public.datasets (session_id, created_at);

alter table public.dashboards
    drop constraint if exists dashboards_dataset_id_fkey;
alter table public.dashboards rename column dataset_id to session_id;
alter table public.dashboards
    add constraint dashboards_session_id_fkey
    foreign key (session_id) references public.analysis_sessions(id) on delete cascade;

alter table public.messages
    drop constraint if exists messages_dataset_id_fkey;
alter table public.messages rename column dataset_id to session_id;
alter table public.messages
    add constraint messages_session_id_fkey
    foreign key (session_id) references public.analysis_sessions(id) on delete cascade;

alter table public.session_processing
    drop constraint if exists session_processing_dataset_id_fkey;
alter table public.session_processing rename column dataset_id to session_id;
alter table public.session_processing
    add constraint session_processing_session_id_fkey
    foreign key (session_id) references public.analysis_sessions(id) on delete cascade;

alter table public.document_chunks add column if not exists session_id uuid;
update public.document_chunks set session_id = dataset_id where session_id is null;
alter table public.document_chunks alter column session_id set not null;
alter table public.document_chunks
    drop constraint if exists document_chunks_session_id_fkey;
alter table public.document_chunks
    add constraint document_chunks_session_id_fkey
    foreign key (session_id) references public.analysis_sessions(id) on delete cascade;
create index if not exists document_chunks_session_idx
    on public.document_chunks (session_id);

drop policy if exists dashboards_owner_access on public.dashboards;
drop policy if exists messages_owner_access on public.messages;
drop policy if exists document_chunks_owner_access on public.document_chunks;
drop policy if exists session_processing_owner_access on public.session_processing;
drop policy if exists analysis_sessions_owner_access on public.analysis_sessions;
drop policy if exists datasets_owner_access on public.datasets;

alter table public.analysis_sessions enable row level security;
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
    exists (select 1 from public.analysis_sessions s
            where s.id = dashboards.session_id and s.user_id = auth.uid())
) with check (
    exists (select 1 from public.analysis_sessions s
            where s.id = dashboards.session_id and s.user_id = auth.uid())
);
create policy messages_owner_access on public.messages
for all to authenticated using (
    exists (select 1 from public.analysis_sessions s
            where s.id = messages.session_id and s.user_id = auth.uid())
) with check (
    exists (select 1 from public.analysis_sessions s
            where s.id = messages.session_id and s.user_id = auth.uid())
);
create policy document_chunks_owner_access on public.document_chunks
for all to authenticated using (
    exists (select 1 from public.analysis_sessions s
            where s.id = document_chunks.session_id and s.user_id = auth.uid())
) with check (
    exists (select 1 from public.analysis_sessions s
            where s.id = document_chunks.session_id and s.user_id = auth.uid())
);
create policy session_processing_owner_access on public.session_processing
for all to authenticated using (
    exists (select 1 from public.analysis_sessions s
            where s.id = session_processing.session_id and s.user_id = auth.uid())
) with check (
    exists (select 1 from public.analysis_sessions s
            where s.id = session_processing.session_id and s.user_id = auth.uid())
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

grant all privileges on public.analysis_sessions to service_role;
grant execute on function public.replace_session_document_chunks(uuid, jsonb)
    to service_role;
grant execute on function public.match_session_document_chunks(
    uuid, extensions.vector, integer, double precision
) to service_role;

commit;
