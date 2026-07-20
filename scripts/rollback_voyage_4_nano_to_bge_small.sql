-- Roll back the 1024-dimensional voyage-4-nano index to 384-dimensional
-- BGE-small embeddings. Uploaded datasets and sessions are preserved;
-- derived dashboards, processing state, and chunks are rebuilt on demand.

begin;

drop function if exists public.match_session_document_chunks(
    uuid, extensions.vector, integer, double precision
);
drop index if exists public.document_chunks_embedding_hnsw_idx;

delete from public.document_chunks;
delete from public.dashboards;
delete from public.session_processing;

alter table public.document_chunks
    alter column embedding type extensions.vector(384)
    using embedding::extensions.vector(384);

create index document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

create function public.match_session_document_chunks(
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

revoke execute on function public.match_session_document_chunks(
    uuid, extensions.vector, integer, double precision
) from public;
grant execute on function public.match_session_document_chunks(
    uuid, extensions.vector, integer, double precision
) to service_role;

update public.analysis_sessions
set rag_status = 'pending',
    updated_at = now();

commit;
