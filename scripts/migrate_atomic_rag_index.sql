-- Apply this migration once to existing Supabase projects.
-- It replaces one dataset's vector index in a single PostgreSQL transaction,
-- so a failed insert rolls the deletion back automatically.

create or replace function public.replace_document_chunks(
    p_dataset_id uuid,
    p_chunks jsonb
)
returns integer
language plpgsql
volatile
security invoker
set search_path = public, extensions
as $$
declare
    inserted_count integer;
begin
    if jsonb_typeof(p_chunks) <> 'array' then
        raise exception 'p_chunks must be a JSON array';
    end if;

    delete from public.document_chunks
    where dataset_id = p_dataset_id;

    insert into public.document_chunks (
        dataset_id,
        source_id,
        document_type,
        chunk_index,
        content,
        metadata,
        embedding
    )
    select
        p_dataset_id,
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

grant execute on function public.replace_document_chunks(uuid, jsonb)
to service_role;
