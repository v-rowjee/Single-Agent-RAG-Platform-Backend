from pathlib import Path


DATABASE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "db.sql"
MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "migrate_qwen_to_bge_small.sql"
)


def test_document_chunk_vectors_use_bge_small_embedding_dimensions() -> None:
    schema = DATABASE_SCHEMA_PATH.read_text(encoding="utf-8")

    assert "embedding extensions.vector(384) not null" in schema
    assert "p_query_embedding extensions.vector(384)" in schema
    assert "vector_cosine_ops" in schema
    assert "vector(1024)" not in schema


def test_database_schema_is_a_non_destructive_fresh_project_bootstrap() -> None:
    schema = DATABASE_SCHEMA_PATH.read_text(encoding="utf-8").casefold()

    assert "\ndrop table" not in schema
    assert "\ndrop function" not in schema
    assert "\ndrop trigger" not in schema
    assert "\nbegin;" in schema
    assert schema.rstrip().endswith("commit;")
    assert "create or replace function" not in schema


def test_bge_migration_preserves_uploaded_datasets_and_rebuilds_derived_data() -> None:
    migration = MIGRATION_PATH.read_text(encoding="utf-8").casefold()

    assert "alter column embedding type extensions.vector(384)" in migration
    assert "p_query_embedding extensions.vector(384)" in migration
    assert "delete from public.document_chunks" in migration
    assert "delete from public.dashboards" in migration
    assert "delete from public.session_processing" in migration
    assert "delete from public.datasets" not in migration
    assert "delete from public.analysis_sessions" not in migration
    assert "rag_status = 'pending'" in migration
