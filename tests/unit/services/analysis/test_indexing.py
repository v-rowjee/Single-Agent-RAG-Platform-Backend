from __future__ import annotations

from app.services.analysis.indexing import AnalysisExecutionPersistenceService


def test_persistence_sanitizers_remove_temporary_paths_recursively() -> None:
    prepared = AnalysisExecutionPersistenceService.persistent_prepared_dataset(
        {
            "prepared_file_path": "temporary.csv",
            "temporal_dataset_path": "temporal.csv",
            "dataset_profile": {"row_count": 3},
            "cleaning_report": {
                "cleaned_file_path": "cleaned.csv",
                "removed_rows": 1,
            },
        }
    )

    assert prepared == {
        "dataset_profile": {"row_count": 3},
        "cleaning_report": {"removed_rows": 1},
    }
