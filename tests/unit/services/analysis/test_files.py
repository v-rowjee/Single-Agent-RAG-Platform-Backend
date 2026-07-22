from __future__ import annotations

from app.services.analysis.files import DatasetFileService


def test_inspection_preview_and_workspace_headers_share_file_owner() -> None:
    service = DatasetFileService()
    content = b"Revenue,revenue,Region\n10,,North\n10,,North\n"

    inspection = service.inspect_file("sales.csv", content)
    columns, rows = service.read_preview_page("sales.csv", content, 1, 1)
    workspace = service.read_workspace_dataframe("sales.csv", content)

    assert inspection.row_count == 2
    assert inspection.column_count == 3
    assert inspection.duplicate_row_count == 1
    assert columns == ["Revenue", "revenue", "Region"]
    assert rows[0]["Revenue"] == 10
    assert list(workspace.columns) == ["revenue", "revenue_2", "region"]
