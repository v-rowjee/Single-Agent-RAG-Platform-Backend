from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.agents.multi.data_preparation_agent import _generic_clean_csv


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_generic_cleaning_accepts_api_supported_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / f"sales{suffix}"
    frame = pd.DataFrame(
        {
            " Sales Date ": ["2025-01-01", "2025-01-01", "2025-02-01"],
            "Revenue": [100, 100, 150],
            "Empty": [None, None, None],
        }
    )
    if suffix == ".csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_excel(source, index=False)

    cleaned, report = _generic_clean_csv(str(source), tmp_path)

    assert len(cleaned) == 2
    assert report.duplicate_rows_removed == 1
    assert report.empty_columns_removed == ["empty"]
    assert Path(report.cleaned_file_path).suffix == ".csv"
    assert Path(report.cleaned_file_path).is_file()
