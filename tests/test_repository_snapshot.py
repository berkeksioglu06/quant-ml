from pathlib import Path

import pandas as pd

from dataset_builder.source import discover
from dataset_builder.verify import verify_output

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "yahoo_20261016T121656Z"


def test_committed_native_snapshot_is_complete_and_unchanged():
    source = discover(ROOT, SNAPSHOT, "main")
    assert len(source.manifest["symbols"]) == 47
    assert source.last_session == pd.Timestamp("2026-10-15")
    assert len(list(source.clean.glob("*.parquet"))) == 47
    assert len(list((source.raw / "raw/earnings").glob("*.parquet"))) == 29
    source.check_unchanged()


def test_committed_dataset_output_is_valid():
    result = verify_output(ROOT / "output" / f"{SNAPSHOT}_dataset_v1_1")
    assert result["rows"] == 65413
    assert result["symbols"] == 29
