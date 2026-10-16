import json

import pandas as pd
import pytest
from test_features import sample

from dataset_builder.common import sha256
from dataset_builder.labels import Costs
from dataset_builder.pipeline import build
from dataset_builder.source import discover
from dataset_builder.verify import verify_output

SNAPSHOT = "yahoo_20230501T220000Z"


def write_source(root, layout):
    universe = {
        "symbols": [{"symbol": "TEST", "asset": "equity", "roles": ["ana20"]}],
        "start": "2022-01-01",
        "horizons": [5, 10, 20],
        "folds": [
            {
                "name": "K1",
                "train_end": "2022-12-30",
                "val_start": "2023-01-01",
                "val_end": "2023-02-28",
            }
        ],
        "test": {"start": "2023-03-01"},
    }
    if layout == "legacy":
        clean = root / "data/clean" / SNAPSHOT
        raw = root / "data/raw" / SNAPSHOT
        raw_prices = raw / "prices"
        quality = root / "data/reports" / SNAPSHOT / "kalite_detay.json"
        config = root / "config/data/universe.json"
    else:
        raw = root / "data" / SNAPSHOT
        raw_prices = raw / "raw"
        clean = raw / "clean"
        quality = raw / "issues.csv"
        config = root / "universe.json"
    for path in (clean, raw_prices, quality.parent, config.parent):
        path.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(universe), encoding="utf-8")
    manifest = {"symbols": {}}
    if layout == "legacy":
        manifest["downloaded_at_utc"] = "2023-05-01T22:00:00+00:00"
    else:
        manifest.update(
            downloaded_at="2023-05-01T22:00:00+00:00", last_complete_session="2023-05-01"
        )
    for symbol, rate in (("TEST", 0.001), ("SPY", 0.0003), ("QQQ", 0.0006)):
        data = sample(rate=rate)
        data.to_parquet(clean / f"{symbol}.parquet")
        data.to_parquet(raw_prices / f"{symbol}.parquet")
        record = {"sha256": sha256(raw_prices / f"{symbol}.parquet")}
        if layout == "legacy":
            record.update(status="OK", file=f"prices/{symbol}.parquet")
        manifest["symbols"][symbol] = record
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if layout == "legacy":
        quality.write_text(
            json.dumps(
                {
                    "manifest": {"snapshot_id": SNAPSHOT},
                    "results": {s: {"durum": "OK"} for s in ("TEST", "SPY", "QQQ")},
                }
            ),
            encoding="utf-8",
        )
    else:
        pd.DataFrame(columns=["symbol", "severity", "code", "msg"]).to_csv(quality, index=False)
    return discover(root, SNAPSHOT)


@pytest.mark.parametrize("layout", ["legacy", "main"])
def test_end_to_end_and_independent_output_verification(tmp_path, layout):
    source = write_source(tmp_path / "project", layout)
    result = build(source.project, SNAPSHOT, tmp_path / "output")
    assert result["rows"] == 320
    assert result["symbols"] == 1
    assert result["numeric_checks_passed"] == 9
    assert verify_output(tmp_path / "output")["status"] == "verified"
    assert (tmp_path / "output/INCELEME.html").is_file()
    source.check_unchanged()


def test_two_layouts_produce_identical_dataset(tmp_path):
    outputs = []
    for layout in ("legacy", "main"):
        source = write_source(tmp_path / layout, layout)
        build(source.project, SNAPSHOT, tmp_path / f"out_{layout}")
        outputs.append(pd.read_parquet(tmp_path / f"out_{layout}/dataset.parquet"))
    pd.testing.assert_frame_equal(*outputs)


def test_overwrite_is_rejected(tmp_path):
    source = write_source(tmp_path / "project", "main")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="zaten var"):
        build(source.project, SNAPSHOT, output)


def test_raw_checksum_failure_stops_before_output(tmp_path):
    source = write_source(tmp_path / "project", "legacy")
    (source.raw / "prices/TEST.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bütünlüğü"):
        build(source.project, SNAPSHOT, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_benchmark_stops_before_output(tmp_path):
    source = write_source(tmp_path / "project", "main")
    (source.clean / "SPY.parquet").unlink()
    with pytest.raises(ValueError, match="Temiz veri eksik"):
        build(source.project, SNAPSHOT, tmp_path / "output")


def test_main_global_warning_is_preserved(tmp_path):
    source = write_source(tmp_path / "project", "main")
    pd.DataFrame(
        [{"symbol": "*", "severity": "UYARI", "code": "CHECK", "msg": "Needs review"}]
    ).to_csv(source.quality_file, index=False)
    source = discover(source.project, SNAPSHOT)
    assert source.statuses["TEST"] == "OK"
    assert source.global_issues[0]["code"] == "CHECK"
    result = build(source.project, SNAPSHOT, tmp_path / "output")
    assert result["global_quality_issues"] == source.global_issues


def test_as_of_does_not_return_future_signal_rows(tmp_path):
    source = write_source(tmp_path / "project", "main")
    build(source.project, SNAPSHOT, tmp_path / "output", as_of="2022-12-30")
    result = pd.read_parquet(tmp_path / "output/dataset.parquet")
    assert result.date.max() == pd.Timestamp("2022-12-30")
    assert result.target_return_20.iloc[-20:].isna().all()


def test_explicit_costs_and_class_threshold(tmp_path):
    source = write_source(tmp_path / "project", "main")
    build(
        source.project, SNAPSHOT, tmp_path / "output", costs=Costs(10, 10), neutral_threshold=0.01
    )
    result = pd.read_parquet(tmp_path / "output/dataset.parquet")
    for horizon in (5, 10, 20):
        target = result[f"target_long_net_{horizon}"]
        classes = result[f"target_class_{horizon}"]
        assert classes.loc[target.isna()].isna().all()
        assert classes.loc[target.gt(0.01)].eq(1).all()
        assert classes.loc[target.lt(-0.01)].eq(-1).all()
        assert classes.loc[target.between(-0.01, 0.01)].eq(0).all()


def test_classes_require_explicit_costs(tmp_path):
    source = write_source(tmp_path / "project", "main")
    with pytest.raises(ValueError, match="maliyetleri"):
        build(source.project, SNAPSHOT, tmp_path / "output", neutral_threshold=0.01)


def test_output_cannot_be_inside_source_data(tmp_path):
    source = write_source(tmp_path / "project", "main")
    with pytest.raises(ValueError, match="ayrı"):
        build(source.project, SNAPSHOT, source.project / "data/output")


def test_verifier_detects_changed_export(tmp_path):
    source = write_source(tmp_path / "project", "main")
    build(source.project, SNAPSHOT, tmp_path / "output")
    (tmp_path / "output/latest.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="bütünlüğü"):
        verify_output(tmp_path / "output")


def test_bad_input_column_types_fail(tmp_path):
    source = write_source(tmp_path / "project", "main")
    file = source.clean / "TEST.parquet"
    data = pd.read_parquet(file)
    data["eligible"] = "True"
    data.to_parquet(file)
    with pytest.raises(ValueError, match="boolean"):
        build(source.project, SNAPSHOT, tmp_path / "output")


def test_time_cutoff_excludes_incomplete_session(tmp_path):
    source = write_source(tmp_path / "project", "main")
    metadata = json.loads(source.manifest_file.read_text(encoding="utf-8"))
    data = sample()
    last = str(data.index[-1].date())
    metadata["downloaded_at"] = last + "T15:00:00+00:00"
    metadata["last_complete_session"] = last
    source.manifest_file.write_text(json.dumps(metadata), encoding="utf-8")
    source = discover(source.project, SNAPSHOT)
    assert source.load("TEST").index.max() < pd.Timestamp(last)


def test_fold_overlap_is_rejected(tmp_path):
    source = write_source(tmp_path / "project", "main")
    config = source.universe
    config["folds"][0]["val_start"] = config["folds"][0]["train_end"]
    source.universe_file.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="çakışıyor"):
        build(source.project, SNAPSHOT, tmp_path / "output")


@pytest.mark.parametrize("layout", ["legacy", "main"])
def test_as_of_skips_later_eligible_symbol_and_preserves_existing_rows(tmp_path, layout):
    source = write_source(tmp_path / "project", layout)
    data = sample()
    data.loc[data.index[:260], "eligible"] = False
    data.to_parquet(source.clean / "NBIS.parquet")
    raw_file = source.raw / ("prices" if layout == "legacy" else "raw") / "NBIS.parquet"
    data.to_parquet(raw_file)
    manifest = source.manifest
    record = {"sha256": sha256(raw_file)}
    if layout == "legacy":
        record.update(status="OK", file="prices/NBIS.parquet")
        quality = json.loads(source.quality_file.read_text(encoding="utf-8"))
        quality["results"]["NBIS"] = {"durum": "OK"}
        source.quality_file.write_text(json.dumps(quality), encoding="utf-8")
    manifest["symbols"]["NBIS"] = record
    source.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    config = source.universe
    config["symbols"].append({"symbol": "NBIS", "asset": "equity", "roles": ["nbis"]})
    source.universe_file.write_text(json.dumps(config), encoding="utf-8")
    full_dir, cut_dir = tmp_path / "full", tmp_path / "cut"
    build(source.project, SNAPSHOT, full_dir)
    result = build(source.project, SNAPSHOT, cut_dir, as_of="2022-12-30")
    assert result["skipped_symbols"] == [
        {"symbol": "NBIS", "reason": "no_eligible_data_on_or_before_cutoff"}
    ]
    full = pd.read_parquet(full_dir / "dataset.parquet")
    cut = pd.read_parquet(cut_dir / "dataset.parquet")
    expected = full.loc[full.date.le(pd.Timestamp("2022-12-30"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        cut[result["feature_columns"]], expected[result["feature_columns"]]
    )
    for h in result["horizons"]:
        ready = cut[f"label_ready_{h}"]
        pd.testing.assert_series_equal(
            cut.loc[ready, f"target_return_{h}"], expected.loc[ready, f"target_return_{h}"]
        )
        assert cut[f"target_return_{h}"].iloc[-h:].isna().all()


def test_no_eligible_equities_is_explicit_error(tmp_path):
    source = write_source(tmp_path / "project", "main")
    file = source.clean / "TEST.parquet"
    data = pd.read_parquet(file)
    data["eligible"] = False
    data.to_parquet(file)
    with pytest.raises(ValueError, match="hiç hisse kalmadı"):
        build(source.project, SNAPSHOT, tmp_path / "output", as_of="2022-12-30")
    assert not (tmp_path / "output").exists()


def test_global_error_is_not_ignored(tmp_path):
    source = write_source(tmp_path / "project", "main")
    pd.DataFrame([{"symbol": "*", "severity": "HATA", "code": "CHECK", "msg": "Stop"}]).to_csv(
        source.quality_file, index=False
    )
    with pytest.raises(ValueError, match="genel HATA"):
        build(source.project, SNAPSHOT, tmp_path / "output")


def test_symbol_warning_remains_local(tmp_path):
    source = write_source(tmp_path / "project", "main")
    pd.DataFrame(
        [{"symbol": "TEST", "severity": "UYARI", "code": "CHECK", "msg": "Review"}]
    ).to_csv(source.quality_file, index=False)
    source = discover(source.project, SNAPSHOT)
    assert source.statuses["TEST"] == "UYARI"
    assert source.statuses["SPY"] == "OK"
    assert source.global_issues == []


@pytest.mark.parametrize("tamper", [False, True])
def test_main_auxiliary_files_are_verified(tmp_path, tamper):
    source = write_source(tmp_path / "project", "main")
    auxiliary = source.raw / "provenance.txt"
    auxiliary.write_text("provider source record", encoding="utf-8")
    manifest = source.manifest
    manifest["auxiliary_files"] = [{"file": "provenance.txt", "sha256": sha256(auxiliary)}]
    source.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    if tamper:
        auxiliary.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="Yardımcı kaynak"):
            discover(source.project, SNAPSHOT)
    else:
        checked = discover(source.project, SNAPSHOT)
        checked.check_unchanged()
