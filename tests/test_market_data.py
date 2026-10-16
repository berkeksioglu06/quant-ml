"""Main veri hattı: bozuk veri örnekleri ve sabit gerçek veriyle uçtan uca denetim."""

import io
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import market_data as m
from dataset_builder.pipeline import build as build_dataset
from dataset_builder.verify import verify_output


def bars(start="2020-01-02", end="2020-04-30"):
    dates = m.CAL.sessions_in_range(start, end).tz_localize(None)
    return pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "splits": 0.0,
            "dividends": 0.0,
            "capital_gains": 0.0,
            "raw_close": 10.0,
            "raw_volume": 100.0,
            "split_factor": 1.0,
            "eligible": True,
            "is_session": True,
        },
        index=dates,
    )


def codes(issues):
    return {e["code"] for e in issues}


@pytest.mark.parametrize(
    "column,value,code",
    [
        ("close", np.inf, "NAN_OHLC"),
        ("volume", np.inf, "NAN_VOLUME"),
        ("volume", -1.0, "NEG_VOL"),
        ("high", 8.0, "HIGH_LOW"),
    ],
)
def test_bad_bars_are_reported_without_mutation(column, value, code):
    df = bars()
    df.loc[df.index[3], column] = value
    before = df.copy(deep=True)
    issues = []
    m.check_ohlcv(df, "equity", issues)
    assert code in codes(issues)
    pd.testing.assert_frame_equal(df, before)


def test_missing_market_fields_are_not_fabricated_as_zero():
    raw = pd.DataFrame({"Date": [pd.Timestamp("2020-01-02")], "Close": [10.0]})
    daily, _ = m.to_daily(raw, pd.Timestamp("2020-01-02"))
    assert daily[["open", "high", "low", "volume", "adj_close"]].isna().all().all()
    assert daily["splits"].iloc[0] == 0


def test_dividend_adjustment_consistency_and_missing_capital_gains():
    df = bars().drop(columns="capital_gains")
    date = df.index[30]
    df.loc[df.index < date, "adj_close"] = 9.0
    df.loc[date, "dividends"] = 1.0
    issues = []
    m.check_dividends(df, issues)
    assert "TEMETTU_FAKTOR" not in codes(issues)
    df.loc[date, "dividends"] = 0.1
    issues = []
    m.check_dividends(df, issues)
    assert "TEMETTU_FAKTOR" in codes(issues)


def test_invalid_adjusted_price_is_error():
    df = bars()
    df.loc[df.index[4], "adj_close"] = np.inf
    issues = []
    m.check_dividends(df, issues)
    assert any(e["code"] == "ADJ_GECERSIZ" and e["severity"] == "HATA" for e in issues)


def test_spac_signature_and_eligible_cut_are_preserved():
    df = bars()
    cut = df.index[40]
    df["eligible"] = df.index >= cut
    issues = []
    result = m.check_history(
        df, {"asset": "equity", "eligible_from": str(cut.date())}, {}, issues, df.index[0]
    )
    assert result["kesim_oncesi_satir"] == 40
    assert any(e["code"] == "KESIM_ONCESI" and "SPAC" in e["msg"] for e in issues)
    assert not df.iloc[:40]["eligible"].any()


def test_first_session_uses_requested_start_and_reports_missing_history():
    expected, _, _ = m.expected_first_session(
        {"expected_first_date": "2010-01-01"}, pd.Timestamp("2020-01-01")
    )
    assert expected == pd.Timestamp("2020-01-02")
    issues = []
    m.check_history(
        bars(start="2020-02-03"), {"asset": "equity"}, {}, issues, pd.Timestamp("2020-01-01")
    )
    assert "BASLANGIC_EKSIK" in codes(issues)


def test_long_missing_session_gap_is_error():
    df = bars().drop(bars().index[10:35])
    issues = []
    m.check_calendar(df, "equity", issues, df.index[-1])
    assert any(e["code"] == "EKSIK_SEANS" and e["severity"] == "HATA" for e in issues)


def test_split_price_break_is_detected_but_adjusted_volume_is_stable(monkeypatch):
    df = bars()
    cut = df.index[40]
    definition = {"asset": "equity", "expected_splits": [[str(cut.date()), 10.0]]}
    monkeypatch.setattr(m, "info", lambda _: definition)
    df.loc[cut, "splits"] = 10.0
    clean = m.quality_checks("TEST", df, df.index[-1], manifest={"start": str(df.index[0].date())})
    assert "SPLIT_FIYAT_KIRILMA" not in codes(clean["issues"])
    assert clean["split_olaylari"][0]["volume_ratio"] == 1.0
    df.loc[cut:, ["open", "high", "low", "close", "adj_close"]] /= 10
    broken = m.quality_checks("TEST", df, df.index[-1], manifest={"start": str(df.index[0].date())})
    assert "SPLIT_FIYAT_KIRILMA" in codes(broken["issues"])


def test_splits_after_snapshot_end_are_not_required(monkeypatch):
    df = bars()
    monkeypatch.setattr(
        m, "info", lambda _: {"asset": "equity", "expected_splits": [["2024-01-01", 10.0]]}
    )
    result = m.quality_checks("TEST", df, df.index[-1], manifest={"start": str(df.index[0].date())})
    assert "SPLIT_EKSIK" not in codes(result["issues"])


def test_special_distribution_blocks_historical_values_only():
    df = bars()
    before = df.copy(deep=True)
    cut = df.index[20]
    out = m.add_raw_columns(
        df,
        {"price": "back_adjusted", "volume": "back_adjusted"},
        {"historical_unverified_before": str(cut.date())},
    )
    assert out.iloc[:20][["raw_close", "raw_volume"]].isna().all().all()
    assert out.iloc[20:][["raw_close", "raw_volume"]].notna().all().all()
    pd.testing.assert_frame_equal(out[["close", "volume"]], before[["close", "volume"]])


def test_earnings_calendar_distinguishes_results_unknown_and_future(tmp_path):
    path = tmp_path / "raw" / "earnings"
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "Earnings Date": pd.to_datetime(["2020-01-02", "2020-02-02", "2020-05-02"], utc=True),
            "Reported EPS": [1.0, np.nan, np.nan],
        }
    ).to_parquet(path / "TEST.parquet")
    issues = []
    result = m.check_earnings(
        tmp_path, "TEST", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-04-30"), issues
    )
    assert [result[k] for k in ["sonuc_kaydi_var", "durumu_belirsiz", "planlanan"]] == [1, 1, 1]
    assert "BILANCO_BELIRSIZ" in codes(issues)


def test_download_preserves_earnings_checksum_and_handles_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA", tmp_path)
    df = bars()
    raw = df[["open", "high", "low", "close", "adj_close", "volume"]].rename(
        columns={v: k for k, v in m.COLUMNS.items()}
    )
    raw.index.name = "Date"
    monkeypatch.setattr(m, "fetch", lambda *_: raw)

    class Ticker:
        def __init__(self, sym):
            self.sym = sym

        def get_earnings_dates(self, limit):
            if self.sym == "MSFT":
                raise RuntimeError("offline")
            return pd.DataFrame(
                {"Reported EPS": [1.0]},
                index=pd.DatetimeIndex(["2020-01-02"], name="Earnings Date"),
            )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    snap = m.download(["AAPL", "MSFT"], pause=0)
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["symbols"]["AAPL"]["earnings_status"] == "OK"
    assert manifest["symbols"]["MSFT"]["earnings_status"] == "HATA"
    assert "error" not in manifest["symbols"]["MSFT"]
    rec = manifest["auxiliary_files"][0]
    assert m.sha256(snap / rec["file"]) == rec["sha256"]


def test_universe_file_travels_with_snapshot(tmp_path, monkeypatch):
    """--universe ile indirilen kayıt kendi evren kopyasını taşır; build o kopyayı okur."""
    monkeypatch.setattr(m, "DATA", tmp_path)
    df = bars()
    raw = df[["open", "high", "low", "close", "adj_close", "volume"]].rename(
        columns={v: k for k, v in m.COLUMNS.items()}
    )
    raw.index.name = "Date"
    monkeypatch.setattr(m, "fetch", lambda *_: raw)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=None))
    wide = dict(m.UNIVERSE)
    wide["version"] = "test-genis"
    wide["symbols"] = [{"symbol": "GENIS", "roles": ["genis"], "asset": "etf"}]
    file = tmp_path / "universe_test.json"
    file.write_text(json.dumps(wide), encoding="utf-8")
    saved, saved_file = dict(m.UNIVERSE), m.UNIVERSE_FILE
    try:
        m.set_universe(file)
        snap = m.download(pause=0)
        manifest = json.loads((snap / "manifest.json").read_text())
        assert list(manifest["symbols"]) == ["GENIS"]
        assert manifest["universe"]["source"] == "universe_test.json"
        assert m.sha256(snap / "universe.json") == manifest["universe"]["sha256"]
        # Varsayılan evrene dönülse de build snapshot'taki kopyayı yükler.
        m.set_universe(saved_file)
        assert m.UNIVERSE["version"] != "test-genis"
        m.build(snap)
        assert m.UNIVERSE["version"] == "test-genis"
        assert (snap / "clean" / "GENIS.parquet").is_file()
    finally:
        m.UNIVERSE.clear()
        m.UNIVERSE.update(saved)
        m.UNIVERSE_FILE = saved_file


def test_archived_snapshot_cannot_be_overwritten(tmp_path):
    (tmp_path / "provenance").mkdir()
    with pytest.raises(ValueError, match="Arşiv"):
        m.build(tmp_path)


def test_real_main_clean_quality_dataset_chain_matches_reviewed_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    name = "yahoo_20261016T121656Z"
    source = root / "data" / name
    snap = tmp_path / "data" / name
    snap.mkdir(parents=True)
    shutil.copytree(source / "raw", snap / "raw")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["auxiliary_files"] = [
        r for r in manifest["auxiliary_files"] if r["file"].startswith("raw/earnings/")
    ]
    (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copyfile(root / "universe.json", tmp_path / "universe.json")
    m.build(snap)
    report = json.loads((snap / "quality_detail.json").read_text(encoding="utf-8"))
    assert len(report["results"]) == 47
    assert not [r for r in report["issues"] if r["severity"] == "HATA"]
    assert (snap / "quality_report.md").is_file()
    output = tmp_path / "dataset"
    build_dataset(tmp_path, name, output, layout="main")
    expected = pd.read_parquet(root / "output" / f"{name}_dataset_v1_1" / "dataset.parquet")
    actual = pd.read_parquet(output / "dataset.parquet")
    pd.testing.assert_frame_equal(actual, expected)
    assert verify_output(output)["rows"] == 65413


def test_utf8_console_survives_cp1254_redirect(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1254", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", out)
    m.utf8_console()
    print("Kapsama ≥252 · → data/x")
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == "Kapsama ≥252 · → data/x\n"
