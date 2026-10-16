import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.data as research_data
from dataset_builder.labels import make_labels
from dataset_builder.source import Source
from research.data import (
    DAY_KEYS,
    cross_sectional_rank,
    daily_returns,
    load_panel,
    residual_return,
    rolling_beta,
)
from research.experiment import EXPERIMENTS, run
from research.metrics import (
    daily_rank_ic,
    neutralize,
    newey_west_tstat,
    top_k_mean,
    top_minus_all,
)
from research.models import RidgeModel, Winsorizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "output" / "yahoo_20261016T121656Z_dataset_v1_1"


def panel_frame(days=40, names=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=days)
    frame = pd.DataFrame(
        [(d, f"S{i}") for d in dates for i in range(names)], columns=["date", "symbol"]
    )
    frame["pred"] = rng.normal(size=len(frame))
    return frame


def test_newey_west_without_lags_is_plain_tstat():
    x = np.random.default_rng(1).normal(0.1, 1, 500)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, 0) == pytest.approx(plain)


def test_newey_west_shrinks_tstat_for_overlapping_sums():
    noise = np.random.default_rng(2).normal(0.05, 1, 3000)
    overlapping = pd.Series(noise).rolling(20).mean().dropna()
    assert abs(newey_west_tstat(overlapping, 19)) < abs(newey_west_tstat(overlapping, 0)) / 2


def test_daily_rank_ic_perfect_reversed_and_small_days():
    frame = panel_frame()
    assert daily_rank_ic(frame.assign(y=frame.pred * 3)).eq(1).all()
    assert np.allclose(daily_rank_ic(frame.assign(y=-frame.pred)), -1)
    small = frame.groupby("date").head(4).assign(y=frame.pred)
    assert daily_rank_ic(small, min_names=5).empty


def test_daily_rank_ic_ignores_a_same_day_constant():
    frame = panel_frame()
    frame["y"] = np.random.default_rng(3).normal(size=len(frame))
    shift = frame.groupby("date")["y"].transform(lambda s: np.random.default_rng(4).normal())
    pd.testing.assert_series_equal(
        daily_rank_ic(frame), daily_rank_ic(frame.assign(y=frame.y - shift))
    )


def test_top_minus_all_uses_highest_scores():
    frame = panel_frame(days=1, names=10)
    frame["y"] = frame["pred"].rank()
    assert top_minus_all(frame, k=5).iloc[0] == pytest.approx(8.0 - 5.5)


def test_winsorizer_clips_to_training_quantiles():
    train = np.arange(101, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.1).fit(train)
    assert w.transform([[-50.0], [50.0], [500.0]]).ravel().tolist() == [10.0, 50.0, 90.0]


def test_ridge_alpha_split_is_purged_and_recovers_signal():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=700)
    rows = pd.DataFrame([(d, f"S{i}") for d in dates for i in range(8)], columns=["date", "symbol"])
    rows["label_end"] = rows["date"] + pd.offsets.BDay(20)
    rows["signal"] = rng.normal(size=len(rows))
    rows["noise"] = rng.normal(size=len(rows))
    rows["y"] = 0.02 * rows["signal"] + rng.normal(0, 0.05, len(rows))
    model = RidgeModel(["signal", "noise"])
    inner_train, inner_val = model.inner_split(rows)
    assert inner_train["label_end"].max() < inner_val["date"].min()
    assert inner_val["date"].nunique() == 252
    model.fit(rows, "y")
    assert model.alpha_ in model.inner_mse_
    coef = model.coefficients()
    assert coef["signal"] > 0 and abs(coef["signal"]) > 5 * abs(coef["noise"])


@pytest.fixture(scope="module")
def panel():
    return load_panel(DATASET, ROOT, 20)


def test_panel_matches_dataset_fold_counts_and_keeps_test_closed(panel):
    counts = pd.read_csv(DATASET / "fold_counts.csv", encoding="utf-8-sig").query("horizon == 20")
    got = panel.rows.groupby(["fold", "partition"]).size().unstack()
    for _, row in counts.iterrows():
        assert got.loc[row.fold, "train"] == row.train_all_24
        assert got.loc[row.fold, "validation"] == row.validation_all_24
    assert panel.test_start == pd.Timestamp("2025-01-01")
    assert panel.rows["date"].max() < panel.test_start
    assert panel.rows["label_end"].max() < panel.test_start
    for fold in panel.folds:
        train, val = panel.split(fold["name"])
        assert train["label_end"].max() <= pd.Timestamp(fold["train_end"])
        assert val["date"].min() >= pd.Timestamp(fold["val_start"])
        assert val["label_end"].max() <= pd.Timestamp(fold["val_end"])


def test_qqq_target_is_stock_minus_same_window_qqq(panel):
    rows = panel.rows
    raw, excess = panel.targets["raw"], panel.targets["vs_qqq"]
    assert np.allclose(rows[excess], rows[raw] - rows["benchmark_return"])
    # QQQ getirisi aynı gün bütün hisselerde aynıdır.
    assert rows.groupby(["fold", "partition", "date"])["benchmark_return"].nunique().eq(1).all()
    qqq = pd.read_parquet(ROOT / "data/yahoo_20261016T121656Z/clean/QQQ.parquet")
    day = pd.Timestamp("2023-06-01")
    sessions = qqq.index[qqq.index > day]
    entry = qqq.loc[sessions[0]]
    expected = qqq.loc[sessions[19], "adj_close"] / (
        entry["open"] * entry["adj_close"] / entry["close"]
    )
    got = rows.loc[rows["date"].eq(day), "benchmark_return"].iloc[0]
    assert got == pytest.approx(expected - 1)


def test_top_k_mean_skips_missing_values():
    frame = panel_frame(days=1, names=6)
    frame["pred"] = [6.0, 5, 4, 3, 2, 1]
    frame["beta"] = [1.0, np.nan, 3.0, 9, 9, 9]
    assert top_k_mean(frame, "beta", k=3).iloc[0] == pytest.approx(2.0)


def test_daily_returns_leave_gaps_and_ineligible_days_missing():
    sessions = pd.bdate_range("2024-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "adj_close": [100.0, 110, 121, 133.1, 146.41],
            "eligible": [False, True, True, True, True],
            "is_session": True,
        },
        index=sessions.delete(3),
    )
    r = daily_returns(frame, sessions)
    assert r.isna().tolist() == [True, True, False, True, True, False]
    assert r.iloc[2] == pytest.approx(0.1)


def test_rolling_beta_recovers_multiplier_and_uses_only_past():
    rng = np.random.default_rng(6)
    index = pd.bdate_range("2020-01-01", periods=400)
    bench = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    stock = 2 * bench + rng.normal(0, 0.001, len(index))
    beta = rolling_beta(stock, bench, window=252, min_obs=200)
    assert beta.iloc[:199].isna().all()
    assert beta.iloc[199:].notna().all()
    assert beta.iloc[300] == pytest.approx(2, abs=0.05)
    changed = stock.copy()
    changed.iloc[301:] = -changed.iloc[301:]
    assert rolling_beta(changed, bench, window=252, min_obs=200).iloc[300] == beta.iloc[300]
    # Pencerede 200'den az ortak gözlem kalırsa beta boş kalır.
    gap = stock.copy()
    gap.iloc[250:310] = np.nan
    gapped = rolling_beta(gap, bench, window=252, min_obs=200)
    assert np.isnan(gapped.iloc[309])
    assert gapped.iloc[249] == beta.iloc[249]


def price_frame(close, index, eligible=True):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "eligible": eligible,
            "is_session": True,
        },
        index=index,
    )


def test_beta_from_prices_ignores_later_prices_and_ineligible_days():
    import exchange_calendars as xc

    index = xc.get_calendar("XNYS").sessions_in_range("2020-01-02", "2022-06-30")
    rng = np.random.default_rng(7)
    q = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(index)))
    s = 50 * np.cumprod(1 + 1.5 * np.r_[0, np.diff(q) / q[:-1]] + rng.normal(0, 0.002, len(index)))
    t = 400
    beta = rolling_beta(
        daily_returns(price_frame(s, index), index), daily_returns(price_frame(q, index), index)
    )
    later = s.copy()
    later[t + 1 :] *= np.linspace(1, 3, len(index) - t - 1)
    moved = rolling_beta(
        daily_returns(price_frame(later, index), index), daily_returns(price_frame(q, index), index)
    )
    assert moved.iloc[t] == beta.iloc[t]
    assert beta.iloc[t] == pytest.approx(1.5, abs=0.05)
    # Eligible olmayan dönem (ör. SPAC öncesi) hesaba girmez: ilk 300 gün yok sayılır.
    eligible = np.arange(len(index)) >= 300
    spac = rolling_beta(
        daily_returns(price_frame(s, index, eligible), index),
        daily_returns(price_frame(q, index), index),
    )
    assert spac.iloc[:500].isna().all()
    assert spac.iloc[501] == pytest.approx(beta.iloc[501], abs=0.2)


def test_residual_target_is_zero_for_a_pure_beta_stock():
    import exchange_calendars as xc

    index = xc.get_calendar("XNYS").sessions_in_range("2020-01-02", "2021-12-31")
    rng = np.random.default_rng(8)
    qr = rng.normal(0, 0.0005, len(index))
    qr[0] = 0
    q = 100 * np.cumprod(1 + qr)
    s = 50 * np.cumprod(1 + 1.5 * qr)
    qqq, stock = price_frame(q, index), price_frame(s, index)
    labels_q = make_labels(qqq, "QQQ", (20,)).set_index("date")
    labels_s = make_labels(stock, "S", (20,)).set_index("date")
    beta = rolling_beta(daily_returns(stock, index), daily_returns(qqq, index))
    resid = residual_return(labels_s["target_return_20"], beta, labels_q["target_return_20"])
    assert labels_s["label_end_20"].equals(labels_q["label_end_20"])
    assert beta.dropna().between(1.5 - 1e-9, 1.5 + 1e-9).all()
    assert resid.dropna().abs().max() < 1e-4
    assert resid.notna().sum() > 200


def test_neutralize_removes_daily_linear_exposure():
    frame = panel_frame(days=30, names=8)
    frame["beta"] = np.random.default_rng(9).normal(1, 0.5, len(frame))
    frame["pred"] = 2 * frame["beta"] + frame["pred"]
    out = frame.assign(n=neutralize(frame, "beta"))
    corr = out.groupby("date")[["n", "beta"]].corr().xs("n", level=1)["beta"]
    assert corr.abs().max() < 1e-12
    assert out.groupby("date")["n"].mean().abs().max() < 1e-12


def test_load_panel_rejects_benchmark_window_mismatch(monkeypatch):
    original = research_data.benchmark_returns

    def shifted(*args, **kwargs):
        bench = original(*args, **kwargs)
        bench["benchmark_label_end"] += pd.Timedelta(days=1)
        return bench

    monkeypatch.setattr(research_data, "benchmark_returns", shifted)
    with pytest.raises(ValueError, match="pencereleri uyuşmuyor"):
        load_panel(DATASET, ROOT, 20)


def test_cross_sectional_rank_is_centered_per_day():
    rows = pd.DataFrame(
        {
            "fold": "K1",
            "partition": "train",
            "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 5),
            "x": [10.0, -3, 7, 1, 2, 3, 4, 5],
        }
    )
    ranks = cross_sectional_rank(rows, ["x"])["x"]
    assert ranks.tolist() == [0.5, -0.5, 0.0, -0.5, -0.25, 0.0, 0.25, 0.5]


def test_panel_diagnostics_and_xs_features(panel):
    rows = panel.rows
    xs = panel.targets["xs"]
    assert rows.groupby(DAY_KEYS)[xs].sum().abs().max() < 1e-12
    assert "xs_market_spy_return_20" not in panel.xs_features
    assert "xs_market_qqq_return_20" not in panel.xs_features
    assert len(panel.xs_features) == 22
    assert rows[panel.xs_features].min().min() == -0.5
    assert rows[panel.xs_features].max().max() == 0.5
    val = rows.loc[rows["partition"].eq("validation")]
    assert val["beta_252"].notna().mean() > 0.95
    median = val.groupby("symbol")["beta_252"].median()
    assert median["XOM"] < 0.5
    assert median["NVDA"] > 1.3
    up = rows["benchmark_return"] > 0.02
    down = rows["benchmark_return"] < -0.02
    assert rows.loc[up, "regime"].eq("yukari").all()
    assert rows.loc[down, "regime"].eq("asagi").all()
    assert rows.loc[~up & ~down, "regime"].eq("yatay").all()
    adj = rows[panel.targets["raw"]] - rows["beta_252"] * rows["benchmark_return"]
    assert np.allclose(rows["beta_adj_return"], adj, equal_nan=True)


def test_panel_never_loads_prices_from_test_period(monkeypatch):
    original = Source.load
    seen = []

    def checked(self, symbol, as_of=None):
        frame = original(self, symbol, as_of)
        seen.append(symbol)
        assert frame.index.max() < pd.Timestamp("2025-01-01")
        return frame

    monkeypatch.setattr(Source, "load", checked)
    load_panel(DATASET, ROOT, 20)
    assert "QQQ" in seen and len(set(seen)) == 29


@pytest.fixture(scope="module")
def runs(panel, tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    return root, {key: run(panel, root / f"deney{key}", EXPERIMENTS[key]) for key in "123"}


def test_experiment_writes_report_and_registry(panel, runs):
    root, results = runs
    config, summary, consistency = results["1"]
    out = root / "deney1"
    assert config["test_period_read"] is False
    assert set(summary["model"]) == {
        "sifir",
        "egitim_ortalamasi",
        "momentum_20",
        "geri_donus_5",
        "en_oynak_20",
        "yuksek_beta",
        "ridge",
    }
    assert consistency["folds"].eq(4).all()
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert predictions["date"].max() < panel.test_start
    report = (out / "RAPOR.md").read_text(encoding="utf-8")
    assert "2025-01-01 ve sonrası (test dönemi) okunmadı" in report
    assert "## 3. Rejim ve risk" in report
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["horizon"] == 20
    registry = pd.read_csv(root / "runs.csv")
    assert registry["run_id"].tolist() == ["deney1", "deney1", "deney2", "deney3"]
    assert registry["target"].tolist() == ["raw", "vs_qqq", "xs", "resid"]
    with pytest.raises(ValueError, match="zaten var"):
        run(panel, out)


def test_deney2_uses_xs_inputs_and_matches_deney1_reference(runs):
    root, results = runs
    config, summary, consistency = results["2"]
    assert config["targets"] == {"xs": "target_xs_20"}
    assert all(f.startswith("xs_") for f in config["features"])
    assert consistency["target"].tolist() == ["xs"]
    ref = summary.set_index("model").loc["ridge_deney1"]
    deney1 = results["1"][1].set_index(["target", "model"]).loc[("raw", "ridge")]
    # Günlük rank IC gün içi sabitten etkilenmez: referans Deney 1 Ridge'iyle aynı olmalı.
    assert ref["ic_mean"] == pytest.approx(deney1["ic_mean"], abs=1e-12)
    assert np.isnan(ref.get("r2_oos", np.nan))
    coefs = pd.read_csv(root / "deney2" / "ridge_coefficients.csv")
    assert coefs["feature"].str.startswith("xs_").all()


def test_deney3_residual_target_neutral_row_and_criteria(panel, runs):
    root, results = runs
    config, summary, _ = results["3"]
    assert config["targets"] == {"resid": "target_resid_20"}
    assert config["features"] == results["2"][0]["features"]
    assert config["universe"] == "beta"
    assert {r["excluded"] for r in config["excluded_rows"]} == {0}
    s = summary.set_index("model")
    assert abs(s.loc["ridge_beta_notr", "corr_beta"]) < 0.05
    assert s.loc["yuksek_beta", "corr_beta"] == pytest.approx(1.0)
    report = (root / "deney3" / "RAPOR.md").read_text(encoding="utf-8")
    assert "Önceden yazılı başarı kriteri" in report
    assert "Ridge · beta-nötrlenmiş skor" in report
    # Hedef: hisse − beta × QQQ, gün içi arındırılmış; Deney 2 satırlarıyla aynı evren.
    rows = panel.with_beta().rows
    resid = rows["target_return_20"] - rows["beta_252"] * rows["benchmark_return"]
    resid -= resid.groupby([rows[k] for k in DAY_KEYS]).transform("mean")
    assert np.allclose(rows["target_resid_20"], resid)
    assert len(rows) == len(panel.rows)
