import csv
import json
from dataclasses import replace
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
    market_coverage,
    market_state_from_frames,
    residual_return,
    rolling_beta,
)
from research.experiment import (
    EXPERIMENTS,
    LEGACY_REGISTRY_FIELDS,
    REGISTRY_FIELDS,
    _migrate_registry,
    evaluate,
    run,
)
from research.metrics import (
    daily_rank_ic,
    neutralize,
    newey_west_tstat,
    top_k_mean,
    top_minus_all,
)
from research.models import LGBMModel, RidgeModel, Winsorizer

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


def test_membership_mask_drops_rows_outside_daily_universe(tmp_path):
    from research.data import apply_membership

    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3),
            "symbol": ["A", "B", "C"] * 2,
            "y": range(6),
        }
    )
    member = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "symbol": ["A", "B", "A", "C"],
            "in_universe": [True, False, True, True],
        }
    )
    member.to_parquet(tmp_path / "membership.parquet", index=False)
    kept, dropped = apply_membership(rows, tmp_path / "membership.parquet")
    assert dropped == 3
    assert list(zip(kept["date"].dt.day, kept["symbol"])) == [(2, "A"), (3, "A"), (3, "C")]


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


def test_top_k_mean_splits_ties_at_the_cutoff():
    frame = panel_frame(days=1, names=5)
    frame["pred"] = [3.0, 2, 2, 2, 1]
    frame["value"] = [10.0, 1, 2, 3, 100]
    # İlk hisse tam, eşit üçlü kalan tek yeri paylaşır: (10 + (1 + 2 + 3) / 3) / 2.
    assert top_k_mean(frame, "value", k=2).iloc[0] == pytest.approx(6.0)
    shuffled = frame.iloc[[4, 2, 0, 3, 1]]
    assert top_k_mean(shuffled, "value", k=2).iloc[0] == pytest.approx(6.0)


def synthetic_rows(days=600, names=8, seed=10):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=days)
    rows = pd.DataFrame(
        [(d, f"S{i}") for d in dates for i in range(names)], columns=["date", "symbol"]
    )
    rows["label_end"] = rows["date"] + pd.offsets.BDay(20)
    rows["signal"] = rng.normal(size=len(rows))
    rows["noise"] = rng.normal(size=len(rows))
    rows["y"] = 0.02 * rows["signal"] + rng.normal(0, 0.05, len(rows))
    return rows


def test_lgbm_same_seed_gives_identical_predictions():
    rows = synthetic_rows(days=1200)
    train, val = rows.iloc[: len(rows) // 2], rows.iloc[len(rows) // 2 :]
    first = LGBMModel(["signal", "noise"]).fit(train, "y")
    second = LGBMModel(["signal", "noise"]).fit(train, "y")
    assert first.n_trees_ == second.n_trees_ > 1
    assert np.array_equal(first.predict(val), second.predict(val))
    assert first.importance()["signal"] > first.importance()["noise"]


def test_lgbm_tree_count_is_stable_on_a_flat_validation_curve():
    # İç eğitim min_child_samples'tan küçük: hiç bölünme yok, eğri düz. Çok iş
    # parçacıklı metrik toplamında bu durumda seçilen tur çalıştırmadan çalıştırmaya oynuyordu.
    rows = synthetic_rows(days=600)
    train = rows.iloc[: len(rows) // 2]
    counts = {LGBMModel(["signal", "noise"]).fit(train, "y").n_trees_ for _ in range(4)}
    assert counts == {1}


def test_lgbm_early_stopping_split_is_purged():
    rows = synthetic_rows()
    model = LGBMModel(["signal", "noise"])
    inner_train, inner_val = model.inner_split(rows)
    assert inner_train["label_end"].max() < inner_val["date"].min()
    assert inner_val["date"].nunique() == 252
    assert set(inner_train.index).isdisjoint(inner_val.index)


def test_registry_migration_keeps_rows_and_adds_main_model(tmp_path):
    registry = tmp_path / "runs.csv"
    legacy = dict.fromkeys(LEGACY_REGISTRY_FIELDS, "x") | {
        "ridge_ic": "-0.08",
        "ridge_ic_t_nw": "-2.1",
        "ridge_r2": "-0.03",
    }
    with registry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerow(legacy)
    _migrate_registry(registry)
    rows = list(csv.DictReader(registry.open(encoding="utf-8")))
    assert list(rows[0]) == REGISTRY_FIELDS
    assert rows[0]["main_model"] == "ridge"
    assert rows[0]["main_ic"] == "-0.08"
    assert rows[0]["ridge_ic"] == "-0.08"


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
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
            "splits": 0.0,
            "eligible": eligible,
            "is_session": True,
        },
        index=index,
    )


def market_frames(seed=12, start="2020-01-02", end="2021-12-31"):
    import exchange_calendars as xc

    index = xc.get_calendar("XNYS").sessions_in_range(start, end)
    rng = np.random.default_rng(seed)
    qqq = price_frame(100 * np.cumprod(1 + rng.normal(0, 0.01, len(index))), index)
    vix = price_frame(20 * np.exp(np.cumsum(rng.normal(0, 0.05, len(index)))), index)
    return index, qqq, vix


def test_market_state_ignores_prices_after_t():
    _, qqq, vix = market_frames()
    base = market_state_from_frames(qqq, vix)
    t = 300
    later_q, later_v = qqq.copy(), vix.copy()
    for frame in (later_q, later_v):
        frame.iloc[t + 1 :, :5] *= 1.7
    moved = market_state_from_frames(later_q, later_v)
    pd.testing.assert_series_equal(base.iloc[t], moved.iloc[t])
    assert not np.allclose(base.iloc[t + 10], moved.iloc[t + 10])
    # Tam pencere: SMA200 ilk 199 seansta, 60 seanslık getiri ilk 60 seansta yok.
    assert base["market_qqq_sma200_dist"].iloc[:199].isna().all()
    assert base["market_qqq_sma200_dist"].iloc[199:].notna().all()
    assert base["market_qqq_return_60"].iloc[:60].isna().all()
    close = qqq["adj_close"]
    assert base["market_qqq_return_5"].iloc[t] == pytest.approx(
        close.iloc[t] / close.iloc[t - 5] - 1
    )
    assert base["market_qqq_sma50_dist"].iloc[t] == pytest.approx(
        close.iloc[t] / close.iloc[t - 49 : t + 1].mean() - 1
    )


def test_vix_is_aligned_to_qqq_sessions():
    index, qqq, vix = market_frames()
    missing = index[250]
    weekend = pd.Timestamp("2021-01-09")
    extra = vix.loc[[index[0]]].set_axis(pd.DatetimeIndex([weekend]))
    vix = pd.concat([vix.drop(missing), extra]).sort_index()
    out = market_state_from_frames(qqq, vix)
    assert out.index.equals(pd.DatetimeIndex(index, name="date"))
    assert weekend not in out.index
    assert np.isnan(out.loc[missing, "market_vix_close"])
    change = out["market_vix_change_5"]
    assert change.iloc[250:256].isna().all()
    assert change.iloc[256:].notna().all()
    assert out.loc[index[249], "market_vix_close"] == pytest.approx(vix.loc[index[249], "close"])
    assert out["market_qqq_return_5"].iloc[250:256].notna().all()


def test_market_coverage_counts_missing_rows_and_dates():
    dates = pd.bdate_range("2021-01-04", periods=6)
    rows = pd.DataFrame(
        {
            "fold": ["K1"] * 6 + ["K2"] * 6,
            "partition": (["train"] * 3 + ["validation"] * 3) * 2,
            "date": list(dates) * 2,
            "a": 1.0,
            "b": 1.0,
        }
    )
    rows.loc[[0, 1], "a"] = np.nan
    rows.loc[[1, 2], "b"] = np.nan
    rows.loc[9, "b"] = np.nan
    summary, detail = market_coverage(rows, ["a", "b"])
    s = summary.set_index(["fold", "partition"])
    assert s.loc[("K1", "train"), "rows"] == 3
    assert s.loc[("K1", "train"), "missing_any"] == 3
    assert s.loc[("K1", "validation"), "missing_any"] == 0
    assert s.loc[("K2", "validation"), "missing_any"] == 1
    d = detail.set_index(["fold", "partition", "feature"])
    assert d.loc[("K1", "train", "a"), "missing"] == 2
    assert d.loc[("K1", "train", "b"), "first_missing"] == dates[1]
    assert d.loc[("K1", "train", "b"), "last_missing"] == dates[2]
    assert d.loc[("K2", "validation", "b"), "first_missing"] == dates[3]
    assert pd.isna(d.loc[("K2", "train", "a"), "first_missing"])


def test_tree_curve_matches_early_stopping_choice():
    rows = synthetic_rows(days=1200)
    train = rows.iloc[: len(rows) // 2]
    model = LGBMModel(["signal", "noise"]).fit(train, "y")
    curve = model.tree_curve(train, "y", counts=[1, model.n_trees_, 50]).set_index("n_trees")
    # Aynı ağaçlar: LightGBM'in kendi l2 hesabıyla yalnız kayan nokta düzeyinde fark.
    assert curve.loc[model.n_trees_, "l2"] == pytest.approx(model.inner_mse_, rel=1e-6)
    assert curve["l2"].min() == pytest.approx(curve.loc[model.n_trees_, "l2"])


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
    assert {"QQQ", "^VIX"} <= set(seen)
    assert len(set(seen)) == 30


@pytest.fixture(scope="module")
def runs(panel, tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    keys = ("1", "2", "3", "4a", "4b")
    return root, {key: run(panel, root / f"deney{key}", EXPERIMENTS[key]) for key in keys}


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
    runs_ = ["deney1", "deney1", "deney2", "deney3", "deney4a", "deney4b"]
    assert registry["run_id"].tolist() == runs_
    assert registry["target"].tolist() == ["raw", "vs_qqq", "xs", "resid", "raw", "raw"]
    assert registry["main_model"].tolist() == ["ridge"] * 4 + ["lightgbm"] * 2
    assert registry["ridge_ic"].isna().tolist() == [False] * 4 + [True] * 2
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


def test_deney4a_lightgbm_outputs_and_reference(runs):
    root, results = runs
    config, summary, consistency = results["4a"]
    out = root / "deney4a"
    assert config["main_model"] == "lightgbm"
    assert config["targets"] == {"raw": "target_return_20"}
    assert config["features"] == results["1"][0]["features"]
    assert len(config["features"]) == 24
    assert consistency["target"].tolist() == ["raw"]
    trees = pd.read_csv(out / "lgbm_trees.csv")
    assert trees["n_trees"].between(1, 2000).all()
    importance = pd.read_csv(out / "lgbm_importance.csv")
    assert np.allclose(importance.groupby("fold")["gain_share"].sum(), 1)
    dependence = pd.read_csv(out / "lgbm_partial_dependence.csv")
    assert set(dependence["subset"]) == {"pozitif", "negatif"}
    assert len(dependence) == 4 * 2 * 5
    # Referans satırı aynı hedefe eğitilmiş Deney 1 Ridge'i: R² ve IC aynı olmalı.
    ref = summary.set_index("model").loc["ridge_deney1"]
    deney1 = results["1"][1].set_index(["target", "model"]).loc[("raw", "ridge")]
    assert ref["ic_mean"] == pytest.approx(deney1["ic_mean"], abs=1e-12)
    assert ref["r2_oos"] == pytest.approx(deney1["r2_oos"], abs=1e-12)
    report = (out / "RAPOR.md").read_text(encoding="utf-8")
    assert "## 7. LightGBM ayrıntıları" in report
    assert "Önceden yazılı başarı kriteri" in report
    assert not (out / "ridge_coefficients.csv").exists()


def test_shuffled_validation_targets_do_not_change_predictions(panel):
    one_fold = replace(panel, folds=panel.folds[:1])
    rows = one_fold.rows.copy()
    val = rows["partition"].eq("validation")
    rng = np.random.default_rng(11)
    for column in one_fold.targets.values():
        rows.loc[val, column] = rng.permutation(rows.loc[val, column].to_numpy())
    shuffled = replace(one_fold, rows=rows)
    original = evaluate(one_fold, EXPERIMENTS["4a"])["predictions"]
    permuted = evaluate(shuffled, EXPERIMENTS["4a"])["predictions"]
    pd.testing.assert_series_equal(original["pred"], permuted["pred"])
    assert not np.allclose(original["y"], permuted["y"])


def test_deney4b_market_features_and_same_row_4a_reference(runs):
    root, results = runs
    config, summary, _ = results["4b"]
    out = root / "deney4b"
    assert config["feature_set"] == "raw_market"
    assert config["features"] == [*results["4a"][0]["features"], *config["market_features"]]
    assert len(config["market_features"]) == 6
    assert "market_qqq_return_20" not in config["market_features"]
    assert {r["missing_any"] for r in config["market_coverage"]["summary"]} == {0}
    # Satır kümesi 4a ile aynı: "lightgbm_4a" referansı Deney 4a'nın LightGBM'iyle birebir aynı.
    p4b = pd.read_parquet(out / "predictions.parquet")
    p4a = pd.read_parquet(root / "deney4a" / "predictions.parquet")
    ref = p4b.loc[p4b["model"].eq("lightgbm_4a")].reset_index(drop=True)
    own = p4a.loc[p4a["model"].eq("lightgbm")].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        ref[["date", "symbol", "fold", "pred", "y"]], own[["date", "symbol", "fold", "pred", "y"]]
    )
    trees = pd.read_csv(out / "lgbm_trees.csv")
    assert set(trees["model"]) == {"lightgbm", "lightgbm_4a"}
    curve = pd.read_csv(out / "lgbm_tree_curve.csv")
    assert set(curve["model"]) == {"lightgbm_4a"}
    dependence = pd.read_csv(out / "lgbm_partial_dependence.csv")
    assert set(dependence["condition"]) == {"market_qqq_sma200_dist", "market_qqq_return_20"}
    report = (out / "RAPOR.md").read_text(encoding="utf-8")
    for heading in ("early stopping incelemesi", "6 yeni piyasa durumu", "## 8. Yeni özelliklerin"):
        assert heading in report
    assert summary.set_index("model").loc["lightgbm_4a", "kind"] == "forecast"
