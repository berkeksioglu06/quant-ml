import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.data import load_panel
from research.experiment import run
from research.metrics import daily_rank_ic, newey_west_tstat, top_minus_all
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


def test_experiment_writes_report_and_registry(panel, tmp_path):
    out = tmp_path / "experiments" / "run1"
    config, summary, consistency = run(panel, out)
    assert config["test_period_read"] is False
    assert set(summary["model"]) == {
        "sifir",
        "egitim_ortalamasi",
        "momentum_20",
        "geri_donus_5",
        "ridge",
    }
    assert consistency["folds"].eq(4).all()
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert predictions["date"].max() < panel.test_start
    report = (out / "RAPOR.md").read_text(encoding="utf-8")
    assert "2025-01-01 ve sonrası (test dönemi) okunmadı" in report
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["horizon"] == 20
    registry = pd.read_csv(tmp_path / "experiments" / "runs.csv")
    assert registry["run_id"].tolist() == ["run1"]
    with pytest.raises(ValueError, match="zaten var"):
        run(panel, out)
