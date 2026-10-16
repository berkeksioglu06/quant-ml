import math
import statistics

import numpy as np
import pandas as pd
import pytest

from dataset_builder.common import us_sessions
from dataset_builder.features import FEATURES, PRICE_FEATURES, VOLUME_FEATURES, make_features


def sample(n=320, rate=0.001):
    index = us_sessions("2022-01-03", "2024-01-01")[0][:n]
    t = np.arange(n)
    close = 100 * np.exp(rate * t + np.sin(t / 9) * 0.01)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "adj_close": close * 0.97,
            "raw_volume": 1000.0 + t * 3,
            "volume": 1000.0 + t * 3,
            "eligible": True,
            "is_session": True,
            "splits": 0.0,
        },
        index=index,
    )


@pytest.fixture
def data():
    return sample(), {"SPY": sample(rate=0.0003), "QQQ": sample(rate=0.0006)}


def calculate(frame, benchmarks, **kwargs):
    return make_features(frame, benchmarks, "TEST", quality_status="OK", **kwargs).set_index("date")


def test_all_24_features_match_independent_scalar_calculation(data):
    frame, benchmarks = data
    actual = calculate(frame, benchmarks).iloc[-1]
    c = frame.adj_close.to_list()
    expected = {}
    for w in (1, 5, 10, 20):
        expected[f"return_{w}"] = c[-1] / c[-1 - w] - 1
    for w in (10, 20, 60):
        now = statistics.mean(c[-w:])
        before = statistics.mean(c[-w - 5 : -5])
        expected[f"sma_distance_{w}"] = c[-1] / now - 1
        expected[f"sma_slope_{w}_5"] = now / before - 1
    for w in (10, 20):
        logs = [math.log(c[j] / c[j - 1]) for j in range(len(c) - w, len(c))]
        expected[f"volatility_{w}"] = statistics.stdev(logs) * math.sqrt(252)
    expected["range_ratio"] = (frame.high.iloc[-1] - frame.low.iloc[-1]) / frame.close.iloc[-1]
    expected["gap_return"] = frame.open.iloc[-1] * 0.97 / c[-2] - 1
    expected["intraday_return"] = frame.close.iloc[-1] / frame.open.iloc[-1] - 1
    ranges = []
    for j in range(len(c) - 14, len(c)):
        high, low = frame.high.iloc[j] * 0.97, frame.low.iloc[j] * 0.97
        ranges.append(max(high - low, abs(high - c[j - 1]), abs(low - c[j - 1])))
    expected["atr_14_ratio"] = statistics.mean(ranges) / c[-1]
    for name, benchmark in benchmarks.items():
        b = benchmark.adj_close.to_list()
        for w in (5, 20):
            br = b[-1] / b[-1 - w] - 1
            expected[f"relative_{name.lower()}_{w}"] = expected[f"return_{w}"] - br
            if w == 20:
                expected[f"market_{name.lower()}_return_20"] = br
    v = frame.volume.to_list()
    expected["relative_volume_20"] = v[-1] / statistics.mean(v[-21:-1])
    expected["volume_change_1"] = v[-1] / v[-2] - 1
    assert set(expected) == set(FEATURES)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-10, abs=1e-12), key


def test_warmup_can_be_configured(data):
    frame, benchmarks = data
    actual = make_features(frame, benchmarks, "TEST", quality_status="OK", warmup_sessions=250)
    assert not actual.warmup_ready.iloc[249]
    assert actual.warmup_ready.iloc[250]


def test_future_mutation_cannot_change_past_features(data):
    frame, benchmarks = data
    cutoff = frame.index[240]
    baseline = calculate(frame.loc[:cutoff], {k: v.loc[:cutoff] for k, v in benchmarks.items()})
    future = frame.copy()
    future.loc[future.index > cutoff, ["open", "high", "low", "close", "adj_close"]] *= 123
    future.loc[future.index > cutoff, "volume"] = 999999999
    changed_benchmarks = {k: v.copy() for k, v in benchmarks.items()}
    for b in changed_benchmarks.values():
        b.loc[b.index > cutoff, "adj_close"] *= 456
    actual = calculate(future, changed_benchmarks).loc[:cutoff]
    pd.testing.assert_frame_equal(actual, baseline)


def test_uniform_historical_adjustment_does_not_change_features(data):
    frame, benchmarks = data
    baseline = calculate(frame, benchmarks)
    rescaled = frame.copy()
    rescaled["adj_close"] *= 0.1
    actual = calculate(rescaled, benchmarks)
    pd.testing.assert_frame_equal(actual, baseline, rtol=1e-10, atol=1e-12)


def test_missing_session_is_not_bridged(data):
    frame, benchmarks = data
    gap = frame.index[250]
    actual = calculate(frame.drop(index=gap), benchmarks)
    assert gap not in actual.index
    assert pd.isna(actual.loc[frame.index[251], "return_1"])
    assert pd.isna(actual.loc[frame.index[255], "return_5"])
    assert pd.notna(actual.loc[frame.index[256], "return_5"])
    assert pd.isna(actual.loc[frame.index[264], "atr_14_ratio"])
    assert pd.notna(actual.loc[frame.index[265], "atr_14_ratio"])


def test_benchmark_alignment_is_by_date_and_does_not_fill(data):
    frame, benchmarks = data
    gap = frame.index[250]
    benchmarks["SPY"] = benchmarks["SPY"].drop(index=gap)
    actual = calculate(frame, benchmarks)
    assert pd.isna(actual.loc[frame.index[255], "relative_spy_5"])
    assert pd.notna(actual.loc[frame.index[256], "relative_spy_5"])
    assert pd.notna(actual.loc[frame.index[255], "relative_qqq_5"])


@pytest.mark.parametrize("ratio", [2, 3, 4, 5, 10, 20])
def test_adjusted_provider_volume_crosses_split_without_reset(data, ratio):
    frame, benchmarks = data
    frame["volume"] = 1000.0
    frame["raw_volume"] = np.nan
    frame.loc[frame.index[250], "splits"] = ratio
    actual = calculate(frame, benchmarks)
    assert actual.loc[frame.index[250:271], "relative_volume_20"].eq(1.0).all()
    assert actual.loc[frame.index[250:271], "volume_change_1"].eq(0.0).all()
    assert actual.loc[frame.index[250:271], PRICE_FEATURES].notna().all().all()


def test_raw_volume_is_not_used_for_relative_features(data):
    frame, benchmarks = data
    baseline = calculate(frame, benchmarks)
    frame = frame.drop(columns="raw_volume")
    actual = calculate(frame, benchmarks)
    pd.testing.assert_frame_equal(actual, baseline)


def test_missing_provider_volume_does_not_block_price_features(data):
    frame, benchmarks = data
    frame["volume"] = np.nan
    actual = calculate(frame, benchmarks)
    assert actual[VOLUME_FEATURES].isna().all().all()
    assert actual.price_sample_candidate.iloc[-1]
    assert not actual.volume_features_ready.any()


def test_constant_provider_volume_scale_does_not_change_ratios(data):
    frame, benchmarks = data
    baseline = calculate(frame, benchmarks)
    frame["volume"] *= 20
    actual = calculate(frame, benchmarks)
    pd.testing.assert_frame_equal(actual, baseline, rtol=1e-10, atol=1e-12)


def test_spac_history_does_not_count_as_warmup(data):
    frame, benchmarks = data
    frame.loc[frame.index[:100], "eligible"] = False
    actual = calculate(frame, benchmarks)
    assert actual.index.min() == frame.index[100]
    assert pd.isna(actual.iloc[0]["return_1"])
    assert not actual.loc[frame.index[299], "warmup_ready"]
    assert actual.loc[frame.index[300], "warmup_ready"]


def test_invalid_prices_are_masked_and_cannot_bridge_windows(data):
    frame, benchmarks = data
    frame.loc[frame.index[250], "high"] = 1
    actual = calculate(frame, benchmarks)
    assert not actual.loc[frame.index[250], "price_valid"]
    assert actual.loc[frame.index[250], FEATURES].isna().all()
    assert pd.isna(actual.loc[frame.index[251], "return_1"])
    assert not np.isinf(actual[FEATURES].to_numpy()).any()


def test_zero_denominator_never_becomes_infinity(data):
    frame, benchmarks = data
    frame["volume"] = 0.0
    actual = calculate(frame, benchmarks)
    assert actual[VOLUME_FEATURES].isna().all().all()
    assert not np.isinf(actual[FEATURES].to_numpy()).any()


def test_duplicate_dates_are_rejected(data):
    frame, benchmarks = data
    with pytest.raises(ValueError, match="Tekil"):
        calculate(pd.concat([frame, frame.iloc[[-1]]]), benchmarks)


def test_non_session_rows_cannot_change_features(data):
    frame, benchmarks = data
    baseline = calculate(frame, benchmarks)
    weekend = frame.iloc[[0]].copy()
    weekend.index = pd.DatetimeIndex(["2022-01-08"])
    actual = calculate(pd.concat([frame, weekend]), benchmarks)
    pd.testing.assert_frame_equal(actual, baseline)


def test_training_exclusions_and_quality_gate(data):
    frame, benchmarks = data
    nbis = calculate(frame, benchmarks, training_member=False)
    assert not nbis.price_sample_candidate.any()
    bad = make_features(frame, benchmarks, "TEST", quality_status="HATA")
    assert not bad.price_sample_candidate.any()


def test_relative_volume_uses_previous_20_days_not_current_day(data):
    frame, benchmarks = data
    frame["volume"] = 1000.0
    frame.loc[frame.index[-1], "volume"] = 100000.0
    actual = calculate(frame, benchmarks)
    assert actual.relative_volume_20.iloc[-1] == 100.0


def test_intraday_adjustment_handles_dividend_without_fake_gap(data):
    frame, benchmarks = data
    frame[["open", "high", "low", "close", "adj_close"]] = 100.0
    # %1 temettü: önceki fiyatlar 0.99 ile düzeltilir, ex-date fiyatı 99 olur.
    frame.loc[: frame.index[249], "adj_close"] = 99.0
    frame.loc[frame.index[250] :, ["open", "high", "low", "close", "adj_close"]] = 99.0
    actual = calculate(frame, benchmarks)
    assert actual.loc[frame.index[250], "return_1"] == 0.0
    assert actual.loc[frame.index[250], "gap_return"] == 0.0
    assert actual.loc[frame.index[250], "atr_14_ratio"] == 0.0
