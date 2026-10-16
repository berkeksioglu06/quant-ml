import numpy as np
import pandas as pd
import pytest

from dataset_builder.labels import Costs, make_labels, partition_mask


def frame(n=50):
    import exchange_calendars as xc

    index = xc.get_calendar("XNYS").sessions_in_range("2023-11-01", "2024-03-01")[:n]
    close = 100.0 + np.arange(n)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "adj_close": close,
            "eligible": True,
            "is_session": True,
        },
        index=index,
    )


def test_entry_is_next_open_and_exit_is_t_plus_h_close():
    x = frame()
    out = make_labels(x, "TEST", (1, 5, 10, 20)).set_index("date")
    t = 3
    for h in (1, 5, 10, 20):
        row = out.iloc[t]
        assert row.entry_date == x.index[t + 1]
        assert row[f"label_end_{h}"] == x.index[t + h]
        assert row[f"target_return_{h}"] == pytest.approx(
            x.close.iloc[t + h] / x.open.iloc[t + 1] - 1
        )
        assert row[f"label_ready_{h}"]


def test_signal_close_is_not_used_as_entry():
    x = frame()
    x.loc[x.index[1], ["open", "high", "low", "close", "adj_close"]] = 200.0
    out = make_labels(x, "TEST", (5,))
    assert out.target_return_5.iloc[0] == pytest.approx(105 / 200 - 1)


def test_holidays_are_sessions_not_calendar_days():
    x = frame()
    out = make_labels(x, "TEST", (1,)).set_index("date")
    assert out.loc["2023-11-22", "entry_date"] == pd.Timestamp("2023-11-24")
    assert out.loc["2023-11-24", "entry_date"] == pd.Timestamp("2023-11-27")


def test_tail_labels_remain_missing():
    x = frame()
    out = make_labels(x, "TEST")
    for h in (5, 10, 20):
        assert out[f"target_return_{h}"].iloc[-h:].isna().all()
        assert out[f"label_status_{h}"].iloc[-h:].eq("future_not_available").all()
        assert not out[f"label_ready_{h}"].iloc[-h:].any()
    assert out.label_end_20.iloc[-1] > x.index[-1]


@pytest.mark.parametrize("broken", ["missing", "bad_high", "zero_price", "ineligible"])
def test_bad_or_missing_bar_anywhere_in_future_invalidates_label(broken):
    x = frame()
    d = x.index[3]
    if broken == "missing":
        x = x.drop(index=d)
    elif broken == "bad_high":
        x.loc[d, "high"] = 1
    elif broken == "zero_price":
        x.loc[d, "close"] = 0
    else:
        x.loc[d, "eligible"] = False
    out = make_labels(x, "TEST", (5,))
    assert pd.isna(out.target_return_5.iloc[0])
    assert out.label_status_5.iloc[0] == "invalid_or_missing_bar"


def test_as_of_does_not_read_later_prices():
    x = frame()
    cutoff = x.index[20]
    expected = make_labels(x.loc[:cutoff], "TEST")
    changed = x.copy()
    changed.loc[changed.index > cutoff, "adj_close"] *= 999
    actual = make_labels(changed, "TEST", as_of=cutoff)
    pd.testing.assert_frame_equal(actual, expected)


def test_settled_labels_do_not_change_when_future_rows_arrive():
    x = frame()
    old = make_labels(x.iloc[:30], "TEST")
    new = make_labels(x, "TEST")
    for h in (5, 10, 20):
        ready = old[f"label_ready_{h}"]
        np.testing.assert_allclose(
            old.loc[ready, f"target_return_{h}"],
            new.loc[old.index[ready], f"target_return_{h}"],
        )


def test_split_adjusted_prices_do_not_create_fake_return():
    x = frame()
    x[["open", "high", "low", "close", "adj_close"]] = 50.0
    # Split sonrası sağlayıcı zaten aynı fiyat ölçeğinde; ikinci kez split uygulanmaz.
    x["splits"] = 0.0
    x.loc[x.index[3], "splits"] = 2.0
    out = make_labels(x, "TEST", (5,))
    assert out.target_return_5.iloc[0] == 0.0


def test_dividend_adjustment_is_consistent_for_entry_and_exit():
    x = frame()
    x[["open", "high", "low", "close", "adj_close"]] = 100.0
    x.loc[x.index[:3], "adj_close"] = 99.0
    x.loc[x.index[3] :, ["open", "high", "low", "close", "adj_close"]] = 99.0
    out = make_labels(x, "TEST", (5,))
    assert out.target_return_5.iloc[0] == 0.0


def test_costs_are_explicit_and_calculated_at_both_sides():
    x = frame()
    unconfigured = make_labels(x, "TEST", (5,))
    assert unconfigured.target_long_net_5.isna().all()
    out = make_labels(x, "TEST", (5,), costs=Costs(buy_bps=10, sell_bps=20))
    assert out.target_long_net_5.iloc[0] == pytest.approx(105 * 0.998 / (100 * 1.001) - 1)
    assert out.target_long_net_5.iloc[0] < out.target_return_5.iloc[0]


@pytest.mark.parametrize("buy,sell", [(-1, 0), (0, 10000), (np.nan, 0), (0, np.inf)])
def test_bad_costs_are_rejected(buy, sell):
    with pytest.raises(ValueError, match="Maliyetler"):
        Costs(buy, sell)


def test_purge_excludes_labels_crossing_period_end():
    x = frame()
    out = make_labels(x, "TEST", (5,))
    out["price_sample_candidate"] = True
    end = x.index[20]
    mask = partition_mask(out, 5, str(x.index[0].date()), str(end.date()))
    assert mask.sum() == 16
    assert not mask.iloc[16:21].any()
    assert out.loc[mask, "label_end_5"].le(end).all()
    out.loc[0, "price_sample_candidate"] = False
    assert partition_mask(out, 5, str(x.index[0].date()), str(end.date())).sum() == 15


def test_inputs_are_not_changed():
    x = frame()
    before = x.copy(deep=True)
    make_labels(x, "TEST")
    pd.testing.assert_frame_equal(x, before)


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (2.5,), (True,), (5, 5)])
def test_bad_horizons_are_rejected(horizons):
    with pytest.raises(ValueError):
        make_labels(frame(), "TEST", horizons)


def test_duplicate_dates_are_rejected():
    x = frame()
    with pytest.raises(ValueError, match="tekil"):
        make_labels(pd.concat([x, x.iloc[[0]]]), "TEST")
