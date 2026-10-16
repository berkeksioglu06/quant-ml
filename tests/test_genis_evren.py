"""Kural tabanlı geniş evren: sembol ayıklama ve günlük kural ölçüleri."""

import numpy as np
import pandas as pd

import genis_evren as g

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
RKLB|Rocket Lab Corporation - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZT|Test Company - Common Stock|Q|Y|N|100|N|N
ACMEW|Acme Corp - Warrant|Q|N|N|100|N|N
ACMEU|Acme Corp - Units|Q|N|N|100|N|N
BABA|Alibaba Group Holding Limited American Depositary Shares|Q|N|N|100|N|N
File Creation Time: 1016202614:01|||||||
"""
OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B
JPM$C|JPMorgan Chase 6% Depositary Shares, each representing a 1/400th Preferred|N|JPMpC|N|100|N|JPM$C
MUJ|Blackrock MuniHoldings New Jersey Quality Fund, Inc. Common Stock|N|MUJ|N|100|N|MUJ
KO|Coca-Cola Company (The) Common Stock|N|KO|N|100|N|KO
File Creation Time: 1016202614:01||||||
"""


def test_directory_keeps_only_common_stock_in_yahoo_format():
    pool = g.parse_directory(NASDAQ, OTHER)
    assert sorted(pool["yahoo"]) == ["BABA", "BRK-B", "KO", "RKLB"]


def _daily(n=400, price=20.0, volume=2e6, daily_vol=0.03, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = price * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))
    return pd.DataFrame(
        {"close": close, "adj_close": close, "volume": np.full(n, volume), "split_factor": 1.0},
        index=idx,
    )


def test_rule_uses_only_past_data():
    base = _daily()
    changed = base.copy()
    changed.iloc[300:, :3] *= 7  # t=299 sonrasını değiştir
    a, b = g.rule_metrics(base), g.rule_metrics(changed)
    pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])


def test_raw_price_and_dollar_volume_are_split_invariant():
    raw = _daily(price=60.0)
    # 10:1 split ileride: sağlayıcı geçmiş fiyatı /10, hacmi ×10 gösterir; split_factor = 10.
    adjusted = raw.assign(
        close=raw["close"] / 10,
        adj_close=raw["adj_close"] / 10,
        volume=raw["volume"] * 10,
        split_factor=10.0,
    )
    a, b = g.rule_metrics(raw), g.rule_metrics(adjusted)
    np.testing.assert_allclose(a["raw_close"], b["raw_close"])
    np.testing.assert_allclose(a["dollar_volume_med"].dropna(), b["dollar_volume_med"].dropna())
    assert a["in_universe"].equals(b["in_universe"])


def test_each_condition_can_exclude():
    ok = g.rule_metrics(_daily(price=20.0, volume=2e6, daily_vol=0.03))
    assert not ok["in_universe"].iloc[: g.MIN_HISTORY - 1].any()  # geçmiş kısa
    assert ok["in_universe"].iloc[-50:].all()
    assert not g.rule_metrics(_daily(price=3.0, volume=2e7))["in_universe"].any()  # fiyat < 5$
    assert not g.rule_metrics(_daily(volume=1e5))["in_universe"].any()  # dolar hacmi düşük
    # Oynaklık kuralda yok: sakin bir hisse de likitse evrende.
    assert g.rule_metrics(_daily(daily_vol=0.005))["in_universe"].iloc[-50:].all()


def test_universe_entries_keep_known_metadata_and_mark_late_listings():
    template = {
        "symbols": [
            {
                "symbol": "RKLB",
                "roles": ["ana20"],
                "asset": "equity",
                "eligible_from": "2021-08-25",
            },
            {"symbol": "QQQ", "roles": ["baglam"], "asset": "etf"},
        ]
    }
    first = {"RKLB": "2020-11-24", "AAA": "2015-01-02", "BBB": "2019-06-03"}
    entries = {e["symbol"]: e for e in g.universe_entries(first, first, template, "2015-01-01")}
    assert (
        entries["RKLB"]["eligible_from"] == "2021-08-25" and "genis" not in entries["RKLB"]["roles"]
    )
    assert entries["AAA"] == {"symbol": "AAA", "roles": ["genis"], "asset": "equity"}
    assert entries["BBB"]["expected_first_date"] == "2019-06-03"
    assert entries["BBB"]["first_date_source"] == "yahoo_first_bar"
