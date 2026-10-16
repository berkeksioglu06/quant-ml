"""Yerel özellik prototipi. Sinyal gününün kapanışından SONRA kullanılabilir.

Oranlar ondalıktır: 0.05 = %5. Eğitim etiketi veya işlem kararı üretmez.
Mutlak fiyat seviyeleri kullanılmaz; fiyat özellikleri düzeltilmiş OHLC'den
hesaplanan ölçekten bağımsız oranlardır. Tarihsel sağlayıcı revizyonları için
point-in-time arşivinin yerine geçmez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import us_sessions

VERSION = "1.1-review"
HORIZONS = (1, 5, 10, 20)
AVERAGES = (10, 20, 60)
WARMUP = 200
PRICE_FEATURES = (
    [f"return_{w}" for w in HORIZONS]
    + [f"sma_distance_{w}" for w in AVERAGES]
    + [f"sma_slope_{w}_5" for w in AVERAGES]
    + [
        "volatility_10",
        "volatility_20",
        "range_ratio",
        "gap_return",
        "intraday_return",
        "atr_14_ratio",
    ]
    + [f"relative_{b}_{w}" for b in ("spy", "qqq") for w in (5, 20)]
    + ["market_spy_return_20", "market_qqq_return_20"]
)
VOLUME_FEATURES = ["relative_volume_20", "volume_change_1"]
FEATURES = PRICE_FEATURES + VOLUME_FEATURES


def _validate_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DatetimeIndex gerekli")
    if df.index.tz is not None or df.index.hasnans:
        raise ValueError("Tarihler saat dilimsiz olmalı ve NaT içermemeli")
    if df.index.has_duplicates or not df.index.equals(df.index.normalize()):
        raise ValueError("Tekil, gece yarısına normalize edilmiş tarihler gerekli")


def _grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    _validate_index(df)
    required = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "eligible",
        "is_session",
        "splits",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")
    if df.empty:
        raise ValueError("Boş fiyat tablosu")
    df = df.sort_index()
    sessions, source = us_sessions(df.index.min(), df.index.max())
    if source == "YOK":
        raise RuntimeError("Seans takvimi gerekli; iş günü varsayımı kullanılmaz")
    x = df.reindex(sessions).copy()
    active = x["eligible"].eq(True) & x["is_session"].eq(True)
    prices = x[["open", "high", "low", "close", "adj_close"]]
    good = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    tol = x["close"].abs() * 1e-6
    good &= x["high"] + tol >= x[["open", "close", "low"]].max(axis=1)
    good &= x["low"] - tol <= x[["open", "close"]].min(axis=1)
    valid = (active & good).fillna(False)
    return x, valid


def _return(close: pd.Series, window: int) -> pd.Series:
    # Uç noktalar mevcut olsa bile aradaki eksik seans üzerinden getiri hesaplanmaz.
    complete = close.rolling(window + 1, min_periods=window + 1).count().eq(window + 1)
    return (close / close.shift(window) - 1).where(complete)


def make_features(
    frame: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame],
    symbol: str,
    *,
    training_member: bool = True,
    quality_status: str = "UNREVIEWED",
    warmup_sessions: int = WARMUP,
) -> pd.DataFrame:
    if type(warmup_sessions) is not int or warmup_sessions < 0:
        raise ValueError("Isınma seans sayısı sıfır veya pozitif tam sayı olmalı")
    x, valid = _grid(frame)
    factor = (x["adj_close"] / x["close"]).where(valid)
    adjusted = x[["open", "high", "low", "close"]].mul(factor, axis=0)
    c, o, h, low = (adjusted[k] for k in ("close", "open", "high", "low"))
    values = {f"return_{w}": _return(c, w) for w in HORIZONS}
    for w in AVERAGES:
        mean = c.rolling(w, min_periods=w).mean()
        values[f"sma_distance_{w}"] = c / mean - 1
        # İki pencere arasındaki eksik seansları da reddet.
        complete = c.rolling(w + 5, min_periods=w + 5).count().eq(w + 5)
        values[f"sma_slope_{w}_5"] = (mean / mean.shift(5) - 1).where(complete & valid)
    log_return = np.log(c / c.shift(1))
    for w in (10, 20):
        values[f"volatility_{w}"] = log_return.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(252)
    values["range_ratio"] = (h - low) / c
    values["gap_return"] = o / c.shift(1) - 1
    values["intraday_return"] = c / o - 1
    true_range = pd.concat([h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()], axis=1)
    # İlk/eksik önceki seans için yalnız high-low ile ATR uydurulmaz.
    tr = true_range.max(axis=1).where(c.notna() & c.shift(1).notna())
    values["atr_14_ratio"] = tr.rolling(14, min_periods=14).mean() / c
    for name in ("SPY", "QQQ"):
        if name not in benchmarks:
            raise ValueError(f"Piyasa karşılaştırması için {name} gerekli")
        bx, bv = _grid(benchmarks[name])
        bc = bx["adj_close"].where(bv).reindex(x.index)
        for w in (5, 20):
            br = _return(bc, w)
            values[f"relative_{name.lower()}_{w}"] = values[f"return_{w}"] - br
            if w == 20:
                values[f"market_{name.lower()}_return_20"] = br

    # Göreli sağlayıcı hacmi: pencere boyunca tutarlı split ölçeği varsayılır.
    # Bu oranlar gerçek tarihsel hisse adedi veya raw_volume doğrulaması değildir.
    volume = x["volume"]
    volume = volume.where(valid & np.isfinite(volume) & volume.ge(0))
    prior = volume.shift(1)
    mean_volume = prior.rolling(20, min_periods=20).mean()
    values["relative_volume_20"] = volume / mean_volume.where(mean_volume.gt(0))
    values["volume_change_1"] = volume / prior.where(prior.gt(0)) - 1

    out = pd.DataFrame(values, index=x.index).reindex(columns=FEATURES)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[~valid, FEATURES] = np.nan
    out["symbol"] = symbol
    out["price_valid"] = valid
    out["eligible"] = x["eligible"].eq(True)
    out["prior_valid_sessions"] = valid.cumsum().shift(1, fill_value=0)
    out["warmup_ready"] = valid & out["prior_valid_sessions"].ge(warmup_sessions)
    out["price_features_ready"] = out[PRICE_FEATURES].notna().all(axis=1)
    out["volume_features_ready"] = out[VOLUME_FEATURES].notna().all(axis=1)
    out["quality_status"] = quality_status
    # Yalnız özellik hesabı için adaydır; etiketi veya eğitim onayı değildir.
    out["price_sample_candidate"] = (
        out["warmup_ready"]
        & out["price_features_ready"]
        & training_member
        & (quality_status in {"OK", "UYARI"})
    )
    out.index.name = "date"
    return out.loc[out["eligible"]].reset_index()
