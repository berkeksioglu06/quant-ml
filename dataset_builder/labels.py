"""Kapanış sonrası sinyal -> ertesi seans açılışı -> t+H kapanışı.

Saf etiket modülü; eski quant_ml.data paketine bağımlı değildir.
Ondalık getiri üretir; sınıf eşiği veya kısa satış varsayımı seçmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import exchange_calendars as xc
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Alış ve satıştaki toplam oransal maliyetler; 1 bps = 0.0001."""

    buy_bps: float
    sell_bps: float

    def __post_init__(self):
        if not all(np.isfinite(v) and 0 <= v < 10000 for v in (self.buy_bps, self.sell_bps)):
            raise ValueError("Maliyetler sonlu ve 0 <= bps < 10000 olmalı")


@lru_cache(maxsize=1)
def _calendar():
    return xc.get_calendar("XNYS", start="2000-01-01")


def make_labels(
    frame: pd.DataFrame,
    symbol: str,
    horizons: tuple[int, ...] = (5, 10, 20),
    *,
    as_of: str | pd.Timestamp | None = None,
    costs: Costs | None = None,
) -> pd.DataFrame:
    """Tamamlanmış temiz günlük tabloyu okur; girdi tablosunu değiştirmez.

    Etiket için t ve t+1..t+H fiyatları geçerli olmalıdır. Arada tek eksik/bozuk
    seans varsa etiket boş kalır. Ufuk birimi işlem seansıdır, takvim günü değil.
    as_of, o tarihten sonrasını henüz bilinmiyormuş gibi tamamen dışarıda bırakır.
    """
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Ufuklar pozitif tam sayı olmalı")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Tekrarlanan ufuk olamaz")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("DatetimeIndex gerekli")
    if (
        frame.index.tz is not None
        or frame.index.hasnans
        or frame.index.has_duplicates
        or not frame.index.equals(frame.index.normalize())
    ):
        raise ValueError(
            "Tarihler tekil, normalize, saat dilimsiz ve NaT içermeyen tarihler olmalı"
        )
    required = {"open", "high", "low", "close", "adj_close", "eligible", "is_session"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Boş fiyat tablosu")
    x = frame.sort_index().copy()
    cutoff = x.index.max() if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is not None or pd.isna(cutoff):
        raise ValueError("as_of saat dilimsiz geçerli tarih olmalı")
    cutoff = cutoff.normalize()
    x = x.loc[:cutoff]
    if x.empty:
        raise ValueError("as_of tarihine kadar veri yok")
    # Etiketi henüz bilinmeyen son günler için bile planlanan çıkış tarihi korunur.
    sessions = (
        _calendar()
        .sessions_in_range(
            x.index.min(),
            x.index.max() + pd.Timedelta(days=max(horizons) * 3 + 10),
        )
        .tz_localize(None)
    )
    grid = x.reindex(sessions)
    prices = grid[["open", "high", "low", "close", "adj_close"]]
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    tol = grid["close"].abs() * 1e-6
    valid &= grid["high"] + tol >= grid[["open", "close", "low"]].max(axis=1)
    valid &= grid["low"] - tol <= grid[["open", "close"]].min(axis=1)
    active = grid["eligible"].eq(True) & grid["is_session"].eq(True)
    valid = (valid & active).fillna(False)
    adj_open = (grid["open"] * grid["adj_close"] / grid["close"]).where(valid)
    close = grid["adj_close"].where(valid)
    dates = pd.Series(sessions, index=sessions)
    entry = adj_open.shift(-1)
    out = pd.DataFrame({"symbol": symbol, "entry_date": dates.shift(-1)}, index=sessions)
    for h in horizons:
        complete = valid.astype(int).rolling(h, min_periods=h).sum().shift(-h).eq(h)
        ready = valid & complete & entry.notna()
        end = dates.shift(-h)
        gross = (close.shift(-h) / entry - 1).where(ready)
        out[f"label_end_{h}"] = end
        out[f"label_ready_{h}"] = ready
        out[f"target_return_{h}"] = gross
        out[f"target_long_net_{h}"] = (
            ((1 + gross) * (1 - costs.sell_bps / 10000) / (1 + costs.buy_bps / 10000) - 1)
            if costs is not None
            else np.nan
        )
        status = pd.Series("ready", index=sessions)
        status.loc[~ready & end.le(cutoff)] = "invalid_or_missing_bar"
        status.loc[end.gt(cutoff)] = "future_not_available"
        status.loc[~valid] = "invalid_signal"
        out[f"label_status_{h}"] = status
    out = out.loc[active & out.index.isin(x.index)].copy()
    out.index.name = "date"
    if np.isinf(out.filter(regex="^target_").to_numpy()).any():
        raise ValueError("Etikette sonsuz değer oluştu")
    return out.reset_index()


def partition_mask(
    table: pd.DataFrame,
    horizon: int,
    start: str,
    end: str,
    *,
    require_training_member: bool = True,
) -> pd.Series:
    """Etiketi dönem sınırını aşan örnekler dahil edilmez (purge).

    Adaylık yalnız özellik tablosundan gelir; gelecek getiri büyüklüğü ile seçilmez.
    Validation için de aynı katı sınır uygulanır. NBIS adaylık flag'i false kalır.
    """
    mask = (
        table["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & table[f"label_ready_{horizon}"]
        & table[f"label_end_{horizon}"].le(pd.Timestamp(end))
    )
    if require_training_member:
        mask &= table["price_sample_candidate"]
    return mask.fillna(False)
