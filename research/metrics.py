"""Tahmin hatası, günlük sıralama başarısı ve örtüşen etiketler için t-istatistiği."""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_metrics(y, pred):
    """MSE ve sıfır tahminine göre R² (1 − SSE / Σy²)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sse = float(np.sum((y - pred) ** 2))
    return {"mse": sse / len(y), "r2_oos": 1 - sse / float(np.sum(y**2))}


def daily_rank_ic(frame, pred_col="pred", target_col="y", min_names=5):
    """Her gün hisseler arası Spearman korelasyonu; sabit tahminde NaN."""
    ranked = frame[["date"]].copy()
    grouped = frame.groupby("date")
    ranked["p"] = grouped[pred_col].rank()
    ranked["y"] = grouped[target_col].rank()
    counts = ranked.groupby("date")["p"].transform("size")
    ranked = ranked.loc[counts >= min_names]
    if ranked.empty:
        return pd.Series(dtype=float, name="rank_ic")
    ic = ranked.groupby("date")[["p", "y"]].corr().xs("p", level=1)["y"]
    return ic.rename("rank_ic")


def top_k_mean(frame, column, k=5, pred_col="pred"):
    """Günlük: en yüksek skorlu k hissede ``column`` ortalaması (eksikler atlanır)."""
    ordered = frame.sort_values(["date", pred_col], ascending=[True, False])
    return ordered.groupby("date").head(k).groupby("date")[column].mean()


def top_minus_all(frame, k=5, pred_col="pred", target_col="y"):
    """Günlük: en yüksek skorlu k hissenin ortalama hedefi − bütün hisselerin ortalaması.

    Maliyetsiz bir sıralama ölçüsüdür; portföy getirisi değildir.
    """
    top = top_k_mean(frame, target_col, k, pred_col)
    return (top - frame.groupby("date")[target_col].mean()).rename("top_minus_all")


def newey_west_tstat(series, lags):
    """Ortalamanın Newey-West (Bartlett) t-istatistiği.

    Ardışık günlerin H seanslık etiketleri örtüştüğü için günlük IC'ler bağımsız
    değildir; lags = H − 1 kullanılır.
    """
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    if s <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(s / n))


def summarize_ic(ic, lags):
    ic = pd.Series(ic).dropna()
    if ic.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_pos_frac": np.nan,
            "ic_t_nw": np.nan,
            "ic_days": 0,
        }
    std = float(ic.std(ddof=1))
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": std,
        "icir": float(ic.mean() / std) if std > 0 else np.nan,
        "ic_pos_frac": float((ic > 0).mean()),
        "ic_t_nw": newey_west_tstat(ic, lags),
        "ic_days": len(ic),
    }
