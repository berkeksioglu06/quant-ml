"""Yahoo Finance (yfinance) günlük veri indirici -> değiştirilmeyen parquet snapshot.

Seçimler (manifest'e yazılır):
  auto_adjust=False  -> Open/High/Low/Close sağlayıcının "düzeltilmemiş" serisi, Adj Close ayrı kolon
  back_adjust=False, repair=False -> sağlayıcı verisine dokunulmaz
  actions=True       -> Dividends / Stock Splits (/ Capital Gains) kolonları saklanır
Snapshot dosyaları yazıldıktan sonra salt-okunur yapılır. Temizleme ayrı katmandadır.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .common import load_universe, make_readonly, safe_name, sha256, write_json

HISTORY_KWARGS = {
    "interval": "1d",
    "auto_adjust": False,
    "back_adjust": False,
    "actions": True,
    "repair": False,
    "prepost": False,
    "keepna": False,
    "raise_errors": True,
}

_META_KEYS = [
    "currency",
    "symbol",
    "exchangeName",
    "fullExchangeName",
    "instrumentType",
    "firstTradeDate",
    "regularMarketTime",
    "exchangeTimezoneName",
    "timezone",
    "longName",
    "shortName",
    "dataGranularity",
    "gmtoffset",
]


# --------------------------------------------------------------------------- yfinance fetchers
def yahoo_fetch(symbol: str, start: str, end: str):
    import yfinance as yf

    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end, **HISTORY_KWARGS)
    meta = {}
    try:
        hm = t.history_metadata or {}
        meta = {k: hm.get(k) for k in _META_KEYS if k in hm}
        if isinstance(meta.get("firstTradeDate"), (int, float)):
            meta["firstTradeDate_iso"] = datetime.fromtimestamp(
                meta["firstTradeDate"], tz=UTC
            ).isoformat()
    except Exception as e:  # noqa: BLE001 - optional provider metadata must not abort prices
        meta = {"_error": repr(e)}
    return df, meta


def yahoo_earnings(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol).get_earnings_dates(limit=100)


def _versions() -> dict:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["yfinance", "pandas", "numpy", "pyarrow", "exchange_calendars", "curl_cffi"]:
        try:
            v[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            v[mod] = None
    return v


def _with_retry(fn: Callable, *args, tries: int = 4, base_sleep: float = 5.0):
    last = None
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - retry injected/provider failures and re-raise
            last = e
            msg = repr(e)
            # Veri yok / delisted -> tekrar denemenin anlamı yok
            if any(
                s in msg.lower()
                for s in ["no data found", "delisted", "no price data", "not found", "404"]
            ):
                break
            time.sleep(base_sleep * (3**i))
    raise last


# --------------------------------------------------------------------------- ana akış
def download_snapshot(
    data_dir: Path,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    earnings: bool = True,
    pause: float = 1.0,
    fetch_fn: Callable = yahoo_fetch,
    earnings_fn: Callable = yahoo_earnings,
    universe: dict | None = None,
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """clock: her sembolün indirme bitiş zamanını verir (test için enjekte edilebilir)."""
    u = universe or load_universe()
    clock = clock or (lambda: datetime.now(UTC))
    now = now or clock()
    start = start or u["start"]
    # yfinance'te end hariçtir; bugünü kapsamak için yarın
    end = end or (now.date() + timedelta(days=1)).isoformat()
    snap_id = "yahoo_" + now.strftime("%Y%m%dT%H%M%SZ")
    out = Path(data_dir) / "raw" / snap_id
    (out / "prices").mkdir(parents=True, exist_ok=True)
    (out / "earnings").mkdir(parents=True, exist_ok=True)

    wanted = symbols or [s["symbol"] for s in u["symbols"]]
    # Eski ticker kontrolü (SQ -> XYZ): legacy sembolü de ayrı indir
    legacy = {
        s["legacy_symbol"]: s["symbol"]
        for s in u["symbols"]
        if s.get("legacy_symbol") and s["symbol"] in wanted
    }
    todo = list(wanted) + [l for l in legacy if l not in wanted]

    manifest = {
        "snapshot_id": snap_id,
        "source": "Yahoo Finance via yfinance Ticker.history",
        "downloaded_at_utc": now.isoformat(),
        "request": {"start": start, "end_exclusive": end, **HISTORY_KWARGS},
        "versions": _versions(),
        "universe_version": u.get("version"),
        "legacy_symbols": legacy,
        "symbols": {},
    }

    for i, sym in enumerate(todo, 1):
        rec = {"symbol": sym, "is_legacy_of": legacy.get(sym)}
        t0 = time.time()
        try:
            df, meta = _with_retry(fetch_fn, sym, start, end)
            rec["meta"] = meta
            if df is None or len(df) == 0:
                rec["status"] = "BOS"
            else:
                df = df.copy()
                df.index.name = "Date"
                p = out / "prices" / f"{safe_name(sym)}.parquet"
                df.reset_index().to_parquet(p, index=False)
                make_readonly(p)
                idx = pd.DatetimeIndex(df.index)
                rec.update(
                    status="OK",
                    file=str(p.relative_to(out)),
                    sha256=sha256(p),
                    rows=len(df),
                    columns=list(df.columns),
                    first_date=str(idx.min().date()),
                    last_date=str(idx.max().date()),
                    index_tz=str(idx.tz),
                )
        except Exception as e:  # noqa: BLE001 - record provider failure in manifest
            rec.update(status="HATA", error=repr(e), trace=traceback.format_exc(limit=2))
        rec["download_seconds"] = round(time.time() - t0, 2)
        # Sembolün kendi indirme zamanı: seans tamamlanma kontrolü bununla yapılır
        rec["downloaded_at_utc"] = clock().isoformat()

        # Bilanço tarihi kaynağı denetimi (yalnız hisseler)
        info = next((s for s in u["symbols"] if s["symbol"] == sym), {"asset": "equity"})
        if earnings and info.get("asset") == "equity" and rec["status"] == "OK":
            try:
                ed = _with_retry(earnings_fn, sym, tries=2)
                if ed is not None and len(ed):
                    ed = ed.copy()
                    ed.index.name = ed.index.name or "Earnings Date"
                    p = out / "earnings" / f"{safe_name(sym)}.parquet"
                    ed.reset_index().to_parquet(p, index=False)
                    make_readonly(p)
                    rec["earnings"] = {
                        "status": "OK",
                        "rows": len(ed),
                        "file": str(p.relative_to(out)),
                        "sha256": sha256(p),
                        "downloaded_at_utc": clock().isoformat(),
                    }
                else:
                    rec["earnings"] = {"status": "BOS"}
            except Exception as e:  # noqa: BLE001 - record provider failure in manifest
                rec["earnings"] = {"status": "HATA", "error": repr(e)}

        manifest["symbols"][sym] = rec
        print(
            f"[{i:>2}/{len(todo)}] {sym:<8} {rec['status']:<5} "
            f"{rec.get('rows', '-'):>5} satır  {rec.get('first_date', '')} → {rec.get('last_date', '')}"
            + (f"  bilanço:{rec['earnings']['status']}" if "earnings" in rec else ""),
            flush=True,
        )
        if pause and fetch_fn is yahoo_fetch:
            time.sleep(pause)

    write_json(manifest, out / "manifest.json")
    make_readonly(out / "manifest.json")
    failed = [s for s, r in manifest["symbols"].items() if r["status"] != "OK"]
    print(f"\nSnapshot: {out}\nBaşarısız/boş: {failed or 'yok'}")
    return out
