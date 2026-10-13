"""Günlük piyasa verisi: indir -> temizle -> kontrol et -> kapsamayı say.

    uv run python market_data.py all                        # indir + build
    uv run python market_data.py download --symbols RKLB SPY
    uv run python market_data.py build [--snapshot yahoo_20261004T171338Z]

Çıktılar (data/ Git'e girmez):
    data/<snapshot>/raw/*.parquet     yfinance'in döndürdüğü tablo, dokunulmaz (salt-okunur)
    data/<snapshot>/manifest.json     indirme zamanı, ayarlar, sürümler, sha256
    data/<snapshot>/clean/*.parquet   temiz tablo (modelde bu kullanılır)
    data/<snapshot>/issues.csv        kalite bulguları: HATA / UYARI / BILGI
    data/<snapshot>/coverage.csv      hisse x katman eğitim örneği sayısı (ön sayım)

Kurallar: satır silinmez, boşluk doldurulmaz, split düzeltmesi ikinci kez yapılmaz.
Yalnız birebir tekrar satırlar ve indirme anında bitmemiş son seans çıkarılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UNIVERSE = json.loads((ROOT / "universe.json").read_text(encoding="utf-8"))
REFERENCES = ROOT / "reference_points.csv"
CAL = xc.get_calendar("XNYS", start="2015-01-01")

YF_OPTIONS = {
    "interval": "1d",
    "auto_adjust": False,  # Open/High/Low/Close + ayrı Adj Close
    "back_adjust": False,
    "repair": False,  # sağlayıcı verisine dokunma
    "actions": True,  # Dividends / Stock Splits / Capital Gains
}
COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
    "Capital Gains": "capital_gains",
}
SESSION_BUFFER = pd.Timedelta(minutes=15)  # kapanıştan sonra son mumun kesinleşmesi için pay
BIG_MOVE = 0.25
REF_TOL = {"price": 0.02, "volume": 0.15}  # hacimde kaynaklar arası fark (konsolide/birincil) büyük
TRAIN_ROLES = {"ana20", "kontrol8"}


def safe(symbol: str) -> str:
    return symbol.replace("^", "_")


def info(symbol: str) -> dict:
    return next(s for s in UNIVERSE["symbols"] if s["symbol"] == symbol)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_complete_session(now: datetime) -> pd.Timestamp:
    """`now` anında kapanışı (erken kapanış dahil) + 15 dk geçmiş son seans."""
    now = pd.Timestamp(now).tz_convert("UTC")
    today = now.tz_localize(None).normalize()
    recent = CAL.sessions_in_range(today - pd.Timedelta(days=14), today)
    for day in reversed(recent):
        if now >= CAL.session_close(day) + SESSION_BUFFER:
            return pd.Timestamp(day).tz_localize(None)
    raise RuntimeError("son 14 günde tamamlanmış seans yok")


def latest_snapshot() -> Path:
    snaps = sorted(DATA.glob("yahoo_*/manifest.json"))
    if not snaps:
        raise SystemExit("data/ altında snapshot yok; önce `download` çalıştır.")
    return snaps[-1].parent


# --------------------------------------------------------------------------- indirme
def fetch(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, **YF_OPTIONS)


def download(symbols: list[str] | None = None, pause: float = 1.0) -> Path:
    now = datetime.now(UTC)
    snap = DATA / f"yahoo_{now:%Y%m%dT%H%M%SZ}"
    (snap / "raw").mkdir(parents=True)
    symbols = symbols or [s["symbol"] for s in UNIVERSE["symbols"]]
    manifest = {
        "source": "Yahoo Finance via yfinance Ticker.history",
        "downloaded_at": now.isoformat(),
        "last_complete_session": str(last_complete_session(now).date()),
        "start": UNIVERSE["start"],
        "options": YF_OPTIONS,
        "versions": {
            p: version(p) for p in ["yfinance", "pandas", "numpy", "pyarrow", "exchange_calendars"]
        },
        "symbols": {},
    }
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}", flush=True)
        try:
            df = fetch(sym, UNIVERSE["start"])
            if df.empty:
                raise ValueError("boş tablo")
            path = snap / "raw" / f"{safe(sym)}.parquet"
            df.reset_index().to_parquet(path)
            os.chmod(path, 0o444)
            manifest["symbols"][sym] = {
                "rows": len(df),
                "first": str(df.index.min().date()),
                "last": str(df.index.max().date()),
                "sha256": sha256(path),
            }
        except Exception as e:  # noqa: BLE001 - bir sembol hatası indirmeyi durdurmaz
            print(f"  indirilemedi: {e!r}")
            manifest["symbols"][sym] = {"error": repr(e)}
        time.sleep(pause)
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"→ {snap}")
    return snap


# --------------------------------------------------------------------------- temizleme
def to_daily(raw: pd.DataFrame, last_session: pd.Timestamp) -> tuple[pd.DataFrame, list]:
    """yfinance tablosu -> tarih indeksli, küçük harf kolonlu tablo. (df, [(severity, code, msg)])"""
    issues = []
    df = raw.rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df.pop("Date")).dt.tz_localize(None).dt.normalize()
    for c in COLUMNS.values():
        if c not in df:
            df[c] = 0.0
    df = df[["date", *COLUMNS.values()]]

    exact = df.duplicated()
    if exact.any():
        issues.append(("BILGI", "TEKRAR_BIREBIR", f"{exact.sum()} birebir tekrar satır tekilleştirildi"))
    df = df[~exact]
    conflict = df["date"].duplicated(keep=False)
    if conflict.any():
        dates = sorted({str(d.date()) for d in df.loc[conflict, "date"]})
        issues.append(("HATA", "TEKRAR_FARKLI", f"aynı tarihte farklı değerler, son satır tutuldu: {dates}"))
        df = df.drop_duplicates("date", keep="last")

    partial = df["date"] > last_session
    if partial.any():
        issues.append(("BILGI", "KISMI_GUN", f"bitmemiş seans çıkarıldı: {df.loc[partial, 'date'].dt.date.tolist()}"))
        df = df[~partial]

    df = df.set_index("date").sort_index()
    # t gününden SONRAKİ split'lerin çarpımı: geri düzeltilmiş fiyat * F = o gün görülen fiyat
    s = df["splits"].replace(0, 1.0).astype(float)
    df["split_factor"] = s[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return df, issues


def verify_split_method(daily: dict[str, pd.DataFrame]) -> dict:
    """Sağlayıcının split yöntemini reference_points.csv'deki BAĞIMSIZ kaynaklı değerlerle belirler.
    gerçek / sağlayıcı ≈ F -> 'back_adjusted', ≈ 1 -> 'raw'. Fiyat ve hacim ayrı.
    Sonuç yalnız ≥3 nokta, ≥2 sembol ve hepsi aynı yöntemi gösteriyorsa kabul edilir; yoksa None."""
    refs = pd.read_csv(REFERENCES, comment="#", parse_dates=["date"])
    result = {}
    for kind, ref_col, col in [("price", "gercek_kapanis", "close"), ("volume", "gercek_hacim", "volume")]:
        points = []
        for r in refs.dropna(subset=[ref_col]).itertuples():
            df = daily.get(r.symbol)
            if df is None or r.date not in df.index:
                continue
            F = df.at[r.date, "split_factor"]
            ratio = getattr(r, ref_col) / df.at[r.date, col]
            adjusted = F if kind == "price" else 1 / F
            if abs(ratio / adjusted - 1) <= REF_TOL[kind]:
                points.append((r.symbol, "back_adjusted"))
            elif abs(ratio - 1) <= REF_TOL[kind]:
                points.append((r.symbol, "raw"))
            else:
                points.append((r.symbol, f"TUTARSIZ ({r.date.date()} oran {ratio:.3f}, F={F:g})"))
        methods = {m for _, m in points}
        ok = (len(points) >= 3 and len({s for s, _ in points}) >= 2
              and methods in ({"back_adjusted"}, {"raw"}))
        result[kind] = methods.pop() if ok else None
        result[f"{kind}_points"] = points
    return result


def add_raw_columns(df: pd.DataFrame, method: dict) -> pd.DataFrame:
    """raw_close / raw_volume: o gün gerçekte görülen fiyat ve hacim.
    Yöntem doğrulanmadıysa son split öncesi NaN kalır (≥5$ filtresi orada uygulanamaz)."""
    F = df["split_factor"]
    for kind, col, raw_col, apply in [
        ("price", "close", "raw_close", lambda x: x * F),
        ("volume", "volume", "raw_volume", lambda x: x / F),
    ]:
        m = method[kind]
        if m == "back_adjusted":
            df[raw_col] = apply(df[col])
        elif m == "raw":
            df[raw_col] = df[col].astype(float)
        else:
            df[raw_col] = df[col].where(F == 1).astype(float)
    return df


# --------------------------------------------------------------------------- kontroller
def check(sym: str, df: pd.DataFrame, last_session: pd.Timestamp) -> list:
    i = info(sym)
    out = []
    px = df[["open", "high", "low", "close"]]
    is_index = i["asset"].startswith("index")

    if (n := px.isna().any(axis=1).sum()):
        out.append(("HATA", "BOS_FIYAT", f"{n} satırda boş OHLC"))
    if (n := (px <= 0).any(axis=1).sum()):
        out.append(("HATA", "NONPOS", f"{n} satırda ≤0 fiyat"))
    if (n := (df["volume"] < 0).sum()):
        out.append(("HATA", "NEG_HACIM", f"{n} satırda negatif hacim"))
    tol = 1e-6 * df["close"].abs()
    oc_hi, oc_lo = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    if (n := ((df["high"] + tol < oc_hi) | (df["low"] - tol > oc_lo)).sum()):
        out.append(("UYARI", "HIGH_LOW", f"{n} satırda High/Low, Open/Close'u kapsamıyor"))
    if not is_index and (n := ((df["volume"] == 0) & df["eligible"]).sum()):
        out.append(("UYARI", "SIFIR_HACIM", f"eligible dönemde {n} gün sıfır hacim"))

    # takvim
    sessions = CAL.sessions_in_range(df.index.min(), df.index.max()).tz_localize(None)
    if len(missing := sessions.difference(df.index)):
        out.append(("UYARI", "EKSIK_SEANS", f"{len(missing)} seans eksik: {[str(d.date()) for d in missing[:5]]}"))
    if len(extra := df.index[~df["is_session"]]):
        out.append(("UYARI", "FAZLA_TARIH", f"{len(extra)} satır seans dışı günde: {[str(d.date()) for d in extra[:5]]}"))
    if df.index.max() < last_session:
        out.append(("HATA", "BAYAT", f"son satır {df.index.max().date()}, beklenen {last_session.date()}"))

    # başlangıç: halka arz/kuruluş tarihi, yoksa faaliyet kesimi, yoksa planın başlangıcı
    first = df.index.min()
    if "expected_first_month" in i:
        if f"{first:%Y-%m}" != i["expected_first_month"]:
            out.append(("UYARI", "ILK_TARIH", f"ilk tarih {first.date()}, beklenen ay {i['expected_first_month']}"))
    else:
        expected = pd.Timestamp(i.get("expected_first_date") or i.get("eligible_from") or UNIVERSE["start"])
        if first > expected + pd.Timedelta(days=7):
            out.append(("HATA", "BASLANGIC_EKSIK", f"veri {first.date()}'de başlıyor, beklenen {expected.date()}"))
        elif "expected_first_date" in i and first < expected:
            out.append(("UYARI", "ILK_TARIH", f"ilk tarih {first.date()} beklenenden ({expected.date()}) önce"))

    # split'ler: universe.json'daki beklenenler ile verideki kayıtlar
    seen = {(d, float(r)) for d, r in df.loc[~df["splits"].isin([0, 1]), "splits"].items()}
    expected = {(pd.Timestamp(d), float(r)) for d, r in i.get("expected_splits", [])}
    for d, r in sorted(expected - seen):
        out.append(("HATA", "SPLIT_EKSIK", f"beklenen split veride yok: {d.date()} oran {r:g}"))
    for d, r in sorted(seen - expected):
        out.append(("BILGI", "SPLIT_YENI", f"universe.json'da olmayan split: {d.date()} oran {r:g}"))

    ret = df.loc[df["eligible"], "close"].pct_change()
    if not is_index and (n := (ret.abs() > BIG_MOVE).sum()):
        out.append(("BILGI", "BUYUK_HAREKET", f"|getiri|>{BIG_MOVE:.0%} olan {n} gün (silinmedi)"))
    if sym == "^TNX":
        out.append(("BILGI", "TNX_BIRIM", f"aralık {df['close'].min():.2f}–{df['close'].max():.2f}: yüzde (4.25 = %4.25)"))
    return [{"symbol": sym, "severity": s, "code": c, "msg": m} for s, c, m in out]


# --------------------------------------------------------------------------- kapsama
def valid_bars(df: pd.DataFrame) -> pd.Series:
    px = df[["open", "high", "low", "close"]]
    ok = np.isfinite(px).all(axis=1) & (px > 0).all(axis=1) & (df["volume"] >= 0)
    tol = 1e-6 * df["close"]
    return ok & (df["high"] + tol >= px.max(axis=1)) & (df["low"] - tol <= px.min(axis=1))


def train_samples(df: pd.DataFrame, train_end: pd.Timestamp, W: int, H: int) -> int:
    """Ön sayım (etiket hesaplanınca kesinleşir). Seans takvimi üzerinde, eligible dönemde:
    t geçerli bar, t'den önce ≥W geçerli bar, t+1..t+H'nin hepsi geçerli ve t+H ≤ training sonu (purge)."""
    e = df[df["eligible"]]
    if e.empty:
        return 0
    sessions = CAL.sessions_in_range(e.index.min(), e.index.max()).tz_localize(None)
    valid = valid_bars(e).reindex(sessions, fill_value=False).to_numpy()
    cs = np.concatenate([[0], np.cumsum(valid)])
    t = np.arange(len(sessions) - H)
    warm = cs[t] >= W
    future_ok = cs[t + H + 1] - cs[t + 1] == H
    purged = sessions[t + H] <= train_end
    return int((valid[t] & warm & future_ok & purged).sum())


def coverage(daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    W, MIN, H = UNIVERSE["warmup_sessions"], UNIVERSE["min_train_samples"], max(UNIVERSE["horizons"])
    rows = []
    for s in UNIVERSE["symbols"]:
        if not TRAIN_ROLES & set(s["roles"]):
            continue
        row = {"symbol": s["symbol"], "roles": ",".join(s["roles"])}
        for f in UNIVERSE["folds"]:
            df = daily.get(s["symbol"])
            row[f["name"]] = train_samples(df, pd.Timestamp(f["train_end"]), W, H) if df is not None else 0
        rows.append(row)
    cov = pd.DataFrame(rows)

    print(f"\nKapsama — H={H} eğitim örneği, ≥{MIN} = YETER (ön sayım)")
    for f in UNIVERSE["folds"]:
        k = f["name"]
        enough = cov[k] >= MIN
        by_role = {r: f"{(enough & cov.roles.str.contains(r)).sum()}/{cov.roles.str.contains(r).sum()}"
                   for r in ["ana20", "kontrol8", "islem10"]}
        ana = cov.roles.str.contains("ana20")
        print(f"  {k} (train ≤{f['train_end']}): {by_role} · örnek ana20 {cov.loc[ana & enough, k].sum()}, "
              f"geniş28 {cov.loc[enough, k].sum()}")
    return cov


# --------------------------------------------------------------------------- build
def build(snap: Path) -> None:
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    last_session = pd.Timestamp(manifest["last_complete_session"])

    bad = [s for s, r in manifest["symbols"].items()
           if "sha256" in r and sha256(snap / "raw" / f"{safe(s)}.parquet") != r["sha256"]]
    if bad:
        raise SystemExit(f"Ham dosya checksum'ı uyuşmuyor, durduruldu: {bad}")

    daily, issues = {}, []
    for sym, rec in manifest["symbols"].items():
        if "error" in rec:
            issues.append({"symbol": sym, "severity": "HATA", "code": "INDIRME", "msg": rec["error"]})
            continue
        df, found = to_daily(pd.read_parquet(snap / "raw" / f"{safe(sym)}.parquet"), last_session)
        issues += [{"symbol": sym, "severity": s, "code": c, "msg": m} for s, c, m in found]
        df["eligible"] = df.index >= pd.Timestamp(info(sym).get("eligible_from", df.index.min()))
        df["is_session"] = df.index.isin(CAL.sessions_in_range(df.index.min(), df.index.max()).tz_localize(None))
        daily[sym] = df

    method = verify_split_method(daily)
    for kind in ["price", "volume"]:
        print(f"Split yöntemi ({kind}): {method[kind] or 'DOĞRULANMADI'} · noktalar {method[f'{kind}_points']}")
        if method[kind] is None:
            issues.append({"symbol": "*", "severity": "UYARI", "code": f"SPLIT_YONTEMI_{kind.upper()}",
                           "msg": "reference_points.csv ile doğrulanmadı; son split öncesi raw değer NaN"})

    (snap / "clean").mkdir(exist_ok=True)
    for sym, df in daily.items():
        df = add_raw_columns(df, method)
        df.to_parquet(snap / "clean" / f"{safe(sym)}.parquet")
        issues += check(sym, df, last_session)

    issues = pd.DataFrame(issues, columns=["symbol", "severity", "code", "msg"])
    issues.to_csv(snap / "issues.csv", index=False)
    coverage(daily).to_csv(snap / "coverage.csv", index=False)

    print(f"\n{snap}")
    print(issues["severity"].value_counts().to_string())
    for r in issues[issues.severity == "HATA"].itertuples():
        print(f"  HATA {r.symbol:<6} {r.code}: {r.msg[:120]}")


def load(snapshot: str | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    """Temiz katmanı (symbol, date) indeksli tek tabloda döndürür."""
    snap = DATA / snapshot if snapshot else latest_snapshot()
    symbols = symbols or [s["symbol"] for s in UNIVERSE["symbols"]]
    frames = {s: pd.read_parquet(p) for s in symbols if (p := snap / "clean" / f"{safe(s)}.parquet").exists()}
    return pd.concat(frames, names=["symbol", "date"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "download", "build"])
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--snapshot")
    ap.add_argument("--pause", type=float, default=1.0, help="semboller arası bekleme (sn)")
    a = ap.parse_args()

    snap = DATA / a.snapshot if a.snapshot else None
    if a.cmd in ("all", "download"):
        snap = download(a.symbols, a.pause)
    if a.cmd in ("all", "build"):
        build(snap or latest_snapshot())
