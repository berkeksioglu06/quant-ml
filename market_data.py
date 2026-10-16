"""Günlük piyasa verisi: indir -> temizle -> kontrol et -> kapsamayı say.

    uv run python market_data.py all                        # indir + build
    uv run python market_data.py download --symbols RKLB SPY
    uv run python market_data.py build [--snapshot yahoo_20261004T171338Z]

Çıktılar (yalnız sabitlenmiş inceleme kaydı Git'e girer):
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
        "auxiliary_files": [],
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
        if "error" not in manifest["symbols"][sym] and info(sym)["asset"] == "equity":
            try:
                import yfinance as yf

                earnings = yf.Ticker(sym).get_earnings_dates(limit=100)
                if earnings is not None and not earnings.empty:
                    ep = snap / "raw" / "earnings" / f"{safe(sym)}.parquet"
                    ep.parent.mkdir(exist_ok=True)
                    earnings.reset_index().to_parquet(ep)
                    os.chmod(ep, 0o444)
                    manifest["auxiliary_files"].append(
                        {"file": ep.relative_to(snap).as_posix(), "sha256": sha256(ep)}
                    )
                    manifest["symbols"][sym]["earnings_status"] = "OK"
                else:
                    manifest["symbols"][sym]["earnings_status"] = "BOS"
            except Exception as exc:  # noqa: BLE001 - fiyatlar korunur, eksik takvim raporlanır
                manifest["symbols"][sym]["earnings_status"] = "HATA"
                manifest["symbols"][sym]["earnings_error"] = repr(exc)
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
            df[c] = 0.0 if c in {"dividends", "splits", "capital_gains"} else np.nan
    df = df[["date", *COLUMNS.values()]]

    exact = df.duplicated()
    if exact.any():
        issues.append(
            ("BILGI", "TEKRAR_BIREBIR", f"{exact.sum()} birebir tekrar satır tekilleştirildi")
        )
    df = df[~exact]
    conflict = df["date"].duplicated(keep=False)
    if conflict.any():
        dates = sorted({str(d.date()) for d in df.loc[conflict, "date"]})
        issues.append(
            ("HATA", "TEKRAR_FARKLI", f"aynı tarihte farklı değerler, son satır tutuldu: {dates}")
        )
        df = df.drop_duplicates("date", keep="last")

    partial = df["date"] > last_session
    if partial.any():
        issues.append(
            (
                "BILGI",
                "KISMI_GUN",
                f"bitmemiş seans çıkarıldı: {df.loc[partial, 'date'].dt.date.tolist()}",
            )
        )
        df = df[~partial]

    df = df.set_index("date").sort_index()
    if df.empty:
        return df.assign(split_factor=pd.Series(dtype=float)), issues
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
    for kind, ref_col, col in [
        ("price", "gercek_kapanis", "close"),
        ("volume", "gercek_hacim", "volume"),
    ]:
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
        ok = (
            len(points) >= 3
            and len({s for s, _ in points}) >= 2
            and methods in ({"back_adjusted"}, {"raw"})
        )
        result[kind] = methods.pop() if ok else None
        result[f"{kind}_points"] = points
    return result


def add_raw_columns(
    df: pd.DataFrame, method: dict, symbol_info: dict | None = None
) -> pd.DataFrame:
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
    if (symbol_info or {}).get("historical_unverified_before"):
        before = df.index < pd.Timestamp(symbol_info["historical_unverified_before"])
        df.loc[before, ["raw_close", "raw_volume"]] = np.nan
    return df


# --------------------------------------------------------------------------- kontroller
# Ayrıntılı kalite kontrolleri; veri değiştirilmez.
DIV_TOL = 0.002
SPLIT_WINDOW = 20
NY_TZ = "America/New_York"


def us_sessions(start, end):
    return CAL.sessions_in_range(start, end).tz_localize(None), "XNYS"


def _issue(lst, sev, code, msg):
    lst.append({"severity": sev, "code": code, "msg": msg})


def check_ohlcv(df: pd.DataFrame, asset: str, issues: list) -> dict:
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    r = {}
    r["nan_ohlc"] = int((~np.isfinite(df[["open", "high", "low", "close"]])).any(axis=1).sum())
    r["nan_volume"] = int((~np.isfinite(v)).sum())
    r["nonpos_price"] = int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
    r["neg_volume"] = int((v < 0).sum())
    is_traded = asset in ("equity", "etf")
    r["zero_volume"] = int((v == 0).sum()) if is_traded else None
    tol = 1e-6 * c.abs()
    bad_hl = (h + tol < np.maximum(o, c)) | (l - tol > np.minimum(o, c)) | (h + tol < l)
    r["high_low_ihlal"] = int(bad_hl.sum())
    r["high_low_ihlal_tarih"] = [str(d.date()) for d in df.index[bad_hl][:10]]
    same = df[["open", "high", "low", "close", "volume"]].diff().abs().sum(axis=1) == 0
    same.iloc[0] = False
    r["donuk_satir"] = int(same.sum())  # önceki günle birebir aynı OHLCV
    r["ohlc_esit_satir"] = int(((o == h) & (h == l) & (l == c)).sum())

    if r["nan_ohlc"]:
        _issue(issues, "HATA", "NAN_OHLC", f"{r['nan_ohlc']} satırda boş OHLC")
    if r["nan_volume"]:
        if is_traded:
            _issue(
                issues,
                "HATA",
                "NAN_VOLUME",
                f"{r['nan_volume']} satırda boş hacim: "
                f"{[str(d.date()) for d in df.index[v.isna()][:5]]}",
            )
        else:
            _issue(issues, "BILGI", "NAN_VOLUME", f"{r['nan_volume']} satırda boş hacim (endeks)")
    if r["nonpos_price"]:
        _issue(issues, "HATA", "NONPOS", f"{r['nonpos_price']} satırda ≤0 fiyat")
    if r["neg_volume"]:
        _issue(issues, "HATA", "NEG_VOL", f"{r['neg_volume']} satırda negatif hacim")
    if r["high_low_ihlal"]:
        sev = "HATA" if r["high_low_ihlal"] > 5 else "UYARI"
        _issue(
            issues,
            sev,
            "HIGH_LOW",
            f"{r['high_low_ihlal']} satırda High/Low ilişkisi bozuk: {r['high_low_ihlal_tarih'][:5]}",
        )
    if r["donuk_satir"] and is_traded:
        _issue(
            issues,
            "UYARI",
            "DONUK",
            f"{r['donuk_satir']} satır önceki günle birebir aynı (donmuş veri?)",
        )
    if is_traded and r["zero_volume"]:
        _issue(issues, "UYARI", "ZERO_VOL", f"{r['zero_volume']} işlem gününde hacim 0")
    return r


def check_calendar(df: pd.DataFrame, asset: str, issues: list, exp_last, ref_dates=None) -> dict:
    r = {
        "takvim": None,
        "eksik_seans": 0,
        "eksik_tarihler": [],
        "fazla_tarih": 0,
        "fazla_tarihler": [],
    }
    if asset == "crypto" or df.empty:
        if asset == "crypto" and len(df):
            full = pd.date_range(df.index.min(), df.index.max(), freq="D")
            miss = full.difference(df.index)
            r.update(
                takvim="7/24 günlük",
                eksik_seans=len(miss),
                eksik_tarihler=[str(d.date()) for d in miss[:30]],
            )
            if len(miss):
                _issue(issues, "UYARI", "EKSIK_GUN", f"{len(miss)} takvim günü eksik (kripto)")
        return r
    sess, src = us_sessions(df.index.min(), df.index.max())
    if not len(sess) and ref_dates is not None:
        sess = ref_dates[(ref_dates >= df.index.min()) & (ref_dates <= df.index.max())]
        src = "SPY tarihleri (exchange_calendars yok)"
    r["takvim"] = src
    miss = sess.difference(df.index)
    extra = df.index.difference(sess)
    r.update(
        eksik_seans=len(miss),
        eksik_tarihler=[str(d.date()) for d in miss[:40]],
        fazla_tarih=len(extra),
        fazla_tarihler=[str(d.date()) for d in extra[:20]],
    )
    # en uzun boşluk (seans cinsinden)
    if len(sess) > 1:
        pos = sess.get_indexer(df.index[df.index.isin(sess)])
        gaps = np.diff(pos) - 1 if len(pos) > 1 else np.array([0])
        r["en_uzun_bosluk_seans"] = int(gaps.max()) if len(gaps) else 0
        if len(gaps) and gaps.max() > 0:
            j = int(gaps.argmax())
            r["en_uzun_bosluk_baslangic"] = str(sess[pos[j] + 1].date())
    if len(miss):
        sev = "BILGI" if asset.startswith("index") else "UYARI"
        if r.get("en_uzun_bosluk_seans", 0) >= 20:
            sev = "HATA" if asset in ("equity", "etf") else sev
        _issue(
            issues,
            sev,
            "EKSIK_SEANS",
            f"{len(miss)} ABD seansı eksik (en uzun boşluk {r.get('en_uzun_bosluk_seans')} seans, "
            f"{r.get('en_uzun_bosluk_baslangic')}): {r['eksik_tarihler'][:6]}",
        )
    if len(extra):
        _issue(
            issues,
            "UYARI",
            "FAZLA_TARIH",
            f"{len(extra)} satır ABD seansı olmayan günde: {r['fazla_tarihler'][:6]}",
        )
    r["son_tarih"] = str(df.index.max().date())
    if exp_last is not None:
        r["beklenen_son_seans"] = str(exp_last.date())
        if df.index.max() < exp_last:
            _issue(
                issues,
                "UYARI",
                "BAYAT",
                f"son tarih {r['son_tarih']} < beklenen {r['beklenen_son_seans']}",
            )
    return r


def check_dividends(df: pd.DataFrame, issues: list) -> dict:
    r = {}
    if "adj_close" not in df.columns or df["adj_close"].isna().all():
        _issue(issues, "HATA", "ADJ_YOK", "Adj Close kolonu yok/boş")
        return r
    invalid = ~np.isfinite(df["adj_close"]) | (df["adj_close"] <= 0)
    if invalid.any():
        _issue(issues, "HATA", "ADJ_GECERSIZ", f"{int(invalid.sum())} satırda geçersiz Adj Close")
    ratio = df["adj_close"] / df["close"]
    payout = df["dividends"].fillna(0) + df.get(
        "capital_gains", pd.Series(0.0, index=df.index)
    ).fillna(0)
    ev_idx = np.where(payout.values > 0)[0]
    errs = []
    for i in ev_idx:
        if i == 0:
            continue
        implied = ratio.iloc[i - 1] / ratio.iloc[i]
        expected = 1 - payout.iloc[i] / df["close"].iloc[i - 1]
        errs.append(
            {
                "date": str(df.index[i].date()),
                "payout": float(payout.iloc[i]),
                "implied": float(implied),
                "expected": float(expected),
                "err": float(abs(implied - expected)),
            }
        )
    r["temettu_olay"] = len(ev_idx)
    r["temettu_toplam"] = float(df["dividends"].sum())
    bad = [e for e in errs if e["err"] > DIV_TOL]
    r["temettu_faktor_max_hata"] = max((e["err"] for e in errs), default=0.0)
    r["temettu_faktor_uyumsuz"] = bad[:10]
    # temettü olmayan günlerde Adj faktör değişimi
    step = (ratio / ratio.shift(1) - 1).abs()
    step.iloc[0] = 0
    unexplained = step[(payout == 0) & (step > 1e-4)]
    r["adj_aciklanamayan_degisim"] = len(unexplained)
    r["adj_aciklanamayan_tarih"] = [str(d.date()) for d in unexplained.index[:10]]
    r["adj_close_ratio_son"] = float(ratio.iloc[-1])
    r["adj_gt_close"] = int((ratio > 1 + 1e-6).sum())
    if bad:
        _issue(
            issues,
            "UYARI",
            "TEMETTU_FAKTOR",
            f"{len(bad)} temettü olayında Adj faktörü beklenenle uyuşmuyor (max hata {r['temettu_faktor_max_hata']:.4f})",
        )
    if r["adj_aciklanamayan_degisim"]:
        _issue(
            issues,
            "UYARI",
            "ADJ_DEGISIM",
            f"{r['adj_aciklanamayan_degisim']} günde temettü yokken Adj/Close oranı değişiyor: {r['adj_aciklanamayan_tarih'][:5]}",
        )
    if abs(r["adj_close_ratio_son"] - 1) > 1e-6:
        _issue(
            issues,
            "UYARI",
            "ADJ_SON",
            f"son satırda Adj/Close = {r['adj_close_ratio_son']:.6f} (1 olmalı)",
        )
    if r["adj_gt_close"]:
        _issue(issues, "UYARI", "ADJ_GT", f"{r['adj_gt_close']} satırda Adj Close > Close")
    return r


def check_moves(df: pd.DataFrame, asset: str, issues: list) -> dict:
    c = df["close"]
    ret = c.pct_change()
    gap = df["open"] / c.shift(1) - 1
    big = ret[ret.abs() > BIG_MOVE]
    rows = []
    for d, x in big.items():
        i = df.index.get_loc(d)
        rows.append(
            {
                "date": str(d.date()),
                "ret": round(float(x), 4),
                "gap": round(float(gap.iloc[i]), 4),
                "split": float(df["splits"].iloc[i]),
                "div": float(df["dividends"].iloc[i]),
                "vol_x": round(
                    float(df["volume"].iloc[i] / df["volume"].iloc[max(0, i - 20) : i].median()), 2
                )
                if asset in ("equity", "etf")
                and i > 0
                and df["volume"].iloc[max(0, i - 20) : i].median() > 0
                else None,
            }
        )
    # gizli split şüphesi: ~1/n veya ~n oranlı hareket, split kaydı yok, ertesi gün geri dönmüyor
    suspects = []
    for rr in rows:
        x = 1 + rr["ret"]
        for n in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50):
            if (abs(x - 1 / n) < 0.03 / n or abs(x - n) < 0.03 * n) and rr["split"] in (0, 1):
                suspects.append({**rr, "olasi_oran": n})
    if suspects:
        _issue(
            issues,
            "UYARI",
            "GIZLI_SPLIT",
            f"split kaydı olmadan split oranına benzer hareket (gerçek çöküş de olabilir, incele): {suspects[:3]}",
        )
    if rows and asset in ("equity", "etf"):
        _issue(
            issues,
            "BILGI",
            "BUYUK_HAREKET",
            f"|getiri|>{BIG_MOVE:.0%} olan {len(rows)} gün (silinmedi)",
        )
    return {"buyuk_hareket": rows, "gizli_split_suphesi": suspects}


def expected_first_session(
    info: dict, req_start: pd.Timestamp
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """Beklenen ilk seans ve kabul edilen en geç ilk tarih.
    Öncelik: halka arz / kuruluş tarihi > yaklaşık kuruluş ayı > faaliyet kesimi > istenen başlangıç."""
    if info.get("expected_first_date"):
        base, why, slack = (
            pd.Timestamp(info["expected_first_date"]),
            "halka arz / kuruluş",
            pd.Timedelta(days=5),
        )
    elif info.get("expected_first_month"):
        base, why = pd.Timestamp(info["expected_first_month"] + "-01"), "yaklaşık kuruluş ayı"
        slack = pd.offsets.MonthEnd(0).rollforward(base) - base + pd.Timedelta(days=5)
    elif info.get("eligible_from"):
        base, why, slack = (
            pd.Timestamp(info["eligible_from"]),
            "faaliyet / kesim başlangıcı (veri en geç burada başlamalı)",
            pd.Timedelta(days=5),
        )
    else:
        base, why, slack = req_start, "istenen başlangıç", pd.Timedelta(days=5)
    base = max(base, req_start)
    sess, _ = us_sessions(base, base + pd.Timedelta(days=40))
    first = sess[0] if len(sess) else base
    return first, first + slack, why


def check_history(
    df: pd.DataFrame, info: dict, meta: dict, issues: list, req_start: pd.Timestamp
) -> dict:
    r = {"ilk_tarih": str(df.index.min().date()), "son_tarih": str(df.index.max().date())}
    first = df.index.min()
    if info.get("asset") != "crypto":
        exp, latest_ok, why = expected_first_session(info, req_start)
        r["beklenen_ilk_seans"] = str(exp.date())
        r["beklenen_kaynak"] = why
        r["ilk_tarih_fark_gun"] = int((first - exp).days)
        if first > latest_ok:
            miss, _ = us_sessions(exp, first - pd.Timedelta(days=1))
            r["baslangicta_eksik_seans"] = len(miss)
            _issue(
                issues,
                "HATA",
                "BASLANGIC_EKSIK",
                f"veri {r['ilk_tarih']}'de başlıyor; beklenen ilk seans {exp.date()} ({why}). "
                f"Başta {len(miss)} seans eksik",
            )
        elif info.get("expected_first_date") and first < pd.Timestamp(
            info["expected_first_date"]
        ) - pd.Timedelta(days=5):
            _issue(
                issues,
                "UYARI",
                "ILK_TARIH_ERKEN",
                f"veri {r['ilk_tarih']}'de başlıyor, halka arz/kuruluş {info['expected_first_date']}; öncesi incelenmeli",
            )
    if info.get("expected_first_date"):
        r["beklenen_ilk_tarih"] = info["expected_first_date"]
    ftd = (meta or {}).get("firstTradeDate_iso")
    if ftd:
        r["yahoo_firstTradeDate"] = ftd[:10]
    ef = info.get("eligible_from")
    if ef:
        ef = pd.Timestamp(ef)
        pre, post = df[df.index < ef], df[df.index >= ef]
        r["kesim_tarihi"] = str(ef.date())
        r["kesim_oncesi_satir"] = len(pre)
        r["kesim_sonrasi_satir"] = len(post)

        def seg(x):
            if len(x) < 5:
                return None
            ret = x["close"].pct_change().dropna()
            return {
                "satir": len(x),
                "ilk": str(x.index.min().date()),
                "son": str(x.index.max().date()),
                "medyan_kapanis": round(float(x["raw_close"].median()), 2)
                if x["raw_close"].notna().any()
                else None,
                "medyan_abs_getiri": round(float(ret.abs().median()), 4),
                "9_11.5_bandi": round(float(x["raw_close"].between(9, 11.5).mean()), 3),
                "medyan_dolar_hacim": round(float((x["close"] * x["volume"]).median()), 0),
            }

        r["kesim_oncesi"] = seg(pre)
        r["kesim_sonrasi_ilk60"] = seg(post.iloc[:60])
        if len(post) and abs((post.index.min() - ef).days) > 4:
            _issue(
                issues,
                "UYARI",
                "KESIM_TARIHI",
                f"kesim {ef.date()} sonrası ilk satır {post.index.min().date()}",
            )
        if len(pre):
            spac_like = r["kesim_oncesi"] and r["kesim_oncesi"]["9_11.5_bandi"] > 0.5
            _issue(
                issues,
                "BILGI",
                "KESIM_ONCESI",
                f"{len(pre)} satır kesim ({ef.date()}) öncesi; eligible=False işaretlendi"
                + (" — SPAC imzası (fiyat ~$10 bandı)" if spac_like else ""),
            )
    return r


def check_index_scale(df: pd.DataFrame, sym: str, info: dict, issues: list) -> dict:
    r = {}
    c = df["close"]
    r["min"], r["max"] = float(c.min()), float(c.max())
    r["medyan_son252"] = float(c.iloc[-252:].median())
    lr = np.log(c / c.shift(1)).abs()
    jumps = lr[lr > np.log(5)]
    r["olcek_sicramasi"] = [str(d.date()) for d in jumps.index]
    if info["asset"] == "index_rate":
        r["olcek_yorumu"] = (
            "yüzde (ör. 4.25 = %4.25)" if r["max"] < 20 else "x10 olası (ör. 42.5 = %4.25)"
        )
        if jumps.size:
            _issue(
                issues,
                "HATA",
                "TNX_OLCEK",
                f"^TNX'te x10 ölçek kırılması olası tarihler: {r['olcek_sicramasi'][:5]}",
            )
        if r["min"] <= 0:
            _issue(issues, "UYARI", "TNX_NEG", "≤0 getiri değeri (2020 dip dönemi kontrol)")
        _issue(
            issues,
            "BILGI",
            "TNX_BIRIM",
            f"^TNX aralık {r['min']:.2f}–{r['max']:.2f}: {r['olcek_yorumu']}",
        )
    if info["asset"] == "index_vol" and (r["min"] < 5 or r["max"] > 100):
        _issue(
            issues,
            "UYARI",
            "VIX_ARALIK",
            f"^VIX aralığı olağandışı {r['min']:.1f}–{r['max']:.1f}",
        )
    return r


def liquidity_stats(df: pd.DataFrame, info: dict) -> dict:
    x = df[df["eligible"]] if "eligible" in df else df
    if len(x) < 30 or info["asset"] not in ("equity", "etf"):
        return {}
    last = x.iloc[-252:]
    lr = np.log(last["adj_close"] / last["adj_close"].shift(1)).dropna()
    return {
        "medyan_dolar_hacim_1y_musd": round(
            float((last["close"] * last["volume"]).median() / 1e6), 1
        ),
        "yillik_vol_1y": round(float(lr.std() * np.sqrt(252)), 3),
        "gercek_fiyat_dogrulanamayan_orani": round(float(x["raw_close"].isna().mean()), 3),
        "gercek_fiyat_lt5_orani_dogrulanan_icinde": round(
            float((x["raw_close"] < 5).sum() / max(x["raw_close"].notna().sum(), 1)), 3
        ),
        "gercek_fiyat_lt5_son_tarih": str(x.index[x["raw_close"] < 5].max().date())
        if (x["raw_close"] < 5).any()
        else None,
    }


def _flag(observed_log_ratio: float, s: float) -> str:
    """Gözlenen log oranının hangi hipoteze yakın olduğu: yalnız İNCELEME İŞARETİ, doğrulama değildir."""
    d_adj = abs(observed_log_ratio)
    d_raw = abs(observed_log_ratio - np.log(1 / s))
    margin = 0.25 * abs(np.log(s))
    if d_adj + margin < d_raw:
        return "duzeltilmis_gibi"
    if d_raw + margin < d_adj:
        return "ham_gibi"
    return "belirsiz"


def split_flags(df: pd.DataFrame) -> list[dict]:
    """Her split için fiyat ve hacim oranlarından İNCELEME İŞARETİ üretir. Bu işaretler raw_* üretmez.
    Fiyat: split günü açılış ve kapanışın önceki kapanışa oranı (≈1 düzeltilmiş gibi, ≈1/s ham gibi).
    Hacim: split sonrası/öncesi 20 seans medyan hacim oranı (≈1 düzeltilmiş gibi, ≈s ham gibi).
    Hacim oranı gerçek işlem davranışına da bağlı olduğundan özellikle zayıf bir işarettir."""
    out = []
    ev = df.index[(df["splits"] != 0) & (df["splits"] != 1)]
    for d in ev:
        i = df.index.get_loc(d)
        s = float(df["splits"].iloc[i])
        rec = {"date": str(d.date()), "ratio": s}
        if i == 0:
            rec.update(fiyat_isaret="belirsiz", hacim_isaret="belirsiz")
            out.append(rec)
            continue
        pc = df["close"].iloc[i - 1]
        lc = np.log(df["close"].iloc[i] / pc)
        lo = np.log(df["open"].iloc[i] / pc)
        f_c, f_o = _flag(lc, s), _flag(lo, s)
        rec["close_ratio"] = round(float(np.exp(lc)), 4)
        rec["open_ratio"] = round(float(np.exp(lo)), 4)
        rec["fiyat_isaret"] = f_c if f_c == f_o else "belirsiz"
        pre = df["volume"].iloc[max(0, i - SPLIT_WINDOW) : i]
        post = df["volume"].iloc[i + 1 : i + 1 + SPLIT_WINDOW]
        pre, post = pre[pre > 0].dropna(), post[post > 0].dropna()
        if len(pre) >= 5 and len(post) >= 5:
            lv = np.log(post.median() / pre.median())
            rec["volume_ratio"] = round(float(np.exp(lv)), 3)
            rec["hacim_isaret"] = _flag(-lv, s)
        else:
            rec["hacim_isaret"] = "belirsiz"
        out.append(rec)
    return out


def check_earnings(
    snap: Path, sym: str, price_first: pd.Timestamp, asof: pd.Timestamp, issues: list
) -> dict:
    """Yahoo bilanço kayıtları = SAĞLAYICI TAKVİM KAYDI. Gerçekleşme bu kaynakla doğrulanmaz.
    Sınıflar: planlanan (tarih > indirme anı) / sonuç_kaydı_var (geçmiş tarih + Reported EPS dolu) /
    durumu_belirsiz (geçmiş tarih + Reported EPS boş)."""
    p = snap / "raw" / "earnings" / f"{safe(sym)}.parquet"
    if not p.exists():
        _issue(issues, "BILGI", "BILANCO_YOK", "Yahoo bilanço takvim kaydı alınamadı")
        return {"durum": "yok"}
    e = pd.read_parquet(p)
    d = pd.to_datetime(e[e.columns[0]])
    d = d.dt.tz_convert(NY_TZ).dt.tz_localize(None) if d.dt.tz is not None else d
    rep_col = next((c for c in e.columns if str(c).lower().startswith("reported")), None)
    reported = e[rep_col].notna().values if rep_col else np.zeros(len(e), dtype=bool)
    past = (d <= asof).values
    cls = np.where(~past, "planlanan", np.where(reported, "sonuc_kaydi_var", "durumu_belirsiz"))
    years = max((asof - max(price_first, pd.Timestamp("2016-01-01"))).days / 365.25, 0.25)
    r = {
        "durum": "var",
        "kayit": len(e),
        "en_eski": str(d.min().date()),
        "en_yeni": str(d.max().date()),
        "planlanan": int((cls == "planlanan").sum()),
        "sonuc_kaydi_var": int((cls == "sonuc_kaydi_var").sum()),
        "durumu_belirsiz": int((cls == "durumu_belirsiz").sum()),
        "reported_kolonu": rep_col,
        "beklenen_ceyrek_yaklasik": round(years * 4),
        "saat_bilgisi_var": bool((d.dt.hour != 0).any()),
        "not": "Sağlayıcı takvim kaydıdır. Gerçekleşme, ilan saati ve önceden duyurulma zamanı bu kaynakla doğrulanmaz.",
    }
    r["sonuc_kaydi_kapsama"] = round(
        min(r["sonuc_kaydi_var"] / max(r["beklenen_ceyrek_yaklasik"], 1), 1.0), 2
    )
    if r["durumu_belirsiz"]:
        _issue(
            issues,
            "UYARI",
            "BILANCO_BELIRSIZ",
            f"{r['durumu_belirsiz']} geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı",
        )
    if r["sonuc_kaydi_kapsama"] < 0.8:
        _issue(
            issues,
            "BILGI",
            "BILANCO_KISMI",
            f"sonuç kaydı olan {r['sonuc_kaydi_var']} kayıt / ~{r['beklenen_ceyrek_yaklasik']} çeyrek (en eski {r['en_eski']})",
        )
    return r


def quality_checks(
    sym: str,
    df: pd.DataFrame,
    last_session: pd.Timestamp,
    *,
    snap: Path | None = None,
    manifest: dict | None = None,
) -> dict:
    i = info(sym)
    issues = []
    result = {"symbol": sym, "issues": issues}
    if df.empty:
        _issue(issues, "HATA", "BOS_TABLO", "Tamamlanmış seans verisi yok")
        result["durum"] = "HATA"
        return result
    result["ohlcv"] = check_ohlcv(df, i["asset"], issues)
    result["takvim"] = check_calendar(df, i["asset"], issues, last_session)
    # Main'in güncellik kuralı korunur: eksik son seans eğitim için hatadır.
    for issue in issues:
        if issue["code"] == "BAYAT":
            issue["severity"] = "HATA"
    requested = pd.Timestamp((manifest or {}).get("start", UNIVERSE["start"]))
    result["gecmis"] = check_history(df, i, {}, issues, requested)
    result["temettu"] = check_dividends(df, issues)
    result["hareket"] = check_moves(df, i["asset"], issues)
    result["likidite"] = liquidity_stats(df, i)
    if i["asset"].startswith("index"):
        result["endeks"] = check_index_scale(df, sym, i, issues)
    flags = split_flags(df)
    result["split_olaylari"] = flags
    for event in flags:
        if event["fiyat_isaret"] == "ham_gibi":
            _issue(
                issues,
                "HATA",
                "SPLIT_FIYAT_KIRILMA",
                f"{event['date']} split günü fiyat kırılması şüphesi; incelenmeli",
            )
        elif event["fiyat_isaret"] == "belirsiz":
            _issue(issues, "UYARI", "SPLIT_FIYAT_ISARET", f"{event['date']} fiyat oranı belirsiz")
        if event["hacim_isaret"] != "duzeltilmis_gibi":
            _issue(
                issues,
                "BILGI",
                "SPLIT_HACIM_ISARET",
                f"{event['date']} hacim oranı {event.get('volume_ratio')}; yalnız inceleme işareti",
            )
    seen = {(pd.Timestamp(e["date"]), e["ratio"]) for e in flags}
    expected = {
        (pd.Timestamp(d), float(r))
        for d, r in i.get("expected_splits", [])
        if requested <= pd.Timestamp(d) <= last_session
    }
    for d, ratio in sorted(expected):
        if not any(abs((fd - d).days) <= 4 and abs(fr - ratio) < 1e-6 for fd, fr in seen):
            _issue(issues, "HATA", "SPLIT_EKSIK", f"Beklenen split yok: {d.date()} oran {ratio:g}")
    for d, ratio in sorted(seen):
        if not any(abs((d - ed).days) <= 4 and abs(ratio - er) < 1e-6 for ed, er in expected):
            _issue(
                issues,
                "BILGI",
                "SPLIT_YENI",
                f"Manifestoda olmayan split: {d.date()} oran {ratio:g}",
            )
    before = i.get("historical_unverified_before")
    if before and (df.index < pd.Timestamp(before)).any():
        _issue(
            issues,
            "UYARI",
            "KURUMSAL_ISLEM_DOGRULANMADI",
            f"{before} öncesi tarihsel fiyat/hacim doğrulanmadı: {i.get('historical_unverified_reason', '')}",
        )
    if snap is not None and i["asset"] == "equity":
        asof = pd.Timestamp((manifest or {})["downloaded_at"])
        asof = asof.tz_convert(NY_TZ).tz_localize(None) if asof.tzinfo else asof
        result["bilanco"] = check_earnings(snap, sym, df.index.min(), asof, issues)
    severity = {e["severity"] for e in issues}
    result["durum"] = "HATA" if "HATA" in severity else "UYARI" if "UYARI" in severity else "OK"
    return result


def check(sym: str, df: pd.DataFrame, last_session: pd.Timestamp) -> list:
    return [{"symbol": sym, **e} for e in quality_checks(sym, df, last_session)["issues"]]


# --------------------------------------------------------------------------- kapsama
def valid_bars(df: pd.DataFrame) -> pd.Series:
    px = df[["open", "high", "low", "close"]]
    ok = (
        np.isfinite(px).all(axis=1)
        & (px > 0).all(axis=1)
        & np.isfinite(df["volume"])
        & (df["volume"] >= 0)
    )
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
    W, MIN, H = (
        UNIVERSE["warmup_sessions"],
        UNIVERSE["min_train_samples"],
        max(UNIVERSE["horizons"]),
    )
    rows = []
    for s in UNIVERSE["symbols"]:
        if not TRAIN_ROLES & set(s["roles"]):
            continue
        row = {"symbol": s["symbol"], "roles": ",".join(s["roles"])}
        for f in UNIVERSE["folds"]:
            df = daily.get(s["symbol"])
            row[f["name"]] = (
                train_samples(df, pd.Timestamp(f["train_end"]), W, H) if df is not None else 0
            )
        rows.append(row)
    cov = pd.DataFrame(rows)

    print(f"\nKapsama — H={H} eğitim örneği, ≥{MIN} = YETER (ön sayım)")
    for f in UNIVERSE["folds"]:
        k = f["name"]
        enough = cov[k] >= MIN
        by_role = {
            r: f"{(enough & cov.roles.str.contains(r)).sum()}/{cov.roles.str.contains(r).sum()}"
            for r in ["ana20", "kontrol8", "islem10"]
        }
        ana = cov.roles.str.contains("ana20")
        print(
            f"  {k} (train ≤{f['train_end']}): {by_role} · örnek ana20 {cov.loc[ana & enough, k].sum()}, "
            f"geniş28 {cov.loc[enough, k].sum()}"
        )
    return cov


# --------------------------------------------------------------------------- build
def build(snap: Path) -> None:
    # İncelenmiş veri kaydı yerinde yeniden üretilmez.
    if (snap / "provenance").exists():
        raise ValueError("Arşiv veri kaydı değiştirilemez; yeni snapshot oluşturun")
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    last_session = pd.Timestamp(manifest["last_complete_session"])

    for record in manifest.get("auxiliary_files", []):
        file = (snap / record["file"]).resolve()
        if (
            not file.is_relative_to(snap.resolve())
            or not file.is_file()
            or sha256(file) != record.get("sha256")
        ):
            raise ValueError("Yardımcı dosya checksum veya yol hatası")
    bad = [
        s
        for s, r in manifest["symbols"].items()
        if "sha256" in r and sha256(snap / "raw" / f"{safe(s)}.parquet") != r["sha256"]
    ]
    if bad:
        raise SystemExit(f"Ham dosya checksum'ı uyuşmuyor, durduruldu: {bad}")

    daily, issues = {}, []
    for sym, rec in manifest["symbols"].items():
        if "error" in rec:
            issues.append(
                {"symbol": sym, "severity": "HATA", "code": "INDIRME", "msg": rec["error"]}
            )
            continue
        df, found = to_daily(pd.read_parquet(snap / "raw" / f"{safe(sym)}.parquet"), last_session)
        issues += [{"symbol": sym, "severity": s, "code": c, "msg": m} for s, c, m in found]
        if df.empty:
            issues.append(
                {
                    "symbol": sym,
                    "severity": "HATA",
                    "code": "BOS_TABLO",
                    "msg": "Tamamlanmış seans verisi yok",
                }
            )
            continue
        df["eligible"] = df.index >= pd.Timestamp(info(sym).get("eligible_from", df.index.min()))
        df["is_session"] = df.index.isin(
            CAL.sessions_in_range(df.index.min(), df.index.max()).tz_localize(None)
        )
        daily[sym] = df

    method = verify_split_method(daily)
    for kind in ["price", "volume"]:
        print(
            f"Split yöntemi ({kind}): {method[kind] or 'DOĞRULANMADI'} · noktalar {method[f'{kind}_points']}"
        )
        if method[kind] is None:
            issues.append(
                {
                    "symbol": "*",
                    "severity": "UYARI",
                    "code": f"SPLIT_YONTEMI_{kind.upper()}",
                    "msg": "reference_points.csv ile doğrulanmadı; son split öncesi raw değer NaN",
                }
            )

    (snap / "clean").mkdir(exist_ok=True)
    results = {}
    for sym, df in daily.items():
        df = add_raw_columns(df, method, info(sym))
        df.to_parquet(snap / "clean" / f"{safe(sym)}.parquet")
        result = quality_checks(sym, df, last_session, snap=snap, manifest=manifest)
        results[sym] = result
        issues += [{"symbol": sym, **e} for e in result["issues"]]

    issues = pd.DataFrame(issues, columns=["symbol", "severity", "code", "msg"])
    issues.to_csv(snap / "issues.csv", index=False)
    coverage(daily).to_csv(snap / "coverage.csv", index=False)
    report = {
        "snapshot": snap.name,
        "last_complete_session": str(last_session.date()),
        "split_method": method,
        "issues": issues.to_dict("records"),
        "results": results,
    }
    (snap / "quality_detail.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Veri kalite raporu",
        "",
        f"Snapshot: {snap.name}",
        f"Son tamamlanmış seans: {last_session.date()}",
        "",
        "| Sembol | Ciddiyet | Kod | Açıklama |",
        "|---|---|---|---|",
    ]
    for row in issues.itertuples():
        message = row.msg.replace("|", "/").replace("\n", " ")
        lines.append(f"| {row.symbol} | {row.severity} | {row.code} | {message} |")
    (snap / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{snap}")
    print(issues["severity"].value_counts().to_string())
    for r in issues[issues.severity == "HATA"].itertuples():
        print(f"  HATA {r.symbol:<6} {r.code}: {r.msg[:120]}")


def load(snapshot: str | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    """Temiz katmanı (symbol, date) indeksli tek tabloda döndürür."""
    snap = DATA / snapshot if snapshot else latest_snapshot()
    symbols = symbols or [s["symbol"] for s in UNIVERSE["symbols"]]
    frames = {
        s: pd.read_parquet(p)
        for s in symbols
        if (p := snap / "clean" / f"{safe(s)}.parquet").exists()
    }
    return pd.concat(frames, names=["symbol", "date"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
