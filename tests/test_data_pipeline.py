"""Uçtan uca test: yfinance çıktısını taklit eden sentetik veri + bilerek enjekte edilen sorunlar.
Ağ gerekmez.  Çalıştırma:  uv run pytest tests/test_data_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os
import stat

from quant_ml.data.clean import (
    ChecksumError,
    add_historical,
    build_clean,
    drop_partial_last,
    evaluate_method,
    load_clean,
    split_flags,
    to_daily,
)
from quant_ml.data.common import (
    Paths,
    is_session_complete,
    load_universe,
    us_sessions,
)
from quant_ml.data.coverage import coverage
from quant_ml.data.download import download_snapshot
from quant_ml.data.quality import run_quality
from quant_ml.data.report import write_report

# İndirme 2 Ekim 2026 NY 16:05'te başlar; her saat okumasında 20 sn ilerler. Erken semboller kapanış+15dk
# eşiğinden önce (kısmi), geç semboller sonra (tamam) indirilir -> sembol bazlı kontrol test edilir.
NOW = datetime(2026, 10, 2, 20, 5, tzinfo=UTC)


def make_clock(start=NOW, step_s=20):
    state = {"t": pd.Timestamp(start)}

    def clock():
        t = state["t"]
        state["t"] = t + pd.Timedelta(seconds=step_s)
        return t.to_pydatetime()

    return clock


U = load_universe()
FIRST = {
    "RKLB": "2020-09-25",
    "HIMS": "2019-08-02",
    "SOFI": "2020-10-09",
    "XLC": "2018-06-19",
    "UFO": "2019-04-16",
    "NBIS": "2016-01-04",
    "XYZ": "2025-01-21",
    "MARA": "2016-01-04",
}
SPAC_CUT = {"RKLB": "2021-08-25", "HIMS": "2021-01-21", "SOFI": "2021-06-01"}


def _sessions(start, end="2026-10-02"):
    s, _ = us_sessions(start, end)
    return s


def make(sym: str, rng) -> pd.DataFrame:
    info = next(
        (x for x in U["symbols"] if x["symbol"] == sym), {"asset": "equity", "expected_splits": []}
    )
    first = FIRST.get(sym) or info.get("expected_first_date") or "2016-01-04"
    if sym == "SQ":
        first = "2016-01-04"
    if info.get("asset") == "crypto":
        idx = pd.date_range("2016-01-01", "2026-10-02", freq="D", tz="UTC")
    else:
        idx = _sessions(first).tz_localize("America/New_York")
    n = len(idx)
    vol = {"index_vol": 0.06, "index_rate": 0.02}.get(info.get("asset"), 0.025)
    r = rng.normal(0.0004, vol, n)
    if sym in SPAC_CUT:
        cut = idx.tz_localize(None) < pd.Timestamp(SPAC_CUT[sym])
        r[cut] = rng.normal(0, 0.002, cut.sum())
    close = 10.0 * np.exp(np.cumsum(r))
    if sym == "^TNX":
        close = 2.0 + np.abs(np.cumsum(rng.normal(0, 0.03, n)))
    if sym == "^VIX":
        close = 15 + 5 * np.abs(np.sin(np.arange(n) / 50)) + rng.normal(0, 0.5, n)
    o = close * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    if info.get("asset", "").startswith("index"):
        v[:] = 0
    df = pd.DataFrame(
        {
            "Open": o,
            "High": h,
            "Low": l,
            "Close": close,
            "Volume": v,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.DatetimeIndex(idx, name="Date"),
    )
    naive = idx.tz_localize(None)
    for d, ratio in info.get("expected_splits", []):
        pos = naive.searchsorted(pd.Timestamp(d))
        if pos < n:
            df.iloc[pos, df.columns.get_loc("Stock Splits")] = ratio
    # temettüler (Yahoo tarzı Adj Close)
    if sym in ("JPM", "XOM", "WMT", "AAPL", "MSFT", "SPY", "QQQ", "XLE"):
        for pos in range(60, n, 63):
            df.iloc[pos, df.columns.get_loc("Dividends")] = round(
                df["Close"].iloc[pos - 1] * 0.006, 4
            )
    factor = np.ones(n)
    divs, c = df["Dividends"].values, df["Close"].values
    for i in range(n - 1, 0, -1):
        factor[i - 1] = factor[i] * (1 - divs[i] / c[i - 1]) if divs[i] else factor[i]
    df["Adj Close"] = df["Close"] * factor
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]

    # ---- enjekte edilen sorunlar
    if sym == "NVDA":  # birebir tekrar satır
        df = pd.concat([df, df.iloc[[100]]]).sort_index()
    if sym == "TSLA":  # farklı değerli tekrar
        x = df.iloc[[200]].copy()
        x["Close"] *= 1.01
        df = pd.concat([df, x]).sort_index()
    if sym == "AMD":  # 3 eksik seans
        df = df.drop(df.index[[500, 501, 900]])
    if sym == "MU":  # High < Close
        df.iloc[[300, 301], df.columns.get_loc("High")] = df["Close"].iloc[[300, 301]] * 0.98
    if sym == "SMCI":  # split ham bırakılmış: split öncesi fiyatlar x10
        pos = naive.searchsorted(pd.Timestamp("2024-10-01"))
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            df.iloc[:pos, df.columns.get_loc(col)] *= 10
    if sym == "PLUG":  # kayıtsız 1:10 ters split
        pos = 1500
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            df.iloc[pos:, df.columns.get_loc(col)] *= 10
    if sym == "NBIS":  # Yandex geçmişi + uzun askı boşluğu
        keep = (naive < pd.Timestamp("2022-02-28")) | (naive >= pd.Timestamp("2024-10-21"))
        df = df[keep[: len(df)]] if len(keep) == len(df) else df
    if sym == "MRVL":  # 2016'dan beklenen geçmiş 2020'de başlıyor
        df = df[df.index >= pd.Timestamp("2020-01-02", tz="America/New_York")]
    if sym == "ENPH":  # boş hacim
        df.iloc[[700, 701], df.columns.get_loc("Volume")] = np.nan
    if sym == "AMZN":  # hacim split'te DÜZELTİLMEMİŞ (fiyat düzeltilmiş)
        pos = naive.searchsorted(pd.Timestamp("2022-06-06"))
        df.iloc[pos:, df.columns.get_loc("Volume")] *= 20
    if sym == "GOOGL":  # fiyat kararı belirsiz (oran ~1/sqrt(20))
        pos = naive.searchsorted(pd.Timestamp("2022-07-18"))
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            df.iloc[pos:, df.columns.get_loc(col)] /= np.sqrt(20)
    if sym == "WMT":  # bir temettüde bozuk Adj faktörü
        p = np.where(df["Dividends"].values > 0)[0][3]
        df.iloc[:p, df.columns.get_loc("Adj Close")] *= 0.99
    return df


LEGACY_MODE = {"overlap": False}


def fetch(sym, start, end):
    if sym in ("NASA", "DRAM"):
        raise ValueError(f"{sym}: No data found, symbol may be delisted")
    if sym in ("XYZ", "SQ"):  # aynı şirket; senaryoya göre örtüşme
        full = make("SQ", np.random.default_rng(zlib.crc32(b"BLOCK")))
        ny = "America/New_York"
        xyz = full[full.index >= pd.Timestamp("2025-01-21", tz=ny)]
        if LEGACY_MODE["overlap"]:
            sq = full[full.index < pd.Timestamp("2025-03-01", tz=ny)]  # ~27 seans örtüşme
        else:
            sq = full[full.index < pd.Timestamp("2025-01-21", tz=ny)]  # örtüşme YOK
        return (sq if sym == "SQ" else xyz), {}
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    return make(sym, rng), {"currency": "USD", "instrumentType": "EQUITY"}


def earnings(sym):
    if sym in ("PTON",):
        raise RuntimeError("scrape failed")
    d = pd.date_range(
        "2022-02-01", "2026-11-01", freq="QS-FEB", tz="America/New_York"
    ) + pd.Timedelta(hours=16)
    rep_eps = np.where(d <= pd.Timestamp(NOW), 0.12, np.nan)
    if sym == "NVDA":  # geçmiş tarihli ama sonucu olmayan kayıt
        rep_eps[5] = np.nan
    return pd.DataFrame(
        {"EPS Estimate": 0.1, "Reported EPS": rep_eps},
        index=pd.DatetimeIndex(d, name="Earnings Date"),
    )


def write_refs(path, which=("NVDA", "TSLA", "AAPL")):
    """Sentetik 'gerçek' = sağlayıcı geri düzeltilmiş: gerçek kapanış = close×F, gerçek hacim = volume/F."""
    pts = {"NVDA": "2021-07-19", "TSLA": "2020-08-28", "AAPL": "2020-08-28"}
    rows = ["symbol,date,gercek_kapanis,gercek_hacim,kaynak"]
    for sym in which:
        df, _ = fetch(sym, None, None)
        dd, _ = to_daily(df.reset_index())
        dd = dd[~dd.index.duplicated()]
        d = pd.Timestamp(pts[sym])
        F = float(dd.loc[dd.index > d, "splits"].replace(0, 1).prod())
        rows.append(
            f"{sym},{pts[sym]},{dd.at[d, 'close'] * F},{dd.at[d, 'volume'] / F},sentetik-gercek"
        )
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return Path(path)


# --------------------------------------------------------------------------- bağımsız referans (brute force)
def ref_count(df, eligible_from, W, H, lo, hi, label_hi):
    x = df[df["eligible"]]
    start = max(pd.Timestamp(eligible_from), x.index.min())
    sess = list(us_sessions(start, x.index.max())[0])
    ok_rows = set()
    for d, r in x.iterrows():
        vals = [r.open, r.high, r.low, r.close]
        if (
            all(np.isfinite(vals))
            and np.isfinite(r.volume)
            and min(vals) > 0
            and r.volume >= 0
            and r.high >= max(r.open, r.close) - 1e-6 * abs(r.close)
            and r.low <= min(r.open, r.close) + 1e-6 * abs(r.close)
            and r.high >= r.low
        ):
            ok_rows.add(d)
    valid = [d in ok_rows for d in sess]
    n, cnt, prior = len(sess), 0, 0
    for j in range(n):
        if (
            valid[j]
            and prior >= W
            and lo <= sess[j] <= hi
            and (H == 0 or j + H < n and sess[j + H] <= label_hi and all(valid[j + 1 : j + H + 1]))
        ):
            cnt += 1
        prior += valid[j]
    return cnt


def run_pipeline(tmp, approved=False, refs=None):
    U2 = load_universe()
    for x in U2["symbols"]:
        if x["symbol"] == "XYZ":
            x["legacy_merge_approved"] = approved
    snap = download_snapshot(
        tmp, now=NOW, fetch_fn=fetch, earnings_fn=earnings, pause=0, universe=U2, clock=make_clock()
    )
    paths = Paths(tmp, snap.name)
    build_clean(paths, U2, reference_path=refs or (tmp / "bos_referans.csv"))
    q = run_quality(paths, U2)
    cov = coverage(paths, U2)
    write_report(paths, q, cov, U2)
    return paths, q, cov, U2


def main():
    checks = {}
    tmp = Path(tempfile.mkdtemp(prefix="qml_test_"))
    tmps = [tmp]
    try:
        # ================= ana senaryo: SQ örtüşmesiz, onaysız
        paths, q, cov, U2 = run_pipeline(tmp)
        R = q["results"]
        codes = lambda s: {i["code"] for i in R[s]["issues"]}
        import json

        man = json.loads((paths.raw / "manifest.json").read_text(encoding="utf-8"))
        cl = json.loads((paths.clean / "clean_log.json").read_text(encoding="utf-8"))

        # --- önceki kontroller
        checks["NVDA birebir tekrar"] = "TEKRAR_BIREBIR" in codes("NVDA")
        checks["TSLA farklı tekrar = HATA"] = (
            "TEKRAR_FARKLI" in codes("TSLA") and R["TSLA"]["durum"] == "HATA"
        )
        checks["AMD 3 eksik seans"] = R["AMD"]["takvim"]["eksik_seans"] == 3
        checks["MU High/Low ihlali"] = R["MU"]["ohlcv"]["high_low_ihlal"] == 2
        checks["PLUG gizli split"] = "GIZLI_SPLIT" in codes("PLUG")
        checks["WMT temettü faktörü"] = "TEMETTU_FAKTOR" in codes("WMT")
        checks["JPM temettü temiz"] = not ({"TEMETTU_FAKTOR", "ADJ_DEGISIM"} & codes("JPM"))
        checks["RKLB kesim öncesi eligible=False"] = (
            not load_clean(paths, "RKLB").query("index < '2021-08-25'")["eligible"].any()
        )
        checks["NBIS eligible takvimde boşluk yok"] = R["NBIS"]["takvim"]["eksik_seans"] == 0
        checks["Kripto kapsam dışı"] = not any(k in R for k in ("COIN", "MARA", "BTC-USD"))
        checks["29 şirket + 18 bağlam"] = (
            sum(x["asset"] == "equity" for x in U2["symbols"]) == 29
            and sum(x["asset"] != "equity" for x in U2["symbols"]) == 18
        )

        # --- (1) boş hacim
        checks["ENPH boş hacim = HATA"] = (
            "NAN_VOLUME" in codes("ENPH") and R["ENPH"]["ohlcv"]["nan_volume"] == 2
        )

        # --- (2) seans tamamlanma: sembolün kendi zamanı + gerçek kapanış
        ok_all, saw_drop, saw_keep = True, False, False
        for s, rec in man["symbols"].items():
            if rec["status"] != "OK" or s == "SQ":
                continue
            done = is_session_complete("2026-10-02", rec["downloaded_at_utc"])
            last = load_clean(paths, s).index.max()
            exp = pd.Timestamp("2026-10-02") if done else pd.Timestamp("2026-10-01")
            ok_all &= last == exp
            saw_drop |= not done
            saw_keep |= bool(done)
        checks["Kısmi gün sembol bazlı (hem çıkan hem kalan var)"] = (
            ok_all and saw_drop and saw_keep
        )
        df1 = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2025-11-26", "2025-11-28"]))
        a, _ = drop_partial_last(
            df1, "2025-11-28T18:30:00Z", "equity"
        )  # 13:30 ET, erken kapanış 13:00
        b, _ = drop_partial_last(df1, "2025-11-28T18:05:00Z", "equity")  # 13:05 ET (<13:15)
        c, _ = drop_partial_last(df1, "2025-11-28T17:30:00Z", "equity")  # 12:30 ET
        checks["Erken kapanış: 13:30 ET'de tam mum korunur"] = len(a) == 2
        checks["Erken kapanış: 13:05 ve 12:30 ET'de çıkarılır"] = len(b) == 1 and len(c) == 1
        df2 = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
        d_, _ = drop_partial_last(df2, "2026-10-02T20:10:00Z", "equity")  # 16:10 ET
        e_, _ = drop_partial_last(df2, "2026-10-02T20:20:00Z", "equity")  # 16:20 ET
        checks["Normal gün: 16:10 çıkar, 16:20 korur"] = len(d_) == 1 and len(e_) == 2

        # --- (3) checksum
        checks["Bilanço dosyalarında sha256 var"] = all(
            (r.get("earnings") or {}).get("sha256")
            for r in man["symbols"].values()
            if (r.get("earnings") or {}).get("status") == "OK"
        )
        checks["Temizlik öncesi checksum doğrulandı"] = (
            cl["checksum_dogrulama"]["kontrol_edilen"] > 70
            and not cl["checksum_dogrulama"]["uyusmayan"]
        )
        f = paths.raw / "earnings" / "NVDA.parquet"
        os.chmod(f, stat.S_IWUSR | stat.S_IRUSR)
        with open(f, "ab") as fh:
            fh.write(b"tamper")
        try:
            build_clean(paths, U2)
            checks["Bozulan dosyada temizlik DURUR"] = False
        except ChecksumError:
            checks["Bozulan dosyada temizlik DURUR"] = True

        # --- (4) TEST ifadesi
        md = (paths.reports / "kalite_raporu.md").read_text(encoding="utf-8")
        checks["Raporda 'içeriğe bakılmadı' yok, test kuralı var"] = (
            "içeriğe bakıl" not in md and "kullanılmaz" in md and "Test dönemi notu" in md
        )

        # --- (5a) referans YOK: oran işaretlerinden raw_* üretilmez
        nv = load_clean(paths, "NVDA")
        last_split = pd.Timestamp("2024-06-10")
        checks["Ref yok: NVDA raw_close son split öncesi NaN"] = (
            nv.loc[: last_split - pd.Timedelta(days=1), "raw_close"].isna().all()
        )
        checks["Ref yok: NVDA son split sonrası raw_close = close"] = (
            nv.loc[last_split:, "raw_close"] == nv.loc[last_split:, "close"]
        ).all()
        checks["Ref yok: NVDA raw_volume split öncesi NaN (bölünmedi)"] = (
            nv.loc[: last_split - pd.Timedelta(days=1), "raw_volume"].isna().all()
        )
        checks["Ref yok: split'siz hisse (ROKU) raw_close = close"] = (
            load_clean(paths, "ROKU")["raw_close"] == load_clean(paths, "ROKU")["close"]
        ).all()
        checks["Ref yok: yöntem 'dogrulanmadi' + uyarı"] = (
            cl["split_yontemi"]["fiyat"]["durum"] == "dogrulanmadi"
            and "RAW_FIYAT_YOK" in codes("NVDA")
            and "RAW_HACIM_YOK" in codes("NVDA")
        )
        checks["SMCI fiyat kırılma işareti = HATA"] = "SPLIT_FIYAT_KIRILMA" in codes("SMCI")
        checks["Rapor işaret ≠ doğrulama diyor"] = "inceleme işareti" in (
            paths.reports / "kalite_raporu.md"
        ).read_text(encoding="utf-8")

        # --- (5b) arkadaşın örneği: sağlayıcı ham, gerçek hacim 1000 sabit, 2:1 split
        idx = pd.DatetimeIndex(us_sessions("2024-01-02", "2024-03-29")[0])
        n = len(idx)
        k = 30
        px = np.where(np.arange(n) < k, 100.0, 50.0)
        ex = pd.DataFrame(
            {
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": 1000.0,
                "splits": 0.0,
            },
            index=idx,
        )
        ex.iloc[k, ex.columns.get_loc("splits")] = 2.0
        fl = split_flags(ex)
        none_m = {"fiyat": {"durum": "dogrulanmadi"}, "hacim": {"durum": "dogrulanmadi"}}
        a, _ = add_historical(ex, fl, none_m)
        checks["Örnek: hacim oranı 1 iken işaret 'duzeltilmis_gibi' (yalnız işaret)"] = (
            fl[0]["hacim_isaret"] == "duzeltilmis_gibi"
        )
        checks["Örnek: doğrulama yokken 1000 → 500 YAPILMAZ (NaN)"] = (
            a["raw_volume"].iloc[:k].isna().all() and not (a["raw_volume"] == 500).any()
        )
        ham_m = {
            "fiyat": {"durum": "dogrulandi", "yontem": "ham"},
            "hacim": {"durum": "dogrulandi", "yontem": "ham"},
        }
        b_, _notes_b = add_historical(ex, fl, ham_m)
        checks["Örnek: yöntem 'ham' doğrulanınca raw_volume = 1000"] = (
            b_["raw_volume"] == 1000
        ).all()
        checks["Örnek: yöntem 'ham' + fiyat 'ham_gibi' → raw_close = close"] = (
            b_["raw_close"] == ex["close"]
        ).all()
        # referans değerlendirme kuralları
        dd = {"A": ex.copy(), "B": ex.copy()}
        two = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "date": [idx[5], idx[6]],
                "gercek_kapanis": [100.0, 100.0],
                "gercek_hacim": [1000.0, 1000.0],
                "kaynak": "x",
            }
        )
        checks["Ref: 2 nokta yetmez"] = evaluate_method(dd, two)["fiyat"]["durum"] == "dogrulanmadi"
        three = pd.concat([two, two.iloc[[0]].assign(date=idx[7])])
        mm = evaluate_method(dd, three)
        checks["Ref: 3 nokta/2 sembol tutarlı → 'ham' doğrulandı"] = (
            mm["fiyat"]["durum"] == "dogrulandi" and mm["fiyat"]["yontem"] == "ham"
        )
        mixed = three.copy()
        mixed.iloc[2, mixed.columns.get_loc("gercek_kapanis")] = 200.0
        checks["Ref: karışık sonuç → doğrulanmadı"] = (
            evaluate_method(dd, mixed)["fiyat"]["durum"] == "dogrulanmadi"
        )
        bad = three.copy()
        bad.iloc[2, bad.columns.get_loc("gercek_kapanis")] = 137.0
        checks["Ref: tutarsız nokta → doğrulanmadı"] = (
            evaluate_method(dd, bad)["fiyat"]["neden"] == "tutarsız nokta var"
        )

        # --- (2b) başlangıçta eksik yıllar (arkadaşın örneği 2)
        start_err = {s for s in R if "BASLANGIC_EKSIK" in codes(s)}
        checks["MRVL 2020'den başlıyor → BASLANGIC_EKSIK HATA"] = (
            "MRVL" in start_err and R["MRVL"]["durum"] == "HATA"
        )
        checks["Yanlış alarm yok (yalnız MRVL ve birleşmemiş XYZ)"] = start_err == {"MRVL", "XYZ"}
        checks["MRVL başta eksik seans sayısı doğru"] = R["MRVL"]["gecmis"][
            "baslangicta_eksik_seans"
        ] == len(us_sessions("2016-01-04", "2020-01-01")[0])

        # --- (3b) bilanço: sağlayıcı takvim kaydı (arkadaşın örneği 3)
        bn = R["NVDA"]["bilanco"]
        md_ = (paths.reports / "kalite_raporu.md").read_text(encoding="utf-8")
        checks["Bilanço: sonucu olmayan geçmiş kayıt 'belirsiz', gerçekleşmiş sayılmaz"] = (
            bn["durumu_belirsiz"] == 1
            and bn["sonuc_kaydi_var"] == R["AMD"]["bilanco"]["sonuc_kaydi_var"] - 1
            and "BILANCO_BELIRSIZ" in codes("NVDA")
        )
        checks["Bilanço: gelecek kayıt 'planlanan'"] = bn["planlanan"] == 1
        checks["Raporda 'gerçekleşen ilan' iddiası yok"] = (
            "gerçekleşen" not in md_ and "sağlayıcı takvim kaydı" in md_
        )

        # --- (6) SQ birleştirmesi
        b = cl["symbols"]["XYZ"]["legacy_birlestirme"]
        checks["SQ örtüşmesiz + onaysız: birleşmez, HATA"] = (
            not b["birlesti"]
            and "LEGACY_KISA" in codes("XYZ")
            and load_clean(paths, "XYZ").index.min() == pd.Timestamp("2025-01-21")
        )
        checks["SQ örtüşmesiz: doğrulama geçmez"] = (
            b["dogrulama_gecti"] is False and b["ortusen_gun"] == 0
        )

        # --- (7) kapsama: bağımsız referansla
        W, Hs = U2["warmup_sessions"], U2["horizons"]
        long = cov["long"]
        mism = []
        for s in ["NVDA", "AMD", "ENPH", "MU", "RKLB", "HIMS", "PLTR", "RIVN"]:
            df = load_clean(paths, s)
            info = next(x for x in U2["symbols"] if x["symbol"] == s)
            ef = info.get("eligible_from") or df.index.min()
            for fold in U2["folds"]:
                te, vs, ve = (pd.Timestamp(fold[k]) for k in ("train_end", "val_start", "val_end"))
                row = long[(long.symbol == s) & (long.katman == fold["name"])].iloc[0]
                for H in Hs:
                    r_ = ref_count(df, ef, W, H, pd.Timestamp(U2["start"]), te, te)
                    if r_ != row[f"egitim_H{H}"]:
                        mism.append((s, fold["name"], H, r_, row[f"egitim_H{H}"]))
                r_ = ref_count(df, ef, W, 0, vs, ve, None)
                if r_ != row["val_sinyal_gun"]:
                    mism.append((s, fold["name"], "sinyal", r_, row["val_sinyal_gun"]))
                r_ = ref_count(df, ef, W, 20, vs, ve, ve)
                if r_ != row["val_etiketli_H20"]:
                    mism.append((s, fold["name"], "val20", r_, row["val_etiketli_H20"]))
        checks["Kapsama = bağımsız referans (8 hisse × 4 katman × 5 ölçü)"] = not mism
        if mism:
            print("  uyuşmazlık:", mism[:5])
        amd = long[(long.symbol == "AMD") & (long.katman == "K1")].iloc[0]
        nvd = long[(long.symbol == "NVDA") & (long.katman == "K1")].iloc[0]
        checks["AMD boşlukları örnek düşürür (kayıp raporlu)"] = (
            amd["egitim_H20"] < nvd["egitim_H20"] and amd["bosluk_kaybi_H20"] > 0
        )
        checks["ENPH boş hacim satırları örnek düşürür"] = (
            long[(long.symbol == "ENPH") & (long.katman == "K1")].iloc[0]["egitim_H20"]
            < nvd["egitim_H20"]
        )
        s_ = us_sessions("2016-01-04", "2020-12-31")[0]
        checks[f"NVDA K1 H20 = {len(s_) - 200 - 20}"] = int(nvd["egitim_H20"]) == len(s_) - 200 - 20

        # ================= referanslı senaryo: yöntem doğrulanır
        t5 = Path(tempfile.mkdtemp(prefix="qml_test_"))
        tmps.append(t5)
        refp = write_refs(t5 / "refs.csv")
        p5, q5, _, _ = run_pipeline(t5, refs=refp)
        cl5 = json.loads((p5.clean / "clean_log.json").read_text(encoding="utf-8"))
        R5 = q5["results"]
        c5 = lambda s: {i["code"] for i in R5[s]["issues"]}
        m5 = cl5["split_yontemi"]
        checks["Ref var: fiyat ve hacim 'geri_duzeltilmis' doğrulandı"] = (
            m5["fiyat"]["durum"] == "dogrulandi"
            and m5["fiyat"]["yontem"] == "geri_duzeltilmis"
            and m5["hacim"]["durum"] == "dogrulandi"
            and m5["hacim"]["yontem"] == "geri_duzeltilmis"
        )
        nv5 = load_clean(p5, "NVDA")
        i = nv5.index.get_loc(pd.Timestamp("2021-07-20"))
        checks["Ref var: NVDA raw_close split öncesi ×40, raw_volume ÷40"] = (
            abs(nv5["raw_close"].iloc[i - 1] / nv5["close"].iloc[i - 1] - 40) < 1e-9
            and abs(nv5["volume"].iloc[i - 1] / nv5["raw_volume"].iloc[i - 1] - 40) < 1e-9
        )
        sm5 = load_clean(p5, "SMCI")
        checks["Ref var: SMCI işareti yöntemle çelişir → öncesi NaN + HATA"] = sm5.loc[
            :"2024-09-30", "raw_close"
        ].isna().all() and "SPLIT_YONTEM_CELISKI" in c5("SMCI")
        checks["Ref var: AMZN hacim çelişkisi yalnız UYARI"] = any(
            i_["code"] == "SPLIT_YONTEM_CELISKI" and i_["severity"] == "UYARI"
            for i_ in R5["AMZN"]["issues"]
        )
        checks["Ref var: RAW_*_YOK uyarısı kalkar"] = not (
            {"RAW_FIYAT_YOK", "RAW_HACIM_YOK"} & c5("NVDA")
        )
        t6 = Path(tempfile.mkdtemp(prefix="qml_test_"))
        tmps.append(t6)
        refp2 = write_refs(t6 / "refs.csv", which=("NVDA", "TSLA"))
        p6, _, _, _ = run_pipeline(t6, refs=refp2)
        m6 = json.loads((p6.clean / "clean_log.json").read_text(encoding="utf-8"))["split_yontemi"]
        checks["Ref 2 nokta: doğrulanmaz, raw_* NaN kalır"] = (
            m6["fiyat"]["durum"] == "dogrulanmadi"
            and load_clean(p6, "NVDA").loc[:"2021-07-19", "raw_close"].isna().all()
        )

        # ================= SQ senaryoları
        LEGACY_MODE["overlap"] = False
        t2 = Path(tempfile.mkdtemp(prefix="qml_test_"))
        tmps.append(t2)
        p2, _q2, _, _ = run_pipeline(t2, approved=True)
        b2 = json.loads((p2.clean / "clean_log.json").read_text(encoding="utf-8"))["symbols"][
            "XYZ"
        ]["legacy_birlestirme"]
        checks["SQ onaylı ama örtüşmesiz: REDDEDİLİR"] = (
            not b2["birlesti"] and "REDDEDILDI" in b2["karar"]
        )

        LEGACY_MODE["overlap"] = True
        t3 = Path(tempfile.mkdtemp(prefix="qml_test_"))
        tmps.append(t3)
        p3, _q3, _, _ = run_pipeline(t3, approved=False)
        b3 = json.loads((p3.clean / "clean_log.json").read_text(encoding="utf-8"))["symbols"][
            "XYZ"
        ]["legacy_birlestirme"]
        checks["SQ örtüşmeli + onaysız: doğrulama geçer ama birleşmez"] = (
            b3["dogrulama_gecti"] and not b3["birlesti"]
        )

        t4 = Path(tempfile.mkdtemp(prefix="qml_test_"))
        tmps.append(t4)
        p4, _q4, _, _ = run_pipeline(t4, approved=True)
        b4 = json.loads((p4.clean / "clean_log.json").read_text(encoding="utf-8"))["symbols"][
            "XYZ"
        ]["legacy_birlestirme"]
        x4 = load_clean(p4, "XYZ")
        checks["SQ örtüşmeli + onaylı: birleşir, kaynak işaretli"] = (
            b4["birlesti"]
            and x4.index.min() == pd.Timestamp("2016-01-04")
            and (x4.loc[:"2025-01-17", "source_symbol"] == "SQ").all()
            and (x4.loc["2025-01-21":, "source_symbol"] == "XYZ").all()
            and x4.index.is_unique
        )
        LEGACY_MODE["overlap"] = False

        width = max(map(len, checks))
        for k, ok in checks.items():
            print(f"{'✓' if ok else '✗'} {k:<{width}}")
        out = Path(__file__).resolve().parents[1] / "tests" / "_son_data_test_raporu.md"
        out.write_text(md, encoding="utf-8")
        assert all(checks.values()), f"{sum(not v for v in checks.values())} kontrol başarısız"
        print(f"\nTÜM KONTROLLER GEÇTİ ({len(checks)})")
    finally:
        for t in tmps:
            for root, dirs, files in os.walk(t):
                for fn in files:
                    try:
                        os.chmod(os.path.join(root, fn), stat.S_IWUSR | stat.S_IRUSR)
                    except OSError:
                        pass
            shutil.rmtree(t, ignore_errors=True)


def test_synthetic_pipeline(monkeypatch):
    # Retry timing is irrelevant to offline fixtures; exercise failures without sleeping.
    monkeypatch.setattr("quant_ml.data.download.time.sleep", lambda seconds: None)
    main()


if __name__ == "__main__":
    main()
