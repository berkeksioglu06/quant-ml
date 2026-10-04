"""Kalite kontrolleri. Hiçbir kontrol veriyi değiştirmez; sorunlar sembol bazında listelenir.

Ciddiyet: HATA (eğitimden önce çözülmeli) / UYARI (incelenmeli) / BILGI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .clean import load_clean, read_raw, to_daily
from .common import NY_TZ, Paths, last_complete_session, load_universe, safe_name, us_sessions

BIG_MOVE = 0.25  # |kapanış-kapanış getirisi| bilgi eşiği
DIV_TOL = 0.002  # temettü faktörü mutlak toleransı
SPLIT_WINDOW_DAYS = 4  # beklenen split tarihi toleransı


def _issue(lst, sev, code, msg):
    lst.append({"severity": sev, "code": code, "msg": msg})


def expected_last_session(downloaded_at_utc: str) -> pd.Timestamp | None:
    """Sembolün indirildiği anda tamamlanmış son seans (erken kapanışlar takvimden)."""
    return last_complete_session(downloaded_at_utc)


# --------------------------------------------------------------------------- tekil kontroller
def check_ohlcv(df: pd.DataFrame, asset: str, issues: list) -> dict:
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    r = {}
    r["nan_ohlc"] = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    r["nan_volume"] = int(v.isna().sum())
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


def check_splits(
    df: pd.DataFrame, info: dict, flags: list, notes: list, method: dict, issues: list
) -> dict:
    """Split oranları yalnız İNCELEME İŞARETİDİR. raw_* üretimi dış referansla doğrulanan yönteme bağlıdır."""
    out = [dict(e) for e in flags]
    for e in flags:
        tag = f"{e['date']} {e['ratio']:g}:1"
        if e["fiyat_isaret"] == "ham_gibi":
            _issue(
                issues,
                "HATA",
                "SPLIT_FIYAT_KIRILMA",
                f"{tag} split günü fiyat oranı {e.get('close_ratio')} — sağlayıcı fiyatında split kırılması şüphesi; "
                "özellik ve etiket hesaplarını bozar, incelenmeli",
            )
        elif e["fiyat_isaret"] == "belirsiz":
            _issue(
                issues,
                "UYARI",
                "SPLIT_FIYAT_ISARET",
                f"{tag} split günü fiyat oranı belirsiz (kapanış {e.get('close_ratio')}, açılış {e.get('open_ratio')})",
            )
        if e["hacim_isaret"] != "duzeltilmis_gibi":
            _issue(
                issues,
                "BILGI",
                "SPLIT_HACIM_ISARET",
                f"{tag} 20g hacim oranı {e.get('volume_ratio')} ({e['hacim_isaret']}); yalnız inceleme işareti",
            )
    for nt in notes:
        sev = "HATA" if nt["tur"] == "fiyat" else "UYARI"
        _issue(issues, sev, "SPLIT_YONTEM_CELISKI", f"{nt['date']} {nt['not']}")
    if flags:
        for kind, col in (("fiyat", "raw_close"), ("hacim", "raw_volume")):
            m = method.get(kind, {})
            if m.get("durum") != "dogrulandi":
                n_nan = int(df[col].isna().sum())
                _issue(
                    issues,
                    "UYARI",
                    f"RAW_{'FIYAT' if kind == 'fiyat' else 'HACIM'}_YOK",
                    f"sağlayıcının {kind} split yöntemi dış referansla doğrulanmadı ({m.get('neden')}); "
                    f"son split öncesi {n_nan} satırda {col} üretilmedi",
                )

    exp = info.get("expected_splits", [])
    found = {(pd.Timestamp(e["date"]), e["ratio"]) for e in out}
    for d, ratio in exp:
        d = pd.Timestamp(d)
        ok = any(
            abs((fd - d).days) <= SPLIT_WINDOW_DAYS and abs(fr - ratio) < 1e-6 for fd, fr in found
        )
        if not ok:
            _issue(
                issues, "HATA", "SPLIT_EKSIK", f"beklenen {ratio:g}:1 split ({d.date()}) veride yok"
            )
    for fd, fr in found:
        if not any(abs((fd - pd.Timestamp(d)).days) <= SPLIT_WINDOW_DAYS for d, _ in exp):
            _issue(
                issues,
                "BILGI",
                "SPLIT_YENI",
                f"manifestoda olmayan split: {fd.date()} oran {fr:g} (manifestoya eklenmeli)",
            )
    return {"split_olaylari": out}


def check_dividends(df: pd.DataFrame, issues: list) -> dict:
    r = {}
    if "adj_close" not in df.columns or df["adj_close"].isna().all():
        _issue(issues, "HATA", "ADJ_YOK", "Adj Close kolonu yok/boş")
        return r
    ratio = df["adj_close"] / df["close"]
    payout = df["dividends"].fillna(0) + df.get("capital_gains", 0).fillna(0)
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


def check_earnings(
    paths: Paths, sym: str, price_first: pd.Timestamp, asof: pd.Timestamp, issues: list
) -> dict:
    """Yahoo bilanço kayıtları = SAĞLAYICI TAKVİM KAYDI. Gerçekleşme bu kaynakla doğrulanmaz.
    Sınıflar: planlanan (tarih > indirme anı) / sonuç_kaydı_var (geçmiş tarih + Reported EPS dolu) /
    durumu_belirsiz (geçmiş tarih + Reported EPS boş)."""
    p = paths.raw / "earnings" / f"{safe_name(sym)}.parquet"
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


def compare_legacy(
    paths: Paths, new_sym: str, old_sym: str, change_date: str, clean_log: dict, issues: list
) -> dict:
    r = {"eski_sembol": old_sym, "degisim_tarihi": change_date}
    raw_new = read_raw(paths.raw, new_sym)
    if raw_new is not None:
        dn, _ = to_daily(raw_new)
        r["yeni_ilk"], r["yeni_son"], r["yeni_satir"] = (
            str(dn.index.min().date()),
            str(dn.index.max().date()),
            len(dn),
        )
        cd = pd.Timestamp(change_date)
        around = dn[
            (dn.index >= cd - pd.Timedelta(days=7)) & (dn.index <= cd + pd.Timedelta(days=7))
        ]
        if len(around) > 1:
            r["degisim_cevresi_max_abs_getiri"] = round(
                float(around["close"].pct_change().abs().max()), 4
            )
    b = clean_log["symbols"].get(new_sym, {}).get("legacy_birlestirme") or {}
    r["birlestirme"] = b
    if b.get("gerekli") and not b.get("birlesti"):
        _issue(
            issues,
            "HATA",
            "LEGACY_KISA",
            f"{new_sym} geçmişi {r.get('yeni_ilk')}'de başlıyor; {old_sym} birleştirilmedi — {b.get('karar')}. "
            f"Doğrulama: {b.get('kontroller')}",
        )
    elif b.get("birlesti"):
        _issue(
            issues,
            "UYARI",
            "LEGACY_BIRLESTI",
            f"{old_sym} geçmişi onayla birleştirildi (örtüşme {b.get('ortusen_gun')} gün, close max fark "
            f"{b.get('close_max_fark')}); sınırdaki davranış elle kontrol edilmeli",
        )
    else:
        _issue(
            issues,
            "BILGI",
            "LEGACY_OK",
            f"{new_sym} geçmişi {r.get('yeni_ilk')}'den başlıyor; {old_sym} gerekmiyor ({b.get('karar')})",
        )
    return r


# --------------------------------------------------------------------------- toplu
def run_quality(paths: Paths, universe: dict | None = None) -> dict:
    u = universe or load_universe()
    with open(paths.raw / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(paths.clean / "clean_log.json", encoding="utf-8") as f:
        clean_log = json.load(f)
    exp_last = expected_last_session(manifest["downloaded_at_utc"])
    asof = pd.Timestamp(manifest["downloaded_at_utc"])
    asof = (
        asof.tz_convert(NY_TZ) if asof.tzinfo else asof.tz_localize("UTC").tz_convert(NY_TZ)
    ).tz_localize(None)
    spy = load_clean(paths, "SPY")
    ref_dates = spy.index if spy is not None else None

    results = {}
    for info in u["symbols"]:
        sym = info["symbol"]
        issues: list = []
        mrec = manifest["symbols"].get(sym, {"status": "ISTENMEDI"})
        res = {
            "symbol": sym,
            "roles": info["roles"],
            "asset": info["asset"],
            "indirme": mrec.get("status"),
            "event": info.get("event"),
            "issues": issues,
        }
        if mrec.get("status") != "OK":
            sev = "BILGI" if info.get("optional") else "HATA"
            _issue(
                issues,
                sev,
                "INDIRME",
                f"indirme durumu {mrec.get('status')}: {mrec.get('error', '')[:200]}",
            )
            results[sym] = res
            continue
        df = load_clean(paths, sym)
        cl = clean_log["symbols"].get(sym, {})
        if cl.get("tekrar_farkli"):
            _issue(
                issues,
                "HATA",
                "TEKRAR_FARKLI",
                f"{len(cl['tekrar_farkli'])} tarihte farklı değerli tekrar satır (son satır tutuldu)",
            )
        if cl.get("tekrar_birebir"):
            _issue(
                issues,
                "BILGI",
                "TEKRAR_BIREBIR",
                f"{cl['tekrar_birebir']} tarihte birebir tekrar satır tekilleştirildi",
            )
        if cl.get("kismi_son_gun_cikarildi"):
            _issue(
                issues,
                "BILGI",
                "KISMI_GUN",
                f"seansı bitmemiş {cl['kismi_son_gun_cikarildi']} çıkarıldı",
            )
        if cl.get("saat_gece_yarisi_degil"):
            _issue(
                issues,
                "UYARI",
                "SAAT",
                f"{cl['saat_gece_yarisi_degil']} satırın zaman damgası gece yarısı değil",
            )

        res["satir"] = len(df)
        res["eligible_satir"] = int(df["eligible"].sum())
        res["ohlcv"] = check_ohlcv(df, info["asset"], issues)
        res["takvim"] = check_calendar(
            df[df["eligible"]] if info.get("eligible_from") else df,
            info["asset"],
            issues,
            expected_last_session(mrec.get("downloaded_at_utc") or manifest["downloaded_at_utc"]),
            ref_dates,
        )
        res["gecmis"] = check_history(df, info, mrec.get("meta"), issues, pd.Timestamp(u["start"]))
        if info["asset"] in ("equity", "etf"):
            res["split"] = check_splits(
                df,
                info,
                cl.get("split_isaretleri", []),
                cl.get("split_celiski", []),
                clean_log.get("split_yontemi", {}),
                issues,
            )
            res["temettu"] = check_dividends(df, issues)
        res["hareket"] = check_moves(df, info["asset"], issues)
        if info["asset"].startswith("index"):
            res["endeks"] = check_index_scale(df, sym, info, issues)
        res["likidite"] = liquidity_stats(df, info)
        if info["asset"] == "equity":
            res["bilanco"] = check_earnings(paths, sym, df.index.min(), asof, issues)
        if info.get("legacy_symbol"):
            res["legacy"] = compare_legacy(
                paths, sym, info["legacy_symbol"], info["ticker_change_date"], clean_log, issues
            )
        results[sym] = res

    for r in results.values():
        sev = {i["severity"] for i in r["issues"]}
        r["durum"] = "HATA" if "HATA" in sev else ("UYARI" if "UYARI" in sev else "OK")
    return {
        "exp_last_session": str(exp_last.date()) if exp_last is not None else None,
        "checksum": clean_log.get("checksum_dogrulama"),
        "split_yontemi": clean_log.get("split_yontemi"),
        "manifest": {k: v for k, v in manifest.items() if k != "symbols"},
        "results": results,
    }
