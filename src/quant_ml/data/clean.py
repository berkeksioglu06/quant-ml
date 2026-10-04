"""Ham snapshot -> temiz katman. Ham dosyaya dokunulmaz; her işlem clean_log'a yazılır.

Yapılanlar:
  * Önce bütün ham dosyaların sha256'sı manifest'e karşı doğrulanır; uyuşmazlıkta temizleme DURUR.
  * Tarih: sağlayıcının saat dilimindeki gün -> tz'siz tarih
  * Aynı tarihte tekrar eden satırlar: birebir aynıysa tekilleştirilir; farklıysa son satır tutulur, HATA olarak loglanır
  * Seansı tamamlanmamış son gün çıkarılır. Kontrol, o günün GERÇEK kapanışı (erken kapanış dahil,
    XNYS takvimi) + 15 dk ile, sembolün KENDİ indirme zamanına göre yapılır.
  * Split oranları (fiyat/hacim) yalnız İNCELEME İŞARETİ üretir; doğrulama sayılmaz.
  * Sağlayıcının split yöntemi (fiyat ve hacim ayrı) config/data/reference_points.csv'deki BAĞIMSIZ kaynaklı gerçek
    değerlerle belirlenir. Doğrulanmadıysa raw_close / raw_volume split öncesinde NaN kalır.
  * Son split'ten sonraki satırlarda sağlayıcı değeri = tarihsel değer (geri düzeltme etkilemez).
  * eligible kolonu: SPAC / yeniden yapılanma kesim tarihinden önceki satırlar SİLİNMEZ, eligible=False işaretlenir
  * SQ -> XYZ: OTOMATİK BİRLEŞTİRME YOK. Doğrulama raporu her zaman üretilir; birleştirme yalnız
    universe.json'da legacy_merge_approved=true VE tüm doğrulamalar geçerse yapılır.
Yapılmayanlar: ileri/geri doldurma, sıçrama silme, ikinci kez split/temettü düzeltmesi, takvim dışı satır yaratma.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    ROOT,
    Paths,
    is_session_complete,
    load_universe,
    safe_name,
    sha256,
    us_sessions,
    write_json,
)

COLMAP = {
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

LEGACY_MIN_OVERLAP = 20  # birleştirme için gereken en az örtüşen seans
LEGACY_TOL = 0.005  # örtüşmede izin verilen göreli fark (close, adj_close)
SPLIT_WINDOW = 20  # hacim kararı için split öncesi/sonrası medyan penceresi


class ChecksumError(RuntimeError):
    pass


# --------------------------------------------------------------------------- IO
def read_raw(raw_dir: Path, symbol: str) -> pd.DataFrame | None:
    p = raw_dir / "prices" / f"{safe_name(symbol)}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def verify_checksums(raw_dir: Path, manifest: dict) -> dict:
    """Manifest'teki her fiyat ve bilanço dosyasının sha256'sını yeniden hesaplar."""
    res = {"kontrol_edilen": 0, "uyusmayan": [], "eksik_dosya": [], "checksumsiz": []}
    for sym, rec in manifest["symbols"].items():
        items = []
        if rec.get("status") == "OK":
            items.append((rec.get("file"), rec.get("sha256"), "fiyat"))
        e = rec.get("earnings") or {}
        if e.get("status") == "OK":
            items.append((e.get("file"), e.get("sha256"), "bilanço"))
        for rel, digest, kind in items:
            p = raw_dir / rel
            if not p.exists():
                res["eksik_dosya"].append(f"{sym} {kind}")
                continue
            if not digest:
                res["checksumsiz"].append(f"{sym} {kind}")
                continue
            res["kontrol_edilen"] += 1
            if sha256(p) != digest:
                res["uyusmayan"].append(f"{sym} {kind}")
    return res


# --------------------------------------------------------------------------- dönüşümler
def to_daily(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ham çerçeveyi tarih indeksli, küçük harfli kolonlu çerçeveye çevirir (değer değiştirmez)."""
    log = {}
    ts = pd.to_datetime(raw["Date"])
    local = ts if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    not_midnight = (local.dt.hour != 0) | (local.dt.minute != 0)
    log["saat_gece_yarisi_degil"] = int(not_midnight.sum())
    dates = local.dt.tz_localize(None).dt.normalize()
    df = raw.drop(columns=["Date"]).rename(columns=COLMAP).copy()
    df.index = pd.DatetimeIndex(dates, name="date")
    for c in ["dividends", "splits", "capital_gains"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    return df, log


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"tekrar_tarih": 0, "tekrar_birebir": 0, "tekrar_farkli": []}
    dup = df.index.duplicated(keep=False)
    if dup.any():
        log["tekrar_tarih"] = int(df.index[dup].nunique())
        for d, g in df[dup].groupby(level=0):
            if len(g.drop_duplicates()) == 1:
                log["tekrar_birebir"] += 1
            else:
                log["tekrar_farkli"].append(
                    {
                        "date": str(d.date()),
                        "rows": g[["open", "high", "low", "close", "volume"]].to_dict("records"),
                    }
                )
        df = df[~df.index.duplicated(keep="last")]
    return df.sort_index(), log


def drop_partial_last(
    df: pd.DataFrame, downloaded_at_utc: str, asset: str
) -> tuple[pd.DataFrame, dict]:
    """Son satırın seansı, sembolün indirildiği anda (gerçek kapanış + 15 dk) tamamlanmamışsa çıkar."""
    log = {"kismi_son_gun_cikarildi": None, "indirme_zamani": downloaded_at_utc}
    if asset == "crypto" or df.empty or not downloaded_at_utc:
        return df, log
    last = df.index[-1]
    done = is_session_complete(last, downloaded_at_utc)
    if done is False:
        log["kismi_son_gun_cikarildi"] = str(last.date())
        df = df.iloc[:-1]
    return df, log


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


# --------------------------------------------------------------------------- dış referansla yöntem doğrulama
REF_PRICE_TOL = 0.02  # gerçek kapanış ile hipotez arasında izin verilen göreli fark
REF_VOLUME_TOL = (
    0.15  # kaynaklar arası hacim farkı (konsolide / birincil borsa) daha büyük olabilir
)
REF_MIN_POINTS = 3
REF_MIN_SYMBOLS = 2


def load_reference_points(path: Path | None) -> pd.DataFrame:
    cols = ["symbol", "date", "gercek_kapanis", "gercek_hacim", "kaynak"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    r = pd.read_csv(path, comment="#", dtype={"symbol": str, "kaynak": str})
    for c in cols:
        if c not in r:
            r[c] = np.nan
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["gercek_kapanis"] = pd.to_numeric(r["gercek_kapanis"], errors="coerce")
    r["gercek_hacim"] = pd.to_numeric(r["gercek_hacim"], errors="coerce")
    return r.dropna(subset=["symbol", "date"])


def _future_split_product(df: pd.DataFrame, d: pd.Timestamp) -> float:
    s = df.loc[df.index > d, "splits"].replace(0, 1).astype(float)
    return float(s.prod()) if len(s) else 1.0


def evaluate_method(daily: dict, refs: pd.DataFrame) -> dict:
    """Sağlayıcının split yöntemini BAĞIMSIZ kaynaktan alınmış gerçek değerlerle belirler.
    Bilgi taşıyan nokta: tarihinden SONRA split olan nokta (F≠1).
      fiyat:  gerçek/sağlayıcı ≈ F  -> geri_duzeltilmis ;  ≈ 1 -> ham
      hacim:  gerçek/sağlayıcı ≈ 1/F -> geri_duzeltilmis ;  ≈ 1 -> ham
    Yöntem ancak ≥3 bilgi taşıyan nokta, ≥2 sembol ve TÜM noktalar aynı sonuçta ise 'dogrulandi' olur."""
    out = {}
    for kind, col, tol in (
        ("fiyat", "gercek_kapanis", REF_PRICE_TOL),
        ("hacim", "gercek_hacim", REF_VOLUME_TOL),
    ):
        pts = []
        for _, r in refs.dropna(subset=[col]).iterrows():
            df = daily.get(r["symbol"])
            d = pd.Timestamp(r["date"]).normalize()
            if df is None or d not in df.index:
                pts.append({"symbol": r["symbol"], "date": str(d.date()), "sonuc": "veride_yok"})
                continue
            prov = float(df.at[d, "close" if kind == "fiyat" else "volume"])
            F = _future_split_product(df, d)
            q = float(r[col]) / prov if prov else np.nan
            target_adj = F if kind == "fiyat" else 1 / F
            rec = {
                "symbol": r["symbol"],
                "date": str(d.date()),
                "F": F,
                "gercek/saglayici": round(q, 6),
                "kaynak": r.get("kaynak"),
            }
            if F == 1:
                rec["sonuc"] = "seviye_uyumlu" if abs(q - 1) <= tol else "TUTARSIZ"
            elif abs(q / target_adj - 1) <= tol:
                rec["sonuc"] = "geri_duzeltilmis"
            elif abs(q - 1) <= tol:
                rec["sonuc"] = "ham"
            else:
                rec["sonuc"] = "TUTARSIZ"
            pts.append(rec)
        info_pts = [p for p in pts if p.get("sonuc") in ("geri_duzeltilmis", "ham")]
        bad = [p for p in pts if p.get("sonuc") == "TUTARSIZ"]
        methods = {p["sonuc"] for p in info_pts}
        syms = {p["symbol"] for p in info_pts}
        ok = (
            len(info_pts) >= REF_MIN_POINTS
            and len(syms) >= REF_MIN_SYMBOLS
            and len(methods) == 1
            and not bad
        )
        out[kind] = {
            "durum": "dogrulandi" if ok else "dogrulanmadi",
            "yontem": methods.pop() if ok else None,
            "bilgi_tasiyan_nokta": len(info_pts),
            "sembol": sorted(syms),
            "tutarsiz": bad,
            "noktalar": pts,
            "neden": None
            if ok
            else (
                "referans yok"
                if not pts
                else "tutarsız nokta var"
                if bad
                else "noktalar farklı yöntem gösteriyor"
                if len(methods) > 1
                else f"yetersiz nokta (≥{REF_MIN_POINTS} nokta, ≥{REF_MIN_SYMBOLS} sembol gerekli)"
            ),
        }
    return out


def add_historical(
    df: pd.DataFrame, flags: list[dict], method: dict
) -> tuple[pd.DataFrame, list[dict]]:
    """raw_close / raw_volume yalnız yöntem DIŞ REFERANSLA doğrulandıysa split öncesine uzatılır.
    Son split'ten sonraki satırlarda (sonrasında split yok) sağlayıcı değeri = tarihsel değer.
    Doğrulanmış yöntemle çelişen fiyat işareti (ör. yöntem geri düzeltilmiş ama split 'ham_gibi')
    o split'in öncesini NaN yapar. Hacim işareti zayıf olduğundan yalnız raporlanır."""
    df = df.copy()
    n = len(df)
    notes = []
    for kind, col, raw_col, fcol in (
        ("fiyat", "close", "raw_close", "split_factor_price"),
        ("hacim", "volume", "raw_volume", "split_factor_volume"),
    ):
        m = method.get(kind, {})
        fac = np.ones(n)
        bad = np.zeros(n, dtype=bool)
        for e in flags:
            pos = df.index.get_loc(pd.Timestamp(e["date"]))
            before = np.arange(n) < pos
            if m.get("durum") != "dogrulandi":
                bad[before] = True
                continue
            flag = e[f"{kind}_isaret"]
            if m["yontem"] == "geri_duzeltilmis":
                fac[before] *= e["ratio"]
                conflict = flag == "ham_gibi"
            else:  # ham: sağlayıcı değeri zaten tarihsel
                conflict = flag == "duzeltilmis_gibi"
            if conflict and kind == "fiyat":
                bad[before] = True
                notes.append(
                    {
                        "date": e["date"],
                        "tur": kind,
                        "not": f"doğrulanmış yöntem ({m['yontem']}) ile işaret ({flag}) çelişiyor",
                    }
                )
            elif conflict:
                notes.append(
                    {
                        "date": e["date"],
                        "tur": kind,
                        "not": f"hacim işareti ({flag}) yöntemle ({m['yontem']}) çelişiyor; yalnız inceleme",
                    }
                )
        df[fcol] = np.where(bad, np.nan, fac)
        df[raw_col] = df[col] * df[fcol] if kind == "fiyat" else df[col] / df[fcol]
    return df, notes


# --------------------------------------------------------------------------- legacy (SQ -> XYZ)
def evaluate_legacy(new: pd.DataFrame, old: pd.DataFrame, start_req: pd.Timestamp) -> dict:
    """Birleştirme adayını doğrular; KARAR VERMEZ."""
    r = {
        "yeni_ilk": str(new.index.min().date()),
        "eski_ilk": str(old.index.min().date()),
        "eski_son": str(old.index.max().date()),
    }
    r["gerekli"] = bool(new.index.min() > start_req + pd.Timedelta(days=30))
    overlap = new.index.intersection(old.index)
    r["ortusen_gun"] = len(overlap)
    checks = {
        "yeterli_ortusme": len(overlap) >= LEGACY_MIN_OVERLAP,
        "eski_daha_erken": bool(old.index.min() < new.index.min()),
    }
    if len(overlap):
        for col in ("close", "adj_close", "volume"):
            if col in new and col in old:
                a, b = new.loc[overlap, col].astype(float), old.loc[overlap, col].astype(float)
                ok = (b != 0) & a.notna() & b.notna()
                rel = (a[ok] / b[ok] - 1).abs()
                r[f"{col}_max_fark"] = float(rel.max()) if len(rel) else None
                r[f"{col}_medyan_oran"] = float((a[ok] / b[ok]).median()) if len(rel) else None

        def _le(key, tol):
            v = r.get(key)
            return v is not None and np.isfinite(v) and v <= tol

        checks["close_uyumlu"] = _le("close_max_fark", LEGACY_TOL)
        checks["adj_close_uyumlu"] = _le("adj_close_max_fark", LEGACY_TOL)
        m = r.get("close_medyan_oran")
        checks["olcek_ayni"] = m is not None and np.isfinite(m) and abs(m - 1) <= LEGACY_TOL
        # split kayıtları örtüşmede aynı olmalı
        sn = new.loc[overlap, "splits"].replace(0, 1)
        so = old.loc[overlap, "splits"].replace(0, 1)
        checks["split_kayitlari_ayni"] = bool((sn.values == so.values).all())
    r["kontroller"] = checks
    r["dogrulama_gecti"] = all(checks.values()) and len(checks) >= 6
    return r


def merge_legacy(new: pd.DataFrame, old: pd.DataFrame, old_sym: str, new_sym: str) -> pd.DataFrame:
    pre = old[old.index < new.index.min()].copy()
    pre["source_symbol"] = old_sym
    post = new.copy()
    post["source_symbol"] = new_sym
    return pd.concat([pre, post]).sort_index()


# --------------------------------------------------------------------------- ana akış
def build_clean(
    paths: Paths, universe: dict | None = None, reference_path: Path | None = None
) -> dict:
    u = universe or load_universe()
    reference_path = reference_path or (ROOT / "config" / "data" / "reference_points.csv")
    with open(paths.raw / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    cs = verify_checksums(paths.raw, manifest)
    if cs["uyusmayan"] or cs["eksik_dosya"]:
        raise ChecksumError(
            f"Ham snapshot doğrulanamadı, temizleme durduruldu: "
            f"uyuşmayan={cs['uyusmayan']} eksik={cs['eksik_dosya']}"
        )

    paths.clean.mkdir(parents=True, exist_ok=True)
    full_log = {
        "snapshot_id": paths.snapshot_id,
        "created_at": datetime.now(UTC).isoformat(),
        "kurallar": __doc__,
        "checksum_dogrulama": cs,
        "symbols": {},
    }
    legacy_map = manifest.get("legacy_symbols", {})  # {"SQ": "XYZ"}
    daily: dict[str, pd.DataFrame] = {}
    info_of = {s["symbol"]: s for s in u["symbols"]}

    for sym, rec in manifest["symbols"].items():
        if rec.get("status") != "OK":
            continue
        raw = read_raw(paths.raw, sym)
        df, l1 = to_daily(raw)
        df, l2 = dedupe(df)
        info = info_of.get(sym, {"asset": "equity"})
        ts = rec.get("downloaded_at_utc") or manifest["downloaded_at_utc"]
        df, l3 = drop_partial_last(df, ts, info.get("asset", "equity"))
        daily[sym] = df
        full_log["symbols"][sym] = {**l1, **l2, **l3, "raw_rows": len(raw)}

    # legacy: yalnız raporla; onay + doğrulama varsa birleştir
    for old_sym, new_sym in legacy_map.items():
        if new_sym not in daily:
            continue
        info = info_of[new_sym]
        entry = {"eski": old_sym, "onay": bool(info.get("legacy_merge_approved", False))}
        if old_sym not in daily:
            entry.update(
                karar="eski sembol verisi yok",
                birlesti=False,
                gerekli=bool(
                    daily[new_sym].index.min() > pd.Timestamp(u["start"]) + pd.Timedelta(days=30)
                ),
            )
        else:
            ev = evaluate_legacy(daily[new_sym], daily[old_sym], pd.Timestamp(u["start"]))
            entry.update(ev)
            if not ev["gerekli"]:
                entry.update(
                    karar="gerek yok: yeni sembol geçmişi istenen başlangıcı kapsıyor",
                    birlesti=False,
                )
            elif not entry["onay"]:
                entry.update(
                    karar="ONAY BEKLIYOR: birleştirme yapılmadı (universe.json legacy_merge_approved)",
                    birlesti=False,
                )
            elif not ev["dogrulama_gecti"]:
                entry.update(karar="REDDEDILDI: onay var ama doğrulama geçmedi", birlesti=False)
            else:
                daily[new_sym] = merge_legacy(daily[new_sym], daily[old_sym], old_sym, new_sym)
                entry.update(karar="birleştirildi (onaylı + doğrulandı)", birlesti=True)
        full_log["symbols"][new_sym]["legacy_birlestirme"] = entry

    refs = load_reference_points(reference_path)
    method = evaluate_method(daily, refs)
    full_log["split_yontemi"] = {
        "referans_dosyasi": str(reference_path),
        "referans_satir": len(refs),
        **method,
    }

    for sym, df in daily.items():
        info = info_of.get(sym, {"asset": "equity", "roles": ["legacy"]})
        flags = split_flags(df) if info.get("asset") in ("equity", "etf") else []
        df, notes = add_historical(df, flags, method)
        ef = pd.Timestamp(info["eligible_from"]) if info.get("eligible_from") else df.index.min()
        df["eligible"] = df.index >= ef
        if "source_symbol" not in df.columns:
            df["source_symbol"] = sym
        if info.get("asset") != "crypto" and len(df):
            sess, _ = us_sessions(df.index.min(), df.index.max())
            df["is_session"] = df.index.isin(sess) if len(sess) else True
        else:
            df["is_session"] = True
        df.to_parquet(paths.clean / f"{safe_name(sym)}.parquet")
        L = full_log["symbols"].setdefault(sym, {})
        L.update(
            clean_rows=len(df),
            eligible_from=str(ef.date()),
            eligible_rows=int(df["eligible"].sum()),
            split_isaretleri=flags,
            split_celiski=notes,
            raw_close_nan=int(df["raw_close"].isna().sum()),
            raw_volume_nan=int(df["raw_volume"].isna().sum()),
        )

    write_json(full_log, paths.clean / "clean_log.json")
    return full_log


def load_clean(paths: Paths, symbol: str) -> pd.DataFrame | None:
    p = paths.clean / f"{safe_name(symbol)}.parquet"
    return pd.read_parquet(p) if p.exists() else None
