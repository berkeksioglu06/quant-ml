"""Hisse × katman kapsama matrisi — ÖN SAYIM.

Takvim bazlıdır ve geçerli bar şartı arar; etiketler (bariyer, belirsiz mum) hesaplanınca kesinleşir.
Bu matris kesin eğitim uygunluğu ilan etmez.

Tanımlar (XNYS seans takvimi üzerinde, satır sırası üzerinde DEĞİL):
  * Geçerli bar: seans günü verisi var, OHLC sonlu ve > 0, hacim sonlu ve ≥ 0, High/Low tutarlı.
  * Isınma: t'den ÖNCE en az W=200 geçerli bar (yalnız eligible dönemde).
  * Sinyal (tahmin) uygunluğu: t geçerli bar ve ısınma tamam.
  * Etiketli örnek (H): sinyal uygun VE t+1 … t+H seanslarının HEPSİ geçerli bar.
    Giriş t+1 açılışı, son bar t+H. Arada eksik/bozuk seans varsa örnek düşer ("boşluk kaybı" olarak sayılır).
  * Purge: eğitim örneğinde t+H seansının tarihi ≤ training sonu.
  * Ön eşik: H=20 için ≥ 252 eğitim örneği -> YETER.
  * Validation etiketli (H=20): etiketi validation yılı içinde biten örnekler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import load_clean
from .common import Paths, load_universe, us_sessions

TRAIN_ROLES = ("ana20", "kontrol8", "nbis")


def valid_bars(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    fin = np.isfinite(df[["open", "high", "low", "close"]]).all(axis=1) & np.isfinite(v)
    pos = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (v >= 0)
    tol = 1e-6 * c.abs()
    hl = (h + tol >= np.maximum(o, c)) & (l - tol <= np.minimum(o, c)) & (h + tol >= l)
    return (fin & pos & hl).fillna(False)


class SessionGrid:
    """Bir sembolün eligible döneminin seans ızgarası ve geçerlilik dizisi."""

    def __init__(self, df: pd.DataFrame, eligible_from: pd.Timestamp, ref_dates=None):
        x = df[df["eligible"]]
        start = max(eligible_from, x.index.min()) if len(x) else eligible_from
        end = x.index.max() if len(x) else eligible_from
        sess, _ = us_sessions(start, end)
        if not len(sess) and ref_dates is not None:
            sess = ref_dates[(ref_dates >= start) & (ref_dates <= end)]
        self.sessions = pd.DatetimeIndex(sess)
        vb = valid_bars(x).reindex(self.sessions, fill_value=False)
        self.valid = vb.values.astype(bool)
        cs = np.concatenate([[0], np.cumsum(self.valid)])
        self.prior_valid = cs[:-1]  # t'den önceki geçerli bar sayısı
        self._cs = cs

    def signal_ok(self, W: int) -> np.ndarray:
        return self.valid & (self.prior_valid >= W)

    def label_ok(self, H: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.sessions)
        j = np.arange(n)
        has = j + H < n
        nxt = np.zeros(n, dtype=bool)
        nxt[has] = (self._cs[j[has] + H + 1] - self._cs[j[has] + 1]) == H
        end = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        end[has] = self.sessions.values[j[has] + H]
        return nxt, end, has

    def count(self, W, H, lo, hi, label_hi):
        d = self.sessions.values
        base = self.signal_ok(W) & (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))
        if H == 0:
            return int(base.sum()), 0
        nxt, end, has = self.label_ok(H)
        in_bound = has & (end <= np.datetime64(label_hi))
        ok = base & in_bound & nxt
        lost = base & in_bound & ~nxt
        return int(ok.sum()), int(lost.sum())


def coverage(paths: Paths, universe: dict | None = None) -> dict:
    u = universe or load_universe()
    W, MIN = u["warmup_sessions"], u["min_train_samples"]
    Hs = u["horizons"]
    Hmax = max(Hs)
    start = pd.Timestamp(u["start"])
    test_start = pd.Timestamp(u["test"]["start"])
    rows, ctx_rows = [], []
    spy = load_clean(paths, "SPY")
    ref_dates = spy.index if spy is not None else None

    for info in u["symbols"]:
        sym = info["symbol"]
        df = load_clean(paths, sym)
        roles = ",".join(info["roles"])
        if not set(info["roles"]) & set(TRAIN_ROLES):
            if df is None or info["asset"] == "crypto":
                ctx_rows.append(
                    {
                        "symbol": sym,
                        "roles": roles,
                        "ilk_tarih": None if df is None else str(df.index.min().date()),
                        "durum": "veri yok" if df is None else "7/24 (ayrı hizalama)",
                    }
                )
                continue
            first = df.index.min()
            warm = df.index[W] if len(df) > W else None
            rec = {
                "symbol": sym,
                "roles": roles,
                "ilk_tarih": str(first.date()),
                "isinma_bitis": str(warm.date()) if warm is not None else None,
                "2016_basindan_tam": bool(first <= start + pd.Timedelta(days=7)),
            }
            for f in u["folds"]:
                te = pd.Timestamp(f["train_end"])
                rec[f["name"]] = (
                    "tam" if rec["2016_basindan_tam"] else ("kısmi" if first <= te else "yok")
                )
            rec["TEST"] = "var" if df.index.max() >= test_start else "yok"
            ctx_rows.append(rec)
            continue

        if df is None:
            for f in u["folds"] + [u["test"]]:
                rows.append(
                    {"symbol": sym, "roles": roles, "katman": f["name"], "durum": "VERI YOK"}
                )
            continue
        ef = pd.Timestamp(info["eligible_from"]) if info.get("eligible_from") else df.index.min()
        g = SessionGrid(df, ef, ref_dates)
        base = {
            "symbol": sym,
            "roles": roles,
            "ilk_eligible": str(g.sessions.min().date()) if len(g.sessions) else None,
            "eligible_seans": len(g.sessions),
            "gecersiz_veya_eksik_seans": int((~g.valid).sum()),
        }
        sig = np.where(g.signal_ok(W))[0]
        base["isinma_bitis"] = str(g.sessions[sig[0]].date()) if len(sig) else None
        far = pd.Timestamp("2100-01-01")
        for f in u["folds"]:
            te, vs, ve = (pd.Timestamp(f[k]) for k in ("train_end", "val_start", "val_end"))
            rec = {**base, "katman": f["name"]}
            for H in Hs:
                rec[f"egitim_H{H}"], rec[f"bosluk_kaybi_H{H}"] = g.count(W, H, start, te, te)
            rec["val_sinyal_gun"], _ = g.count(W, 0, vs, ve, None)
            rec[f"val_etiketli_H{Hmax}"], _ = g.count(W, Hmax, vs, ve, ve)
            n = rec[f"egitim_H{Hmax}"]
            if "nbis" in info["roles"]:
                rec["durum"] = "EGITIM DISI (NBIS)"
            else:
                rec["durum"] = "YETER" if n >= MIN else ("AZ" if n > 0 else "YOK")
            rows.append(rec)
        rec = {**base, "katman": u["test"]["name"]}
        rec["val_sinyal_gun"], _ = g.count(W, 0, test_start, far, None)
        rec[f"val_etiketli_H{Hmax}"], _ = g.count(W, Hmax, test_start, far, far)
        rec["durum"] = "SINYAL VAR" if rec["val_sinyal_gun"] > 0 else "YOK"
        rows.append(rec)

    long = pd.DataFrame(rows)
    ctx = pd.DataFrame(ctx_rows)

    # Pivot: hücre = "durum eğitim(H20) / val sinyal"
    def cell(r):
        if r.get("durum") == "VERI YOK":
            return "VERİ YOK"
        if r["katman"] == "TEST":
            return f"{r['val_sinyal_gun']} sinyal"
        return (
            f"{r['durum'].split(' (')[0]} {int(r[f'egitim_H{Hmax}'])} / {int(r['val_sinyal_gun'])}"
        )

    long["hucre"] = long.apply(cell, axis=1)
    order = [s["symbol"] for s in u["symbols"]]
    pivot = long.pivot(index="symbol", columns="katman", values="hucre")
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    pivot = pivot[[f["name"] for f in u["folds"]] + [u["test"]["name"]]]

    # Özet: her katmanda kaç hisse eğitime giriyor
    summ = []
    groups = {"Ana 20": "ana20", "Kontrol 8": "kontrol8", "İşlem 10": "islem10"}
    for f in u["folds"]:
        r = {"katman": f["name"], "train_end": f["train_end"]}
        sub = long[long["katman"] == f["name"]]
        for g, role in groups.items():
            gs = sub[sub["roles"].str.contains(role)]
            r[g] = f"{int((gs['durum'] == 'YETER').sum())}/{len(gs)}"
        b20 = sub[sub["roles"].str.contains("ana20")]
        b28 = sub[sub["roles"].str.contains("ana20|kontrol8")]
        r[f"Odaklı-20 örnek (H{Hmax})"] = int(
            b20.loc[b20["durum"] == "YETER", f"egitim_H{Hmax}"].sum()
        )
        r[f"Geniş-28 örnek (H{Hmax})"] = int(
            b28.loc[b28["durum"] == "YETER", f"egitim_H{Hmax}"].sum()
        )
        it = sub[sub["roles"].str.contains("islem10")]
        r["İşlem-10 val sinyal günü"] = int(it["val_sinyal_gun"].sum())
        summ.append(r)
    summary = pd.DataFrame(summ)

    paths.reports.mkdir(parents=True, exist_ok=True)
    long.drop(columns=["hucre"]).to_csv(
        paths.reports / "kapsama_uzun.csv", index=False, encoding="utf-8-sig"
    )
    pivot.to_csv(paths.reports / "kapsama_matrisi.csv", encoding="utf-8-sig")
    ctx.to_csv(paths.reports / "baglam_kapsama.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths.reports / "kapsama_ozet.csv", index=False, encoding="utf-8-sig")
    return {"long": long, "pivot": pivot, "context": ctx, "summary": summary}
