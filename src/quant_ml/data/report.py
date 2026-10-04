"""Kalite raporu (Markdown) + CSV/JSON ekleri."""

from __future__ import annotations

import pandas as pd

from .common import Paths, load_universe, write_json


def _md_table(df: pd.DataFrame, index=False) -> str:
    if df is None or df.empty:
        return "_(boş)_\n"
    d = df.reset_index() if index else df

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e12:
            return str(int(v))
        return str(v)

    d = d.astype(object).map(fmt) if hasattr(d, "map") else d.astype(object).applymap(fmt)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append(
            "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r.values) + " |"
        )
    return "\n".join(lines) + "\n"


def write_report(paths: Paths, q: dict, cov: dict, universe: dict | None = None) -> str:
    u = universe or load_universe()
    R = q["results"]
    m = q["manifest"]
    out = []
    w = out.append

    # ---------------------------------------------------------------- 0. özet
    n_ok = sum(r["durum"] == "OK" for r in R.values())
    n_w = sum(r["durum"] == "UYARI" for r in R.values())
    n_e = sum(r["durum"] == "HATA" for r in R.values())
    w(f"# Veri kalite raporu — {m['snapshot_id']}\n")
    w(
        f"Kaynak: {m['source']} · indirme {m['downloaded_at_utc'][:19]} UTC · "
        f"istenen başlangıç {m['request']['start']} · beklenen son seans {q['exp_last_session']}\n"
    )
    v = m["versions"]
    w(
        f"Sürümler: yfinance {v.get('yfinance')}, pandas {v.get('pandas')}, pyarrow {v.get('pyarrow')}, "
        f"exchange_calendars {v.get('exchange_calendars')}, Python {v.get('python')}\n"
    )
    opts = {
        k: m["request"][k] for k in ("auto_adjust", "back_adjust", "actions", "repair", "interval")
    }
    w(f"İndirme seçenekleri: `{opts}` — ham snapshot salt-okunur, temizleme ayrı katmanda.\n")
    cs = q.get("checksum") or {}
    w(
        f"Checksum doğrulaması (temizlemeden önce): {cs.get('kontrol_edilen', 0)} dosya doğrulandı, "
        f"uyuşmayan {len(cs.get('uyusmayan', []))}, checksumsız {len(cs.get('checksumsiz', []))}.\n"
    )
    w(
        "\n> **Test dönemi notu:** Kalite kontrolleri (eksik seans, tutarlılık, sıçrama, likidite istatistikleri) "
        "2025–2026 test dönemini de kapsar; veri kalitesi için bu inceleme gereklidir. Kural: bu rapordaki test dönemi "
        "bilgisi model, özellik, eşik veya evren seçiminde **kullanılmaz**.\n"
    )
    w(f"\n**Durum:** {n_ok} OK · {n_w} UYARI · {n_e} HATA (toplam {len(R)} sembol)\n")

    blockers = [(s, i) for s, r in R.items() for i in r["issues"] if i["severity"] == "HATA"]
    w("\n## 1. Eğitimden önce çözülmesi gerekenler (HATA)\n")
    if blockers:
        for s, i in blockers:
            w(f"- **{s}** `{i['code']}` — {i['msg']}\n")
    else:
        w("- Yok.\n")

    # ---------------------------------------------------------------- 1. sembol özeti
    w("\n## 2. Sembol özeti\n")
    rows = []
    for s, r in R.items():
        t = r.get("takvim", {}) or {}
        o = r.get("ohlcv", {}) or {}
        g = r.get("gecmis", {}) or {}
        sp = (r.get("split") or {}).get("split_olaylari", [])
        rows.append(
            {
                "Sembol": s,
                "Rol": ",".join(r["roles"]),
                "Durum": r["durum"],
                "İndirme": r.get("indirme"),
                "Satır": r.get("satir"),
                "Eligible": r.get("eligible_satir"),
                "İlk": g.get("ilk_tarih"),
                "Kesim": g.get("kesim_tarihi", ""),
                "Son": g.get("son_tarih"),
                "Eksik seans": t.get("eksik_seans"),
                "Fazla tarih": t.get("fazla_tarih"),
                "H/L ihlal": o.get("high_low_ihlal"),
                "0 hacim": o.get("zero_volume"),
                "Split": len(sp),
                "Temettü": (r.get("temettu") or {}).get("temettu_olay"),
                "Büyük hareket": len((r.get("hareket") or {}).get("buyuk_hareket", [])),
            }
        )
    summary = pd.DataFrame(rows)
    w(_md_table(summary))

    # ---------------------------------------------------------------- 2. kurumsal işlemler
    w("\n## 3. Kurumsal işlemler\n")
    w(
        "`auto_adjust=False` gerçek tarihsel fiyat garantisi vermez. Split günü fiyat oranı ve 20 seans hacim oranı "
        "yalnız **inceleme işaretidir** (`*_gibi`), doğrulama sayılmaz. Sağlayıcının split yöntemi fiyat ve hacim için "
        "ayrı ayrı, `config/data/reference_points.csv`'deki bağımsız kaynaklı gerçek değerlerle belirlenir. Yöntem doğrulanmadıkça "
        "`raw_close` / `raw_volume` son split'ten önceki satırlarda üretilmez (NaN); ≥5$ filtresi o dönemde uygulanamaz.\n\n"
    )
    SM = q.get("split_yontemi") or {}
    for kind in ("fiyat", "hacim"):
        m = SM.get(kind) or {}
        w(
            f"- **{kind.capitalize()} yöntemi:** {m.get('durum')}"
            + (f" → `{m.get('yontem')}`" if m.get("yontem") else f" ({m.get('neden')})")
            + f" · bilgi taşıyan referans noktası {m.get('bilgi_tasiyan_nokta', 0)}, semboller {m.get('sembol', [])}\n"
        )
        for t in (m.get("tutarsiz") or [])[:5]:
            w(f"  - TUTARSIZ referans: {t}\n")
    w(
        f"- Referans dosyası: `{SM.get('referans_dosyasi')}` ({SM.get('referans_satir', 0)} satır)\n\n"
    )
    sp_rows = []
    for s, r in R.items():
        for e in (r.get("split") or {}).get("split_olaylari", []):
            sp_rows.append({"Sembol": s, **e})
    w(_md_table(pd.DataFrame(sp_rows)))
    sp_issues = [(s, i) for s, r in R.items() for i in r["issues"] if i["code"].startswith("SPLIT")]
    for s, i in sp_issues:
        w(f"- {s}: {i['msg']}\n")

    w(
        "\n**Temettü / Adj Close tutarlılığı** — her ex-date'te Adj/Close faktör değişimi, "
        "`1 − temettü/önceki kapanış` ile karşılaştırılır.\n\n"
    )
    dv = []
    for s, r in R.items():
        t = r.get("temettu")
        if t and t.get("temettu_olay"):
            dv.append(
                {
                    "Sembol": s,
                    "Olay": t["temettu_olay"],
                    "Max faktör hatası": round(t["temettu_faktor_max_hata"], 5),
                    "Uyumsuz olay": len(t["temettu_faktor_uyumsuz"]),
                    "Temettüsüz Adj değişimi": t["adj_aciklanamayan_degisim"],
                    "Son Adj/Close": round(t["adj_close_ratio_son"], 6),
                }
            )
    w(_md_table(pd.DataFrame(dv)))

    # ---------------------------------------------------------------- 3. geçmişin uygunluğu
    w("\n## 4. Geçmişin uygunluğu: SPAC, ticker değişimi, yeniden yapılanma, halka arz\n")
    w(
        "Kesim öncesi satırlar silinmedi; temiz katmanda `eligible=False`. SPAC imzası: kesim öncesi gerçek fiyatın "
        "~$10 bandında kalma oranı ve çok düşük günlük getiri.\n\n"
    )
    cut = []
    for s, r in R.items():
        g = r.get("gecmis") or {}
        if "kesim_tarihi" in g:
            pre, post = g.get("kesim_oncesi") or {}, g.get("kesim_sonrasi_ilk60") or {}
            cut.append(
                {
                    "Sembol": s,
                    "Kesim": g["kesim_tarihi"],
                    "Öncesi satır": g["kesim_oncesi_satir"],
                    "Öncesi aralık": f"{pre.get('ilk', '')} → {pre.get('son', '')}" if pre else "",
                    "Öncesi medyan $": pre.get("medyan_kapanis"),
                    "Öncesi $9–11.5 payı": pre.get("9_11.5_bandi"),
                    "Öncesi medyan |r|": pre.get("medyan_abs_getiri"),
                    "Sonrası (60g) medyan |r|": post.get("medyan_abs_getiri"),
                    "Sonrası satır": g["kesim_sonrasi_satir"],
                }
            )
    w(_md_table(pd.DataFrame(cut)))
    w(
        "\n**Beklenen ilk seans kontrolü** — her sembol için: halka arz/kuruluş tarihi, yoksa faaliyet kesimi, "
        "yoksa istenen başlangıç (2016). Veri daha geç başlıyorsa HATA.\n\n"
    )
    ipo = []
    for s, r in R.items():
        g = r.get("gecmis") or {}
        if "beklenen_ilk_seans" in g:
            ipo.append(
                {
                    "Sembol": s,
                    "Beklenen ilk seans": g["beklenen_ilk_seans"],
                    "Dayanak": g.get("beklenen_kaynak"),
                    "Veride ilk": g["ilk_tarih"],
                    "Fark (gün)": g["ilk_tarih_fark_gun"],
                    "Başta eksik seans": g.get("baslangicta_eksik_seans", 0),
                    "Yahoo firstTradeDate": g.get("yahoo_firstTradeDate", ""),
                }
            )
    w(_md_table(pd.DataFrame(ipo)))
    for s, r in R.items():
        if r.get("legacy"):
            L = r["legacy"]
            w(f"\n**{s} ← {L['eski_sembol']} (ticker değişimi {L['degisim_tarihi']})**\n\n")
            w(f"- {s}: {L.get('yeni_ilk')} → {L.get('yeni_son')} ({L.get('yeni_satir')} satır)\n")
            B0 = L.get("birlestirme") or {}
            w(
                f"- {L['eski_sembol']}: "
                + (
                    f"{B0.get('eski_ilk')} → {B0.get('eski_son')}"
                    if B0.get("eski_ilk")
                    else "Yahoo'da veri yok"
                )
                + "\n"
            )
            if "degisim_cevresi_max_abs_getiri" in L:
                w(
                    f"- Değişim ±7 gün içinde en büyük |getiri|: {L['degisim_cevresi_max_abs_getiri']:.2%} "
                    "(ticker değişimi fiyatta kırılma yaratmamalı)\n"
                )
            B = L.get("birlestirme") or {}
            w(
                f"- Onay (universe.json `legacy_merge_approved`): {B.get('onay')} · Karar: **{B.get('karar')}**\n"
            )
            if B.get("kontroller"):
                w(
                    f"- Doğrulama: örtüşen {B.get('ortusen_gun')} gün; close max fark {B.get('close_max_fark')}, "
                    f"adj max fark {B.get('adj_close_max_fark')}, ölçek oranı {B.get('close_medyan_oran')}; "
                    f"kontroller {B.get('kontroller')}\n"
                )
    w("\n**Manifesto notları**\n\n")
    for x in u["symbols"]:
        if x.get("event") and x["asset"] == "equity":
            w(f"- {x['symbol']}: {x['event']}\n")

    # ---------------------------------------------------------------- 4. takvim
    w("\n## 5. Takvim ve eksikler\n")
    w(
        "ABD (XNYS) seans takvimine göre, sembolün ilk–son (eligible) aralığında. Eksikler doldurulmadı. "
        "Endekslerde (^VIX, ^TNX) farklı yayın takvimi BİLGI düzeyinde.\n\n"
    )
    cal = []
    for s, r in R.items():
        t = r.get("takvim") or {}
        if t.get("eksik_seans") or t.get("fazla_tarih"):
            cal.append(
                {
                    "Sembol": s,
                    "Eksik": t.get("eksik_seans"),
                    "En uzun boşluk": t.get("en_uzun_bosluk_seans"),
                    "Örnek eksik": ", ".join(t.get("eksik_tarihler", [])[:8]),
                    "Fazla": t.get("fazla_tarih"),
                    "Örnek fazla": ", ".join(t.get("fazla_tarihler", [])[:5]),
                }
            )
    w(_md_table(pd.DataFrame(cal)) if cal else "Tüm sembollerde eksik/fazla seans yok.\n")

    # ---------------------------------------------------------------- 5. büyük hareketler
    w("\n## 6. Büyük hareketler (|kapanış getirisi| > %25) — silinmedi\n")
    mv = []
    for s, r in R.items():
        for e in (r.get("hareket") or {}).get("buyuk_hareket", []):
            mv.append({"Sembol": s, **e})
    mvdf = pd.DataFrame(mv)
    if len(mvdf):
        mvdf = mvdf.sort_values("ret", key=lambda x: -x.abs())
    w(_md_table(mvdf.head(40)))
    if len(mvdf) > 40:
        w(f"\n_… toplam {len(mvdf)} olay; tamamı `buyuk_hareketler.csv`._\n")
    sus = [(s, i) for s, r in R.items() for i in r["issues"] if i["code"] == "GIZLI_SPLIT"]
    for s, i in sus:
        w(f"- ⚠ {s}: {i['msg']}\n")

    # ---------------------------------------------------------------- 6. bağlam
    w("\n## 7. Bağlam serileri\n")
    for s, r in R.items():
        if r.get("endeks"):
            e = r["endeks"]
            w(
                f"- **{s}**: aralık {e['min']:.2f}–{e['max']:.2f}, son 1 yıl medyan {e['medyan_son252']:.2f}"
                + (f"; ölçek: {e.get('olcek_yorumu')}" if e.get("olcek_yorumu") else "")
                + (f"; ölçek sıçraması: {e['olcek_sicramasi'][:5]}" if e["olcek_sicramasi"] else "")
                + "\n"
            )
    w("\nBağlam ETF'lerinin kapsaması (ısınma = ilk 200 seans):\n\n")
    w(_md_table(cov["context"]))

    # ---------------------------------------------------------------- 7. bilanço
    w("\n## 8. Bilanço: sağlayıcı takvim kayıtları (Yahoo `get_earnings_dates`)\n")
    w(
        "Bunlar **sağlayıcı takvim kaydıdır**; gerçekleşme, ilan saati ve önceden duyurulma zamanı bu kaynakla doğrulanmaz. "
        "Sınıflar: `planlanan` (indirme anından sonra), `sonuç kaydı var` (geçmiş tarih + sağlayıcıda raporlanan EPS; "
        "yine doğrulanmamış), `durumu belirsiz` (geçmiş tarih, sonuç bilgisi yok — gerçekleşmiş sayılmaz). "
        '"Bilançoya kalan gün" özelliği, tarihsel duyuru zamanı doğrulanmadan kullanılmaz.\n\n'
    )
    er = []
    for s, r in R.items():
        b = r.get("bilanco")
        if b:
            er.append(
                {
                    "Sembol": s,
                    "Durum": b.get("durum"),
                    "Kayıt": b.get("kayit", ""),
                    "En eski": b.get("en_eski", ""),
                    "Sonuç kaydı var": b.get("sonuc_kaydi_var", ""),
                    "Durumu belirsiz": b.get("durumu_belirsiz", ""),
                    "Planlanan": b.get("planlanan", ""),
                    "~Beklenen çeyrek": b.get("beklenen_ceyrek_yaklasik", ""),
                    "Saat bilgisi": b.get("saat_bilgisi_var", ""),
                }
            )
    w(_md_table(pd.DataFrame(er)))

    # ---------------------------------------------------------------- 8. likidite
    w("\n## 9. Likidite ve oynaklık (bilgi; seçim için kullanılmadı)\n")
    lq = [{"Sembol": s, **r["likidite"]} for s, r in R.items() if r.get("likidite")]
    w(_md_table(pd.DataFrame(lq)))

    # ---------------------------------------------------------------- 9. kapsama
    W, MIN = u["warmup_sessions"], u["min_train_samples"]
    H = max(u["horizons"])
    w("\n## 10. Kapsama matrisi (hisse × katman) — ÖN SAYIM\n")
    w(
        "**Takvim bazlı ön sayımdır.** Etiket ve purge gerçek barlarla hesaplanmadan kesin eğitim uygunluğu veya katman başına uygun şirket sayısı ilan edilmez. `YETER` = ön sayımda ≥ eşik.\n\n"
    )
    w(
        f"Hücre: `durum eğitim_örneği(H={H}) / validation_sinyal_günü`. Isınma {W} seans, purge: etiket sonu "
        f"(t+{H}) training sonunu geçemez. Ön eşik ≥ {MIN} örnek. Sayım XNYS seans takvimi üzerindedir; t+1…t+{H} "
        "seanslarından biri eksik/bozuksa örnek düşer (`kapsama_uzun.csv` → `bosluk_kaybi_*`). "
        "TEST sütunu yalnız sinyal günü sayısıdır.\n\n"
    )
    w(_md_table(cov["pivot"], index=True))
    w("\n**Katman özeti**\n\n")
    w(_md_table(cov["summary"]))

    # ---------------------------------------------------------------- 10. tüm bulgular
    w("\n## 11. Tüm bulgular (sembol bazında)\n")
    w("HATA ve UYARI düzeyindekiler; BILGI satırları `bulgular.csv` içinde.\n")
    partial = [s for s, r in R.items() if any(i["code"] == "KISMI_GUN" for i in r["issues"])]
    if partial:
        w(f"\n_Seansı bitmemiş son gün {len(partial)} sembolde temiz katmandan çıkarıldı._\n")
    for s, r in R.items():
        imp = [i for i in r["issues"] if i["severity"] != "BILGI"]
        if imp:
            w(f"\n**{s}** ({r['durum']})\n")
            for i in imp:
                w(f"- [{i['severity']}] `{i['code']}` {i['msg']}\n")

    md = "".join(out)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "kalite_raporu.md").write_text(md, encoding="utf-8")
    summary.to_csv(paths.reports / "sembol_ozet.csv", index=False, encoding="utf-8-sig")
    if len(mvdf):
        mvdf.to_csv(paths.reports / "buyuk_hareketler.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"symbol": s, **i} for s, r in R.items() for i in r["issues"]]).to_csv(
        paths.reports / "bulgular.csv", index=False, encoding="utf-8-sig"
    )
    write_json(q, paths.reports / "kalite_detay.json")
    return md
