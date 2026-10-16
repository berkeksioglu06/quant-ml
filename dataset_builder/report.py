"""İnternet gerektirmeyen, okunabilir inceleme raporu."""

from html import escape

import pandas as pd

DEFINITIONS = [
    ("return_1/5/10/20", "1, 5, 10, 20 seanslık düzeltilmiş kapanış getirisi"),
    ("sma_distance_10/20/60", "Bugünkü fiyat / bugünü içeren hareketli ortalama − 1"),
    ("sma_slope_10/20/60_5", "Hareketli ortalamanın 5 seanslık oransal değişimi"),
    ("volatility_10/20", "Günlük log getirilerin örnek standart sapması × √252"),
    ("range_ratio", "(En yüksek − en düşük) / kapanış"),
    ("gap_return", "Düzeltilmiş açılış / önceki düzeltilmiş kapanış − 1"),
    ("intraday_return", "Kapanış / açılış − 1"),
    ("atr_14_ratio", "14 seanslık basit ortalama true range / kapanış; Wilder ATR değil"),
    (
        "relative_spy/qqq_5/20",
        "Hisse getirisi − aynı günlerdeki benchmark getirisi; beta düzeltmesi değil",
    ),
    ("market_spy/qqq_return_20", "SPY ve QQQ'nun 20 seanslık getirisi"),
    ("relative_volume_20", "Sağlayıcı hacmi / bugünden önceki 20 seansın ortalama hacmi"),
    ("volume_change_1", "Sağlayıcı hacmi / önceki seans hacmi − 1"),
]


def write_review(output, manifest, coverage, folds, latest):
    title = "Quant ML · Veri üreticisi"
    info = pd.DataFrame(DEFINITIONS, columns=["Özellik grubu", "Hesap"])
    feature_count = len(manifest["feature_columns"])
    costs = "Belirtilmedi; net getiri boş" if manifest["costs"] is None else str(manifest["costs"])
    classes = (
        "Seçilmedi; sürekli getiri hedefleri üretildi"
        if manifest["neutral_threshold"] is None
        else str(manifest["neutral_threshold"])
    )
    preview_cols = [
        "symbol",
        "return_5",
        "return_20",
        "volatility_20",
        "relative_volume_20",
        "quality_status",
    ]
    preview = latest[preview_cols].copy()
    for name in ("return_5", "return_20", "volatility_20"):
        preview[name] = preview[name].map(lambda v: "—" if pd.isna(v) else f"%{v * 100:.2f}")
    preview["relative_volume_20"] = preview.relative_volume_20.map(
        lambda v: "—" if pd.isna(v) else f"{v:.2f}×"
    )
    skipped = pd.DataFrame(manifest["skipped_symbols"], columns=["symbol", "reason"])
    global_issues = pd.DataFrame(
        manifest["global_quality_issues"], columns=["symbol", "severity", "code", "msg"]
    )
    skipped_html = (
        skipped.to_html(index=False, border=0, escape=True)
        if len(skipped)
        else "<p>Atlanan hisse yok.</p>"
    )
    issues_html = (
        global_issues.to_html(index=False, border=0, escape=True)
        if len(global_issues)
        else "<p>Ayrı genel uyarı kaydı yok.</p>"
    )
    body = f"""<!doctype html>
<html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f9;color:#1a293c;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1150px;margin:auto;padding:42px 24px}}h1{{font-size:36px;margin:10px 0}}h2{{font-size:23px;margin-top:34px}}
.eyebrow{{color:#50657f;font-size:13px;letter-spacing:1px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}
.card,section{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px}}.card strong{{display:block;font-size:27px}}
.notice{{border-left:4px solid #bc8a24;background:#fff8e9;padding:16px 20px;border-radius:6px}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e6ebf1;white-space:nowrap}}th{{background:#eff3f8}}
section{{margin:16px 0}}.muted{{color:#576a80}}code{{font-size:13px;background:#eef2f6;padding:3px 5px}}a{{color:#175e9c}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:28px}}main{{padding:22px 12px}}}}
</style><main><div class="eyebrow">YEREL İNCELEME SÜRÜMÜ · 1.1</div>
<h1>{title}</h1><p class="muted">{escape(manifest["snapshot"])} · Kaynak düzeni: {escape(manifest["layout"])}</p>
<div class="cards"><div class="card"><strong>{manifest["symbols"]}</strong>hisse</div>
<div class="card"><strong>{manifest["rows"]:,}</strong>hisse/gün satırı</div>
<div class="card"><strong>{feature_count}</strong>özellik</div>
<div class="card"><strong>{manifest["last_date"]}</strong>son veri günü</div></div>
<div class="notice"><strong>İncelemeye hazır; model eğitimi yapılmadı.</strong>
Eksik değerler doldurulmadı. Kalite uyarısı bulunan semboller ayrıca incelenmeli.
Özellik adaylığı ve testlerin geçmesi, eğitim verisinin nihai onayı anlamına gelmez.</div>
<h2>Ne üretiyor?</h2><section>
<p>Gün kapanışında bilinen fiyat, trend, volatilite, piyasa karşılaştırması ve uygun dönemlerde hacim özellikleri.</p>
<p>Hedef: ertesi seans açılışından giriş, sinyalden {escape(str(list(manifest["horizons"])))} seans sonraki kapanıştan çıkış.
Son günlerin geleceği bilinmediği için etiketleri boş kalır.</p>
<p><b>Maliyetler:</b> {escape(costs)}<br><b>Sınıf eşiği:</b> {escape(classes)}</p>
<p>Model girdileri yalnız <code>manifest.json → feature_columns</code> listesidir.
Hedefler, çıkış tarihleri ve durum kolonları girdiye katılmaz.</p></section>
<h2>Tarih kesiminde atlanan hisseler</h2><section class="scroll">{skipped_html}</section>
<h2>Genel kaynak uyarıları</h2><section class="scroll">{issues_html}</section>
<h2>Son gün · örnek özellikler</h2><section class="scroll">{preview.to_html(index=False, border=0, escape=True)}</section>
<h2>Hisse bazında kapsama</h2><p class="muted">price_ready: 22 fiyat/piyasa özelliği. volume_ready: 2 hacim özelliği.
200 önceki geçerli seans ve evren/kalite koşulları uygulanınca price_candidates oluşur.</p>
<section class="scroll">{coverage.to_html(index=False, border=0, escape=True)}</section>
<h2>Kronolojik eğitim ve doğrulama sayımları</h2><p class="muted">price_only: 22 özellik; all_24: hacim dahil 24 özellik.
Etiketi dönem sonunu aşan eğitim örnekleri purged_train sayısına girer. UYARI kayıtları sonradan ayrıca incelenmelidir.</p>
<section class="scroll">{folds.to_html(index=False, border=0, escape=True)}</section>
<h2>Formül sözlüğü</h2><section class="scroll">{info.to_html(index=False, border=0, escape=True)}</section>
<h2>Kontroller ve sınırlar</h2><section><ul>
<li>{manifest["numeric_checks_passed"]} bağımsız etiket hesabı karşılaştırıldı.</li>
<li>{manifest["partition_checks"]} dönem/ufuk için etiket sınırı uygulandı.</li>
<li>Kaynak dosyalar önce ve sonra dosya özetleriyle kontrol edildi; kaynaklara yazılmaz.</li>
<li>Takvim XNYS; tatiller ve erken kapanışlar kullanılır. Aynı gün kapanışında işlem varsayılmaz.</li>
<li>Hacim oranları sağlayıcının volume kolonundan hesaplanır; pencere boyunca tutarlı split ölçeği varsayılır. Split kaydı tek başına pencereyi sıfırlamaz. Gerçek tarihsel hisse adedi doğrulanmış sayılmaz.</li>
<li>Düzeltilmiş fiyatlar geçmiş veri revizyonlarına karşı point-in-time veri garantisi değildir.</li>
<li>Long net getiri sabit oransal maliyet modelidir. Short, ödünç alma, bariyer veya portföy simülasyonu yoktur.</li>
<li>2025+ son test döneminde ufuk veya eşik seçilmedi; hiçbir model eğitilmedi.</li></ul></section>
<h2>Dosyalar</h2><section><ul>
<li><a href="latest.csv">Son gün CSV</a> · <a href="sample_200.csv">200 satırlık örnek CSV</a></li>
<li><a href="coverage.csv">Kapsama</a> · <a href="fold_counts.csv">Dönem sayımları</a> · <a href="missingness.csv">Boş değer oranları</a></li>
<li><a href="manifest.json">Kaynak/çıktı kayıtları ve girdi listesi</a></li>
<li>dataset.parquet: birleşik veri; features.parquet: ileri bilgi içermeyen özellik tablosu.</li>
<li>partition_membership.parquet: dönem, ufuk, hisse, tarih eşleşmeleri.</li></ul></section>
<p class="muted">İnternet bağlantısı gerekmez. Bu rapor yalnız birlikte üretildiği dosyalarla kullanılmalıdır.</p></main></html>"""
    (output / "INCELEME.html").write_text(body, encoding="utf-8")
    lines = [
        "# Quant ML veri üreticisi · inceleme",
        "",
        f"{manifest['rows']:,} satır · {manifest['symbols']} hisse · {feature_count} özellik.",
        f"Son veri günü: {manifest['last_date']}. Kaynak: {manifest['layout']}.",
        "",
        "Ayrıntılı inceleme için INCELEME.html dosyasını açın.",
        "",
        f"Maliyet: {costs}. Sınıflandırma: {classes}.",
        "",
        "Hedefler ve ileri tarih bilgileri model girdisine alınmaz. Etiketler eksikse doldurulmaz.",
        "Bu çıktı model eğitimi veya eğitim verisi kabulü değildir.",
    ]
    (output / "OZET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
