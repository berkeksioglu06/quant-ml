# Quant ML — veri etabı (Etap 2: indirme ve kalite)

Claude tarafından hazırlanan `quantml_data.zip` paketinin repo düzenine uyarlanmış sürümüdür.
Kaynak ZIP SHA-256: `bfcfd25384180433d3887f1065017b205c5752ab60d2adbea7d3863fa19bf5d3`.
Kod `src/quant_ml/data/`, ayarlar `config/data/`, sentetik testler `tests/test_data_pipeline.py` altındadır.

Veri kapsamı planındaki evrenin tamamını (20 ana + 8 kontrol + NBIS = 29 şirket, 18 bağlam serisi; kripto kapsam dışı) 2016-01-01'den bugüne
Yahoo Finance'ten indirir, ham kopyayı değiştirmeden saklar, ayrı bir temiz katman üretir ve kalite raporu ile
hisse × katman kapsama matrisini çıkarır.

## Çalıştırma

```bash
uv sync
uv run pytest tests/test_data_pipeline.py      # ağsız sentetik test; pytest başarılı sonucu gösterir
uv run python -m quant_ml.data all                   # gerçek indirme + rapor (~2-4 dk)
```

Rapor: `data/reports/<snapshot>/kalite_raporu.md`

Sonradan yalnız raporu yeniden üretmek (indirmeden): `uv run python -m quant_ml.data build`
Tek sembol denemesi: `uv run python -m quant_ml.data download --symbols RKLB SPY` ve ardından `uv run python -m quant_ml.data build`
Bu sınırlı indirmede, indirilmeyen evren üyeleri raporda eksik görünür; tam kapsama için `all` kullanılır.
Rate limit yersen: `uv run python -m quant_ml.data all --pause 3`

## Çıktılar

| Yol | İçerik |
|---|---|
| `data/raw/<snapshot>/prices/*.parquet` | yfinance'in döndürdüğü çerçeve, olduğu gibi. Salt-okunur. |
| `data/raw/<snapshot>/earnings/*.parquet` | Yahoo bilanço tarihleri (yalnız kaynak denetimi için) |
| `data/raw/<snapshot>/manifest.json` | indirme zamanı, seçenekler, paket sürümleri, her sembolün ilk/son tarihi, satır sayısı, sha256 |
| `data/clean/<snapshot>/*.parquet` | temiz katman (aşağıda) + `clean_log.json` |
| `data/reports/<snapshot>/kalite_raporu.md` | ana rapor |
| `.../kapsama_matrisi.csv`, `kapsama_uzun.csv`, `kapsama_ozet.csv` | hisse × katman kapsaması |
| `.../sembol_ozet.csv`, `bulgular.csv`, `buyuk_hareketler.csv`, `kalite_detay.json` | ekler |

## Kararlar (koda gömülü)

* **auto_adjust=False, repair=False, back_adjust=False.** Bu seçenek gerçek tarihsel fiyat garantisi vermez.
  İkinci kez split düzeltmesi yapılmaz.
* **Temiz katman kolonları:** `open high low close adj_close volume dividends splits capital_gains`
  + `raw_close`, `raw_volume`, `split_factor_price`, `split_factor_volume`,
  `eligible` (SPAC / yeniden yapılanma kesiminden sonra mı), `is_session`, `source_symbol`.
* **Split yöntemi = dış referans:** split günü fiyat oranı ve hacim oranı yalnız *inceleme işaretidir*.
  Sağlayıcının yöntemi (fiyat ve hacim ayrı) `config/data/reference_points.csv`'ye **bağımsız kaynaktan** (borsa kaydı,
  şirket duyurusu, haber arşivi — Yahoo değil) girilen gerçek kapanış/hacim değerleriyle belirlenir: en az 3 dolu nokta,
  en az 2 sembol, hepsi aynı sonuç. Doğrulanmadıkça `raw_close` / `raw_volume` son split öncesinde NaN kalır ve ≥5$
  filtresi o dönemde uygulanamaz. Split'siz hisselerde ve son split sonrasında sağlayıcı değeri = tarihsel değer.
  Doğrulanmış yöntemle çelişen fiyat işareti o split'in öncesini NaN yapar (HATA).
* **Beklenen ilk seans:** her sembol için halka arz/kuruluş tarihi, yoksa faaliyet kesimi, yoksa 2016 başı; veri
  daha geç başlıyorsa HATA ve baştaki eksik seans sayısı.
* **Bilanço:** Yahoo kayıtları *sağlayıcı takvim kaydı* olarak raporlanır (`planlanan` / `sonuç kaydı var` /
  `durumu belirsiz`); gerçekleşme ve duyuru zamanı bu kaynakla doğrulanmaz.
* **Checksum:** fiyat ve bilanço dosyalarının sha256'sı manifest'te; temizleme başlamadan hepsi yeniden doğrulanır,
  uyuşmazlıkta durur.
* **Seans tamamlanma:** son satır, o günün gerçek kapanışı (erken kapanış dahil) + 15 dk ile, sembolün kendi
  indirme zamanına göre kontrol edilir.
* **Hiçbir şey silinmez veya doldurulmaz.** SPAC öncesi satırlar `eligible=False`; büyük sıçramalar yalnız raporlanır;
  eksik seanslar doldurulmaz. Yalnızca (a) birebir tekrar satırlar tekilleştirilir, (b) indirme anında seansı
  bitmemiş son gün çıkarılır; ikisi de loglanır.
* **SQ → XYZ:** otomatik birleştirme yok. Doğrulama (≥20 örtüşen seans; close, adj close, ölçek ve split kayıtları
  %0.5 içinde) her zaman raporlanır. Birleştirme yalnız `config/data/universe.json`'da `legacy_merge_approved: true`
  VE doğrulama geçerse yapılır. Gerekli olup birleşmediyse HATA.
* **Kapsama sayımı (ön sayım, etiket hesaplanınca kesinleşir):** 200 seans ısınma, purge (t+H ≤ training sonu), H=20 için ≥252 örnek → `YETER`.
  Sayım seans takvimi üzerinde; t+1…t+H'de eksik/bozuk bar varsa örnek düşer. TEST için yalnız sinyal günü sayılır.
* **Test dönemi:** kalite kontrolleri 2025–2026'yı da inceler (veri kalitesi için gerekli). Kural: bu bilgi model,
  özellik, eşik veya evren seçiminde kullanılmaz.

## Senin doldurman gereken: `config/data/reference_points.csv`

Dosyada önerilen satırlar (split'ten önceki son seanslar) boş duruyor. Her biri için o gün **gerçekte görülen**
kapanış ve hacmi bağımsız bir kaynaktan gir, `kaynak` sütununa nereden aldığını yaz. Doldurduktan sonra
`uv run python -m quant_ml.data build` raporu yeniden üretir; yeni indirme gerekmez.

## Elle girilmiş tarihler — teyit edilmeli

`config/data/universe.json` içindeki SPAC kesimleri (RKLB 2021-08-25, HIMS 2021-01-21, SOFI 2021-06-01),
NBIS yeniden başlangıcı (2024-10-21), SQ→XYZ (2025-01-21), halka arz tarihleri ve beklenen split'ler
paketi hazırlayan kişi tarafından girildi. Script bunları veriyle karşılaştırıp uyuşmazlığı raporlar ama resmi kaynaktan
(şirket basın bülteni / borsa duyurusu) bir kez teyit edilmeleri gerekir.

## Ciddiyet

* **HATA:** eğitimden önce çözülmeli (farklı değerli tekrar tarih, ham split, beklenen split'in olmaması, ≤0 fiyat, uzun veri boşluğu …)
* **UYARI:** incelenmeli (birkaç eksik seans, High/Low ihlali, temettü faktörü uyuşmazlığı, kayıtsız split'e benzeyen hareket …)
* **BILGI:** not (büyük hareket, SPAC öncesi satır, kısmi bilanço geçmişi …)
