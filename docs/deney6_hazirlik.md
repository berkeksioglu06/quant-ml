# Geniş evren (2.957 hisse) → Deney 6 hazırlığı

Amaç: Kural tabanlı geniş evreni repodaki kodla tekrar üretilebilir hale getirmek, deney koduna
(`research/`) bellek dostu şekilde bağlamak ve geniş evrendeki ilk deneyi (Deney 6) **önceden yazılmış**
haliyle koşmak. 2025+ kapalı kalır. Sonuçlar görüldükten sonra hiçbir ayar değişmez. Veri Git'e girmez.

Adlandırma: Deney 1–4b 28 hisselik evrende, Deney 2/4a tekrarı 281 hisselik evrende koşuldu.
Deney 5 üç bariyerli hedefe ayrılmıştı. Bu deney Deney 6'dır.

## 0. Bağlam ve alınmış kararlar

- **Evren kodu:** `genis_evren.py` (`genis-evren` dalı, senin dalına merge edildi). Aday havuzu bugünkü
  Nasdaq sembol dizinindeki ~5.600 adi hisse (ETF, warrant, unit, tercihli, tahvil ve fonlar hariç).
- **Kural** (sabit; her gün, YALNIZ o güne kadarki veriyle):
  - split'e göre düzeltilmemiş gerçek fiyat > 5$
  - son 60 seans medyan dolar hacmi > 10 M$
  - en az 252 seans geçmiş
- **Oynaklık bilinçli olarak kuralda yok.** Evren büyüklüğü rejimle sallanırdı ve evren, etkisi işaret
  değiştiren değişkene göre seçilmiş olurdu (Deney 1–4 bulgusu).
- **Sonuç** (snapshot `genis_yahoo_20261016T191133Z`, son seans 2026-10-15): 5.599 indirildi
  (CTY boş). 2.957 hisse en az bir gün evrende, bugün 2.278. Günlük evren 2016'da ~1.100,
  2024'te ~1.800. Günlük giriş-çıkış ~%0,4.
- **Bilinen sınırlar:**
  - Survivorship: batmış ve satın alınmış hisseler yok (ör. CFLT, CYBR, EA, EXAS, LC, PSTG, QRVO,
    ZI bugünkü dizinde bulunmuyor).
  - SPAC kabuk dönemi 252 seanslık geçmiş sayımına giriyor (HIMS, ASTS, VRT gibi hisseler birleşmeden
    hemen sonra evrene giriyor). Bu sürümde **düzeltilmeyecek**, raporda not edilecek.
- **281 hisselik elle seçilmiş evrenle fark:** Seçimi kural yapıyor, örnek sayısı ~10 kat daha fazla.
  281 sonuçları (Deney 2: IC −0,061; 4a: −0,105) karşılaştırma için raporda referans olarak verilsin.

## 1. Veri üretimini repoya taşı (`genis_evren.py`'ye iki komut)

Şu an bu iki adım bende repo dışı script olarak duruyor. `tools/genis_evren_ref/ref_build_clean.py` ve
`ref_build_dataset.py` mantığı birebir koruyarak taşınsın:

- **`clean`:** `evrene_girenler` listesindeki her hisse için `market_data.to_daily` →
  `is_session` (XNYS), `eligible = True` (bütün gerçek işlem satırları), `raw_close`,
  kural ölçüleri ve `in_universe`. Ham dosyada checksum kontrolü yapılsın.
  Çıktı: `data/genis_<snapshot>/clean/*.parquet`.
- **`dataset`:** Her hisse için `dataset_builder.features.make_features` (v1.1, 24 özellik) ve
  `labels.make_labels` (5/10/20 seans).
  - Benchmark SPY/QQQ temiz tabloları sabit snapshot'tan (`data/yahoo_20261016T121656Z/clean`).
  - **Aday = `price_sample_candidate` (200 seans ısınma, 22 fiyat özelliği hazır) VE o gün `in_universe`.**
    Özellik pencereleri bütün satırlarla hesaplanır. Evrene girdiği gün geçmiş zaten hazır olur.
  - K1–K4 bölmeleri `partition_mask` ile, purge'lü.
  - Çıktı: `output/genis_<snapshot>_dataset/` (Git'e girmez): parça parça parquet, `manifest.json`
    (feature_columns, horizons, folds, test_start, snapshot, kural), `partition_membership.parquet`.
- Her ikisi de **kaldığı yerden devam edebilsin**. 4 süreçle paralel çalışabilir.

**Doğrulama** (benim sonuçlarımla birebir tutmalı):
- 2.957 hisse, hata 0.
- `tools/genis_evren_ref/beklenen_fold_counts.csv` ile aynı sayılar. Ör. K1/H20 eğitim 1.572.431 satır / 1.843 hisse,
  K4/H20 eğitim 2.825.678 / 2.383.
- `tools/genis_evren_ref/ref_crosscheck.py`: 28 hisselik eski veri setiyle 65.413 satırın hepsi eşleşiyor. En büyük fark
  ~1,5e-6. Bu fark temettü ödeyen hisselerde Yahoo yuvarlamasından geliyor. `range_ratio`,
  `intraday_return`, hacim ve piyasa özellikleri birebir aynı.

## 2. Deney koduna bağla: bellek dostu yükleyici

Sorun: Mevcut `load_panel` bölme kayıtlarıyla birleşim yaparken satırları fold başına kopyalıyor.
H20'de bu ~10,4 M satır × ~60 kolon demek, 5 GB'ı aşıyor. Makinede 16 GB RAM var, ~7–8 GB boş.

- **`WidePanel`** (aynı arayüz: `split(fold) → (train, val)`, `features`, `targets`, `folds`, `rows`
  benzeri erişim):
  - Tek bir taban tablo: tekil (date, symbol), yalnız `date < test_start`, yalnız en az bir bölmede
    geçen satırlar. Özellikler `float32`.
  - Bölmeler `partition_membership`'ten indeks olarak tutulur. `split` istendiğinde oluşturulur,
    satır kopyası tutulmaz.
  - Kesitsel `xs_*` özellikleri Deney 6'da **gerekmiyor**. Hesaplanmasın (bellek).
- **Beta:** 252 seans, en az 200 gözlem, yalnız t'ye kadar. Bütün hisseler için **geniş `clean/`
  klasöründen**. QQQ ve ^VIX sabit snapshot'tan. Mevcut `rolling_beta` / `daily_returns` kullanılsın.
- **Rejim** (QQQ ↑ / yatay / ↓, ±%2) ve 6 piyasa durumu özelliği mevcut fonksiyonlarla.
  Fiyatlar test başlangıcından önce kesilsin (`as_of`).
- **Oynaklığa göre ölçekli hedef (yeni):**
  `target_vs_20 = target_return_20 / (max(volatility_20, 0.10) × √(20/252))`.
  `volatility_20` t'de bilinen değer. 0,10 tabanı sabit (çok sakin hisselerde sonsuza gitmesin).
  Eğitimde %1/%99 kırpma aynı, değerlendirme kırpılmamış hedefle.
- Mevcut deney ve rapor fonksiyonları (`evaluate`, `pooled`, `_daily_diagnostics`) değişmeden
  çalışmalı. Gerekirse `daily_rank_ic` ve `feature_ic` hızlandırılsın, sonuçları aynı kalsın
  (test ile).

## 3. Deney 6: önceden yazılı (koşmadan önce `docs/deney6_plan.md` olarak commit'le)

- **Evren:** 2.957 kural tabanlı (aday satırlar yukarıdaki gibi). H = 20.
- **Hedef:** `target_vs_20` (oynaklığa göre ölçekli).
- **Model:** LightGBM, **4a ayarları aynen** (`LGBM_PARAMS`, seed, purge'lü iç early stopping, l2).
  - İş parçacığı: 4a'da `n_jobs=1`. Duman koşusunda tek fit 20 dakikayı aşarsa `n_jobs=4` +
    `deterministic=True` kullanılsın. Bu karar sonuçlara bakmadan, yalnız süreye göre verilsin ve
    belgeye yazılsın.
- **Kollar:**
  - **A (birincil):** 24 özellik. Değerlendirmede skor her gün `volatility_20`'ye karşı kesitsel
    olarak nötrlenir (`neutralize`). Ham skor da bilgi satırı olarak raporlanır.
  - **B (ikincil):** oynaklık ailesi çıkarılmış: `volatility_10`, `volatility_20`, `atr_14_ratio`,
    `range_ratio` hariç 20 özellik.
- **Kıyaslar** (aynı satırlar, aynı hedef): momentum_20, geri_donus_5, en_oynak_20, yuksek_beta,
  Ridge (aynı hedef, 24 özellik), sıfır ve eğitim ortalaması (hata için).
- **Ölçüler:**
  - `target_vs_20`'ye göre günlük rank IC, ham `target_return_20`'ye göre rank IC (bilgi)
  - NW t (19 gecikme), dönem bazında IC
  - Rejim ayrımı ve QQQ ↓ NW t
  - Skor–beta ve skor–volatility_20 sıra korelasyonu, ilk 5'in beta ve oynaklığı
  - Ağaç sayısı, importance
- **Alt kümeler** (aynı model, yalnız değerlendirme): bütün evren; mevcut 29 hisse; 263'lük aday
  listesi; 60 günlük medyan dolar hacmi terciline göre üç grup (o günkü değerle).
- **Başarı kriteri** (birincil kol A, `target_vs_20` IC'si):
  1. Birleşik rank IC > momentum_20 VE 4 dönemin en az 3'ünde pozitif
  2. QQQ ↓ rejiminde NW t > −2
  3. |skor–beta sıra korelasyonu| < 0,3
  - B kolu bilgi içindir. A geçmez B geçerse sonuç "karışık" olarak raporlanır, kazanan ilan edilmez.
- **Referans:** 281 hisselik evrenin Deney 2 ve 4a sonuçları ile eski 28 hisselik sonuçlar aynı
  tabloda.
- runs.csv'ye kayıt düşsün. Denenen hiçbir kombinasyon silinmesin.

## 4. Testler

- `clean` / `dataset`: küçük sentetik evrende aday = `price_sample_candidate & in_universe`.
  `in_universe` False olan gün aday değil ama özellik penceresi o günü kullanıyor.
- `WidePanel.split`, eski `load_panel` ile aynı küçük veri setinde birebir aynı train/val satırlarını
  ve değerlerini veriyor (28 hisselik veri setinde karşılaştırma testi).
- `target_vs_20`: tanım, 0,10 tabanı, yalnız t'de bilinen oynaklık.
- Beta sızma testi (t sonrası fiyat değişince beta_t aynı), geniş klasörden okuma.
- Hızlandırılan IC fonksiyonları eskiyle aynı sonucu veriyor.
- Mevcut testlerin hepsi geçmeye devam ediyor.

## 5. Duman koşusu (Deney 6'dan ÖNCE, sonuç okunmadan)

- 150 rastgele hisse (sabit seed) ile bütün zinciri bir kez çalıştır. Yalnız süreyi, belleği ve
  hatasız bittiğini raporla. **Metriklere bakma, yorumlama.** Duman çıktısı ayrı klasörde kalsın,
  runs.csv'ye "smoke" olarak işaretlensin.
- Tam Deney 6 için tahmini süre ve bellek yazılsın.

## 6. Çalıştırma sırası ve teslim

Bu belge bir Claude oturumuna doğrudan verilebilir; baştan sona şu sırayla ilerlesin:

1. Bölüm 1–2–4: kod ve testler. Bütün testler geçsin, doğrulama sayıları tutsun.
2. **Bölüm 3'ü olduğu gibi `docs/deney6_plan.md` olarak commit'le ve push'la. Deneyden ÖNCE.**
   Commit tarihi, planın sonuçtan önce yazıldığının kaydıdır.
3. Bölüm 5: duman koşusu (yalnız süre ve bellek).
4. Deney 6'yı koş. Kod, plan ile commit arasında değişmesin. Değişmesi gerekirse (hata) önce
   düzeltmeyi ve nedenini ayrı commit'le, sonra koş.
5. Sonuç raporunu `docs/deney6_sonuc.md` olarak yaz. Önceden yazılı kriter tablosu, kol A/B,
   kıyaslar, alt kümeler ve 281/28 hisselik referanslar olsun. Sonra push.
   Sonuca bakıp ayar değiştirme, ek kol ekleme.
6. Veri ve çıktılar Git'e girmez. runs.csv'ye kayıt düşsün.

Doğrulama: Push'tan sonra Berk'in tarafındaki Claude aynı commit'ten bağımsız olarak doğrulayacak
(sızma, tekrar üretilebilirlik, metriklerin bağımsız hesabı). Rapor ve kod buna izin verecek kadar
açık olsun.

Referans dosyalar (`tools/genis_evren_ref/`): `ref_build_clean.py`, `ref_build_dataset.py`, `ref_crosscheck.py`,
`beklenen_fold_counts.csv`.
Not: Ana klasörde (`C:\Users\user\Projects\quant-ml`) main üzerinde commit edilmemiş Deney 5 dosyaları
duruyor (`run_experiment5.py`, `docs/deney5_*.md`, `universe_candidates.json` ...). Onlar kimin, ne
olacak? Kaybolmasın.
