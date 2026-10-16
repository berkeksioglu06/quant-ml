# Aşama 1 — model deneyleri

Amaç: hazır veri setinde basit modellerin, basit kıyasları K1–K4 validation yıllarında
geçip geçmediğini görmek. **2025+ test dönemi bu aşamada okunmaz**; fiyatlar ve veri
seti satırları test başlangıcından önce kesilir. Model ve ayarlar dondurulduktan sonra
test dönemi yalnız bir kez açılır (Aşama 3).

```bash
uv run python train_baseline.py --deney 1
uv run python train_baseline.py --deney 2
uv run python train_baseline.py --deney 3
uv run python train_baseline.py --deney 4a
uv run python train_baseline.py --deney 4b
```

LightGBM macOS'ta OpenMP ister: `brew install libomp`. Linux ve Windows paketlerinde gerekmez.

Çıktı `output/experiments/<zaman>_<deney>_h20/RAPOR.md` ve yanındaki CSV/Parquet
dosyalarıdır. Her çalıştırma hedef başına bir satırla `output/experiments/runs.csv`
kaydına eklenir; denenen hiçbir kombinasyon silinmez. Bu klasör Git takibinde değildir.

## Ortak çerçeve

| Seçim | Değer |
|---|---|
| Ufuk | 20 seans (t+1 açılış → t+20 kapanış) |
| Satırlar | `partition_membership.parquet` (purge'lü) ve `volume_features_ready` |
| Hata kıyasları | Sıfır, eğitim ortalaması |
| Sıralama kıyasları | Momentum (`return_20`), geri dönüş (−`return_5`) |
| Risk kıyasları | En oynak (`volatility_20`), yüksek beta (`beta_252`); hükme girmez |
| Model | Ridge (Deney 1–3): %1/%99 kırpma, standartlaştırma, α eğitimin son 252 seansında seçilir; LightGBM (Deney 4a–4b) |
| Ölçüler | MSE ve R²_oos; günlük Spearman rank IC, Newey-West t (19 gecikme); dönem tutarlılığı |
| Tanılar | QQQ ↑/yatay/↓ rejim IC'si (±%2), beta-düzeltmeli IC, ilk 5 hissenin beta/volatilitesi |

Beta: QQQ'ya karşı günlük getirilerle son 252 seans, en az 200 ortak gözlem, yalnız t'ye
kadarki veriyle; eligible olmayan günler hesaba girmez. (Deney 1–2'nin ilk raporları 252
gözlemin tamamını istiyordu; Ridge sonuçları bundan etkilenmez.) Rejim, etiket penceresinde
gerçekleşen QQQ getirisine dayanır; yalnız değerlendirmeyi bölmek içindir, sinyal değildir.

Hüküm kuralı: K1–K4 birleşik ölçüde en iyi kıyastan iyi ve en az 3/4 dönemde iki kıyası
da geçiyorsa *geçiyor*; birleşik ölçüde kötü ve en fazla 1 dönemde geçiyorsa *geçmiyor*;
arası *karışık*.

## Deney 1 — ham özellikler

- Hedefler: `target_return_20` ve `target_vs_qqq_20` = hisse − aynı pencerede QQQ.
- Girdiler: manifestteki 24 `feature_columns`.
- Sonuç: Ridge iki hedefte de geçmiyor. K1–K4 rank IC −0,081 (NW t −2,17), R²_oos −3,1%;
  momentum +0,027. Skor pratikte "en oynak hisseyi seç": volatilite/ATR katsayıları 4/4
  dönemde pozitif.

## Deney 2 — gün içi hedef ve sıralanmış özellikler

- Hedef: `target_xs_20` = hisse getirisi − aynı gün aynı bölmedeki hisselerin ortalaması.
- Girdiler: aynı gün bütün hisselerde aynı olan `market_*` çıkarıldı; kalan 22 özellik her
  gün hisseler arasında sıraya çevrildi (−0,5…+0,5).
- Referans satırı: Deney 1 Ridge'i aynı tablolarda.
- Sonuç: Ridge geçmiyor. K1–K4 rank IC −0,079 (NW t −2,07); QQQ ↓ dönemlerinde −0,34,
  beta-düzeltmeli IC −0,087. İlk 5 hissenin betası 1,58, volatilitesi 0,82
  (evren 1,33 / 0,54). Skor hâlâ volatiliteye bağlı (korelasyon 0,75–0,95).
- Not: Günlük rank IC zaten kesitseldir; hedefi gün ortalamasından arındırmak sıralamayı
  değiştirmez. Volatilite ve betanın beta-düzeltmeli IC'si eğitimde sıfıra yakın/pozitif,
  validation yıllarında belirgin negatif. Yani dönemler arası fark yalnız piyasa yönü değil.

## Deney 3 — risk-nötr sıralama (betadan arındırılmış hedef)

- Deney 2'ye göre tek değişiklik hedefte: `target_resid_20` = hisse getirisi − beta_t ×
  aynı pencerede ve aynı kuralla QQQ getirisi, ardından gün içi arındırılmış.
- Evren: yalnız beta_t'si olan satırlar. 200 seanslık ısınma süresi nedeniyle dışarıda
  kalan satır yok; satırlar Deney 2 ile aynı.
- Ek satır: Ridge skoru her gün beta_t'ye kesitsel OLS ile çekilip kalıntısı alınır
  (yalnız değerlendirme).
- Önceden yazılı başarı kriteri (karar Ridge satırına göre):
  1. Artık IC birleşik ölçüde momentumdan yüksek ve en az 3/4 dönemde pozitif.
  2. Skor–beta günlük sıra korelasyonu −0.20 < r < 0.20 ve ilk 5'in betası evren
     ortalamasının ±0.20 içinde.
  3. QQQ ↓ rejiminde IC belirgin negatif değil: günlük IC'lerin Newey-West t > −2.
- Sonuç: **karşılanmadı**. Ridge artık IC −0,081 (NW t −2,47), 0/4 dönem pozitif;
  skor–beta korelasyonu +0,41, ilk 5 beta 1,54 (evren 1,33); QQQ ↓ IC −0,178 (t −3,19).
  Beta-nötrlenmiş skor risk kriterlerini geçiyor (r +0,03, ilk 5 beta 1,37) ama IC hâlâ
  negatif (−0,042, t −1,36) ve skor–volatilite korelasyonu +0,43. Momentum artık IC +0,023
  (t 0,90); QQQ ↓ rejiminde onun da t'si −2,22.

## Deney 4a — LightGBM, hissenin kendi getirisi

- Deney 1 ile tek fark model: hedef `target_return_20`, manifestteki 24 ham özellik
  (market_* dahil), aynı satırlar; eğitim hedefi %1/%99 kırpılır.
- Sabit ve sığ ayarlar, validation'a bakılarak değiştirilmez: max_depth 3, num_leaves 7,
  learning_rate 0.03, min_child_samples 200, subsample 0.8 (bagging_freq 1),
  colsample_bytree 0.8, reg_lambda 1.0, en fazla 2000 ağaç.
- Ağaç sayısı: early stopping (100 tur) eğitimin son 252 seansında, purge'lü iç bölmeyle;
  sonra bulunan sayıyla bütün eğitime yeniden fit. Seed sabit, deterministic, sütun bazlı
  histogram, tek iş parçacığı (çok iş parçacığında metrik toplamı oynuyor ve düz eğride
  early stopping farklı tur seçebiliyordu).
- Ek tanılar: fold başına ilk 10 kazanç payı, market_* kullanımı, `volatility_20`'nin
  kısmi bağımlılığı `market_qqq_return_20` > 0 ve < 0 satırlarında ayrı.
- Önceden yazılı başarı kriteri:
  1. Birleşik rank IC > momentum_20 ve en az 3/4 dönemde pozitif.
  2. QQQ ↓ rejiminde IC'nin NW t'si > −2.
  3. Skor–beta günlük sıra korelasyonu −0.30 < r < 0.30.
- Sonuç: **karşılanmadı** (yalnız 3. kriter geçti). Rank IC −0,050 (NW t −1,48), 2/4
  dönem pozitif (K3 +0,005, K4 +0,018), R²_oos −2,9%; QQQ ↓ IC −0,257 (t −5,42);
  skor–beta +0,295, skor–volatilite +0,46. Early stopping K1'de 124 ağaç, K2–K4'te 1 ağaç
  seçti: iç validation'da ilk ağaçtan sonra iyileşme yok. K2–K4'te kazancın %59–87'si
  aynı gün bütün hisselerde aynı olan `market_spy_return_20`'den; `market_qqq_return_20`
  yalnız K1'de (%3). Kısmi bağımlılık: K1'de volatilite etkisi iki piyasa durumunda da
  pozitif (koşullu değil), K2–K4'te düz. Model koşullu bir ilişki kurmadı.
- Plan değişikliği: Deney 3'ün önceden yazılı kuralı sıradaki adım olarak uzun momentumu
  gösteriyordu. Plan 4a (LightGBM) → 4b (piyasa durumu özellikleri) → Deney 5 (üç bariyerli
  hedef) olarak güncellendi; uzun momentum şimdilik planda yok.

## Deney 4b — LightGBM + piyasa durumu özellikleri

- 4a ile tek fark 6 yeni özellik (aynı gün bütün hisselerde aynı, sıraya çevrilmez):
  `market_qqq_return_5`, `market_qqq_return_60`, `market_qqq_sma50_dist`,
  `market_qqq_sma200_dist`, `market_vix_close`, `market_vix_change_5`. QQQ seans takvimi
  (XNYS), ^VIX bu takvime hizalanır (XNYS dışı günler düşer), tam pencere, eksik seans NaN,
  yalnız t kapanışına kadarki veri, doldurma yok. Hesap `dataset_builder.features` ile aynı.
- Kapsam: bütün uygun satırlarda 6 özellik de mevcut; çıkarılan satır 0. Satır kümesi 4a
  ile aynı. 4a ayarlı LightGBM aynı koşuda "lightgbm_4a" referansı olarak eğitilir ve
  Deney 4a'nın tahminlerini birebir üretir.
- Önceden yazılı kriter: 4a ile aynı.
- Sonuç: **karşılanmadı** (yalnız 3. kriter geçti). Rank IC −0,023 (NW t −0,56), 1/4 dönem
  pozitif (K3 +0,068); R²_oos −2,1% (4a −2,9%); QQQ ↓ IC −0,216 (t −2,90); skor–beta +0,291,
  skor–volatilite +0,39. K4'te model 232 günün 231'inde bütün hisselere aynı skoru verdi;
  K4 IC'si tek güne dayanıyor. Sabit günler 0 sayılırsa K1–K4 IC −0,016 (bilgi; kural aynı).
- Ağaç sayısı 4a ile aynı örüntü: K1 126, K2–K4 1. Yeni özelliklerin kazanç payı %48–79,
  büyük kısmı `market_vix_close` (aynı gün herkeste aynı; sıralamaya değil zamanlamaya
  katkı). Kısmi bağımlılık: K1'de volatilite etkisi iki piyasa durumunda da pozitif; K2–K4'te
  düz; SMA200 altı alt kümesi K1 ve K4'te boş. Model koşullu bir ilişki kurmadı.
- 4a early stopping incelemesi (madde 0): K2–K4'te seçilen ağaç sayısı 1 doğrulandı. İç
  rank IC K2'de (−0,05…−0,10) ve K3'te (−0,06…−0,17) bütün ağaç sayılarında negatif; K4'te
  küçük pozitif (+0,02…+0,04) ama l2 ağaç ekledikçe kötüleşiyor; K1'de pozitif
  (+0,07…+0,16). Early stopping'in ölçüsü l2 ile sıralama ölçüsü IC aynı yönü göstermiyor.
- Rapor eki: IC'nin hesaplanabildiği gün sayısı ve sabit günleri 0 sayan bilgi ortalaması
  bütün deney raporlarında gösterilir.

Kod: `research/data.py` (hedefler, beta, rejim, bölmeler), `research/models.py`,
`research/metrics.py`, `research/experiment.py`, `research/report.py`.

## Tanı — LightGBM neden öğrenmiyor?

`uv run python -m research.diag_lgbm` (K1–K4, test kapalı; `--jobs 4` ile ~1,5 dk). Deney 4a
ayarlarıyla early stopping'siz sabit ağaç sayılarında eğitim/validation eğrisi, etiket
karıştırma, 12 ayarlık küçük ızgara (yalnız bilgi; seçim yapılmaz), l2 ve IC ile early
stopping, özellik kümesi ablasyonu. Çıktı `output/experiments/<zaman>_diag_lgbm_h20/TANI.md`.

İlk koşu (`20261016T184204Z_diag_lgbm_h20`) bulguları:

- **Az epoch/ağaç değil.** Ağaç 1→1000: eğitim IC +0,07→+0,27, validation IC −0,04→−0,03,
  validation R² −2%→−15%. Ağaç eklemek yalnız eğitime uyumu artırıyor.
- **Ayar değil.** min_child {20,200,1000} × yaprak {7,31} × lr {0,03,0,1}, 300 ağaç: raw24'te
  val IC −0,011…−0,037, xs22'de −0,036…−0,083; 24 ayarın hiçbiri momentumu (+0,027) geçmiyor.
  Eğitim IC +0,65'e çıkıyor (ezber).
- **Eğitimdeki uyum büyük ölçüde gürültü.** Hedef karıştırılınca 200 ağaçta eğitim IC aynı
  kalıyor (K1 gerçek +0,18, karışık +0,19). Gerçek hedefteki ek eğitim R²'si gün bazlı
  özelliklerle dönem ezberinden geliyor: yalnız 8 `market_*` özelliğiyle eğitim R² +18%,
  sıralama IC'si tanım gereği 0, validation R² −11%.
- **Negatif IC'nin kaynağı işaret dönmesi.** Volatilite ailesi (`atr_14_ratio`, `volatility_10/20`,
  `range_ratio`) eğitim yıllarında IC +0,04…+0,05, validation yıllarında −0,09…−0,12. Model
  eğitimde "oynak hisse kazanır"ı öğreniyor; 2021–22'de tersi oluyor. K3–K4'te eğitim bu
  yılları içerince volatilite etkisi eğitimde sönüyor ve val IC sıfıra yaklaşıyor.
- **Early stopping'in 1 ağaç seçmesi hata değil, dürüst sonuç.** İç bölmede l2 ağaç eklendikçe
  kötüleşiyor; IC ile seçmek de validation'da kazandırmıyor (raw24 K2: l2→1 ağaç −0,090,
  IC→150 ağaç −0,117).
- Tavan: tek özelliklerin validation IC'si en çok +0,03 (`sma_distance_20`, `return_10`);
  günde 19–28 hisse, 20 seanslık örtüşen etiket. Bu kesitte ağaç modelinin momentumun
  üstüne koyacağı sinyal görünmüyor.

## Geniş evren

Tanının önerisiyle evren 29'dan 281 şirkete çıkarıldı (`universe_genis.json`, snapshot
`yahoo_20261016T185837Z`, veri seti `output/yahoo_20261016T185837Z_dataset_v1_1`). `train_baseline.py` ve
`research.diag_lgbm` artık varsayılan olarak bu veri setini okur; eski veri seti `--dataset`
ile verilebilir. Deney 1–4b sonuçları yukarıda 29 hisselik evrene aittir; Deney 2, 4a ve tanı
geniş evrende yeniden koşuldu (aşağıda). 2025+ test dönemi yine kapalıdır.

### Geniş evrende Deney 2, 4a ve tanı

Koşular: Deney 2 `20261016T191028Z_ridge_xs_h20`, Deney 4a `20261016T191113Z_lgbm_h20`, tanı
`20261016T192000Z_diag_lgbm_h20` (`--jobs 4`, 6 dk). Kriterler, ayarlar ve satır kuralları değişmedi;
validation'a bakılarak seçim yapılmadı. Kesit K1 validation'da günde 211 hisse (eski 23), K4'te
273 (28); evren betası 1,22 (eski 1,33), volatilitesi 0,54 (aynı). Eski sütun 29 hisselik koşular.

**Deney 2 — Ridge, sıralı özellikler (hüküm: geçmiyor → geçmiyor)**

| Ölçü | 29 hisse | 281 hisse |
|---|---:|---:|
| Ridge K1–K4 rank IC (NW t) | −0,079 (−2,07) | −0,061 (−2,54) |
| Momentum / geri dönüş IC | +0,027 / −0,021 | +0,001 / +0,000 |
| Pozitif IC'li dönem | 0/4 | 0/4 |
| QQQ ↓ IC | −0,343 | −0,223 (t −9,06) |
| Beta-düzeltmeli IC | −0,087 | −0,067 |
| İlk 5 beta / vol. (evren) | 1,58 / 0,82 (1,33 / 0,54) | 1,48 / 0,91 (1,22 / 0,54) |
| Skor–volatilite sıra kor. | 0,75–0,95 (girdi kor.) | +0,68; skor–beta +0,45 |
| R²_oos | −1,09% | −0,18% |

Katsayı yine 4/4 dönemde `xs_atr_14_ratio` başta (+1,00); volatilite ailesi bu kez karışık işaretli.
Momentum kıyası geniş evrende sıfır: eski evrendeki +0,027 bu özellik kümesinin değil, 29 hissenin
özelliğiymiş.

**Deney 4a — LightGBM, sabit ayar (kriter: 1/4 geçti → 0/4)**

| Ölçü | 29 hisse | 281 hisse |
|---|---:|---:|
| Rank IC (NW t) | −0,050 (−1,48) | −0,105 (−2,56); sabit günler 0 sayılırsa −0,052 |
| IC hesaplanan gün | 925/925 | 463/925: K3 0/230, K4 0/232 (her gün bütün hisselere aynı skor) |
| Pozitif IC'li dönem | 2/4 | 0/4 |
| QQQ ↓ IC (NW t) | −0,257 (−5,42) | −0,255 (−6,25) |
| Skor–beta / skor–vol. sıra kor. | +0,30 / +0,46 | +0,52 / +0,86 |
| R²_oos | −2,92% | −1,10% |
| Ağaç K1/K2/K3/K4 | 124/1/1/1 | 147/30/1/2 |
| `market_spy_return_20` kazanç payı | %17–87 | %43–91 |

K3–K4'te 1–2 ağaç yalnız gün bazlı `market_spy_return_20`'ye bölünüyor; sıralama üretmiyor.
Kısmi bağımlılık: K1'de volatilite etkisi QQQ ↑ ve ↓ alt kümelerinde aynı yönde pozitif (Δ +0,36 /
+0,30 puan), K2–K4 düz. Koşullu ilişki yine yok.

**Tanı — eski koşuyla yan yana** (`20261016T184204Z` → geniş)

| Soru | 29 hisse | 281 hisse |
|---|---|---|
| Öğrenme eğrisi raw24, 1→1000 ağaç: eğitim IC | +0,07 → +0,27 | +0,006 → +0,10 |
| … validation IC | −0,04 → −0,03 | −0,02 → −0,03 (en kötü 50 ağaçta −0,057) |
| … validation R² | −2% → −15% | −0,8% → −6% |
| Etiket karıştırma, raw24 K1, 200 ağaç: eğitim IC gerçek / karışık | +0,18 / +0,19 | +0,059 / +0,072 |
| … eğitim R² gerçek / karışık | +22% / +10% | +0,5% / +0,2% |
| Yalnız `market_*` ile eğitim R² (200 ağaç) | +18% | +0,9% |
| Izgara (24 ayar) validation IC aralığı raw24 / xs22 | −0,011…−0,037 / −0,036…−0,083 | −0,020…−0,040 / −0,022…−0,053 |
| Momentumu geçen ayar | 0/24 (kıyas +0,027) | 0/24 (kıyas +0,001) |
| Izgarada en yüksek eğitim IC | +0,65 | +0,24 |
| Tek özellik validation IC tavanı | +0,034 (`sma_distance_20`), +0,031 (`return_10`) | +0,016 (`relative_volume_20`); diğerleri ≤ +0,009 |
| Volatilite ailesi IC eğitim → validation | +0,04…+0,05 → −0,09…−0,12 (ters) | −0,005…+0,007 → −0,06…−0,08 |
| Early stopping l2 / IC, raw24 K2 | 1 ağaç −0,090 / 150 ağaç −0,117 | 25 ağaç −0,122 / 1 ağaç −0,014 |

Okuma:

- **Kesit büyümesi LightGBM'i kurtarmadı.** Günde 20 yerine 170–270 hisseyle de ağaç eklemek
  yalnız eğitime uyumu artırıyor; karıştırılmış etiketle eğitim IC'si gerçek etiketinkinden yüksek.
  Eski evrendeki dönem ezberi (gün bazlı özelliklerle eğitim R² +18%) geniş evrende kayboldu
  (+0,9%), yani eğitimdeki uyumun hemen tamamı gürültü düzeyinde.
- **Tavan düştü, momentum kıyası sıfırlandı.** Tek özelliklerin validation IC'si en çok +0,016;
  `return_20` +0,001. Eski evrenin +0,03'lük momentumu geniş evrende yok. Bu özellik kümesiyle
  20 seanslık sıralama sinyali bu evrende de görünmüyor; sorun kesit darlığı değil.
- **Volatilite işareti artık "dönmüyor", eğitimde sıfır.** Eğitim IC'si −0,005…+0,007 iken Ridge
  katsayısı ve LightGBM kazancı yine ATR/volatiliteye yükleniyor: modeller MSE'ye fit ediliyor ve
  kırpılmış hedefte oynak hisselerin eğitim ortalaması yüksek; validation'da bu −0,06…−0,08 IC
  ve QQQ ↓'da −0,22…−0,26 oluyor. Risk kriterleri (skor–beta +0,45/+0,52) eski evrenden kötü.
- **Veri kalitesi notu.** Eğitim hedef varyansı K1'de 0,71 (validation 0,028; eski evrende
  0,027 / 0,023), gün-arası pay %25 → %1. Kaynak tek hisse: QUBT 2018-07-02/03'te
  `target_return_20` +34.567% (Yahoo'da ters split düzeltmesi: kapanış 3,00 → 0,015, `issues.csv`
  DONUK/ZERO_VOL/GIZLI_SPLIT uyarıları var). 2018 hedef varyansı 5,6; diğer yıllar 0,02–0,07.
  2016–2024'te 26 hissede |getiri| > 200% olan 314 satır (GME, RGTI, LUNR, BBAI, MARA…). Eğitim
  hedefinin %1/%99 kırpılması bunu bastırıyor (kırpılmış varyans 0,024), ama early stopping'in
  iç l2'si ve R²_oos kırpılmamış hedefle hesaplanıyor; K1'in iç penceresi (2020) QS (+459%) ve
  HUT (+456%) gibi satırlar içeriyor. Sıralama ölçüleri (rank IC) etkilenmez. Kodda değişiklik yapılmadı;
  ters split artefaktları veri üreticisinde ayrıca incelenmeli.
- Kod düzeltmesi: `research/diag_lgbm.py` ızgara bölümünde momentum kıyası +0,027 olarak gömülüydü
  (eski evrenin değeri); artık bölüm 1'deki `return_20` validation IC ortalamasından okunuyor.
  İlk geniş koşu (`20261016T191010Z_diag_lgbm_h20`) aynı sayılarla yalnız bu metinde farklıdır.

Sonuç: Deney 2 ve 4a geniş evrende de önceden yazılı kriterleri karşılamadı; "Sonraki deneyler"
sırası değişmedi.

## Sonraki deneyler

Aynı değerlendirme çerçevesine, sırası sonuçlara göre belirlenir:

1. Deney 5: üç bariyerli hedef (4b geçemediği için, 4a'nın önceden yazılı kuralı gereği).
2. Lojistik regresyon (yön olasılığı; AUC ve kalibrasyon eklenir).
3. 5 ve 10 seanslık ufuklar.
4. Maliyetli portföy karşılaştırması (top-N, H dilim, %20 pozisyon sınırı,
   10 bps temel, 20/50 bps stres; yalnız long, "sat" = pozisyondan çık).
