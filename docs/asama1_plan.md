# Aşama 1 — model deneyleri

Amaç: hazır veri setinde basit modellerin, basit kıyasları K1–K4 validation yıllarında
geçip geçmediğini görmek. **2025+ test dönemi bu aşamada okunmaz**; fiyatlar ve veri
seti satırları test başlangıcından önce kesilir. Model ve ayarlar dondurulduktan sonra
test dönemi yalnız bir kez açılır (Aşama 3).

```bash
uv run python train_baseline.py --deney 1
uv run python train_baseline.py --deney 2
uv run python train_baseline.py --deney 3
```

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
| Model | Ridge: %1/%99 kırpma, standartlaştırma, α eğitimin son 252 seansında seçilir |
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

Kod: `research/data.py` (hedefler, beta, rejim, bölmeler), `research/models.py`,
`research/metrics.py`, `research/experiment.py`, `research/report.py`.

## Sonraki deneyler

Aynı değerlendirme çerçevesine, sırası sonuçlara göre belirlenir:

1. Deney 4 (Deney 3'ün önceden yazılı kuralı gereği): uzun momentum, 60/120/250 seans
   getirisi. Seçim validation sonuçlarına bakılarak özellik atmak biçiminde yapılmaz.
2. Lojistik regresyon (yön olasılığı; AUC ve kalibrasyon eklenir).
3. LightGBM (yeni bağımlılık; yalnız bu adımda eklenir).
4. 5 ve 10 seanslık ufuklar.
5. Maliyetli portföy karşılaştırması (top-N, H dilim, %20 pozisyon sınırı,
   10 bps temel, 20/50 bps stres; yalnız long, "sat" = pozisyondan çık).
