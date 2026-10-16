# Aşama 1 — model deneyleri

Amaç: hazır veri setinde basit modellerin, basit kıyasları K1–K4 validation yıllarında
geçip geçmediğini görmek. **2025+ test dönemi bu aşamada okunmaz**; model ve ayarlar
dondurulduktan sonra yalnız bir kez açılır (Aşama 3).

## Deney 1 (bu dal)

```bash
uv run python train_baseline.py
```

Çıktı `output/experiments/<zaman>_ridge_h20/RAPOR.md` ve yanındaki CSV/Parquet
dosyalarıdır. Her çalıştırma `output/experiments/runs.csv` kaydına eklenir; denenen
hiçbir kombinasyon silinmez. Bu klasör Git takibinde değildir.

| Seçim | Değer |
|---|---|
| Ufuk | 20 seans (t+1 açılış → t+20 kapanış) |
| Hedefler | `target_return_20` ve `target_vs_qqq_20` = hisse − aynı pencerede QQQ |
| Girdiler | Manifestteki 24 `feature_columns` |
| Satırlar | `partition_membership.parquet` (purge'lü) ve `volume_features_ready` |
| Kıyaslar | Sıfır, eğitim ortalaması, momentum (`return_20`), geri dönüş (−`return_5`) |
| Model | Ridge: %1/%99 kırpma, standartlaştırma, α eğitimin son 252 seansında seçilir |
| Ölçüler | MSE ve R²_oos; günlük Spearman rank IC, Newey-West t (19 gecikme); dönem tutarlılığı |

Sıfır ve eğitim ortalaması tahmin hatası için, momentum ve geri dönüş sıralama için
kıyastır. Momentum ve geri dönüş getiri birimi üretmediği için MSE'leri raporlanmaz.

Kod: `research/data.py` (hedefler ve bölmeler), `research/models.py`,
`research/metrics.py`, `research/experiment.py`, `research/report.py`.

## Sonraki deneyler

Deney 1 düzgün çalıştıktan sonra, aynı değerlendirme çerçevesine sırayla:

1. Lojistik regresyon (yön olasılığı; AUC ve kalibrasyon eklenir).
2. LightGBM (yeni bağımlılık; yalnız bu adımda eklenir).
3. 5 ve 10 seanslık ufuklar.
4. Maliyetli portföy karşılaştırması (top-N, H dilim, %20 pozisyon sınırı,
   10 bps temel, 20/50 bps stres; yalnız long, "sat" = pozisyondan çık).
