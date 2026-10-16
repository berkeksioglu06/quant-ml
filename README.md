# quant-ml

Günlük piyasa verisini indirir, temizler ve araştırma için özellik/etiket tabloları üretir.

```bash
uv sync --group dev
uv run python market_data.py all
uv run streamlit run viz.py
```

## Tek aktif veri hattı

Yeni geliştirmeler güncel main yapısında yapılır: `market_data.py` indirme,
temizleme ve kalite kontrolünü, `dataset_builder/` özellik ve etiket üretimini,
`viz.py` görselleştirmeyi yürütür. Eski `codex/data-pipeline` dalı geliştirme
kaynağı olarak kullanılmaz; silinmiş `src/quant_ml/data/` yapısı geri getirilmez.

`all` veya `build` yeni kayıtta `issues.csv`, `quality_detail.json` ve
`quality_report.md` üretir. Taşınan kontroller: SPAC/faaliyet öncesi dönem,
ilk/son seans ve uzun veri boşlukları, OHLCV, temettü/Adj Close tutarlılığı,
split fiyat kırılmaları ve gizli split şüphesi, endeks ölçeği, bilanço takvimi.
Bilanço tarihleri yeni indirmelerde alınır ve checksum ile saklanır; eksik veya
belirsiz sonuçlar açıkça raporlanır. Bu kayıtlar 24 özelliğe eklenmez.
XLF'nin özel şirket işlemi öncesinde tarihsel mutlak fiyat/hacim üretilmez.

Ayrıntılar: [docs/main_veri_hatti.md](docs/main_veri_hatti.md).

## Repodaki veri kaydı

`data/yahoo_20261016T121656Z/` 16 Ekim 2026 indirmesidir; son tamamlanmış fiyat günü
15 Ekim 2026. 47 piyasa serisinin ham ve temiz dosyaları, bilanço kaynak kayıtları,
bütünlük kayıtları ve kalite bulguları dahildir. Eski SQ kodu yardımcı kaynak kaydında
saklanır; XYZ geçmişi 2016'dan itibaren mevcut olduğu için birleştirme gerekmez.

Kaydın klasör/manifest biçimi güncel `market_data.py` ve `viz.py` ile uyumlu hale
getirildi. Ham ve temiz Parquet içerikleri değiştirilmedi. İlk üretimin kaynak,
ayarlama ve kalite kayıtları `provenance/` altında korunur.

## Özellik ve etiket üreticisi v1.1

```bash
uv run python build_dataset.py --project . --snapshot yahoo_20261016T121656Z --output output/yeni_veri_seti
uv run python verify_dataset.py output/yeni_veri_seti
uv run pytest -q
```

Çıktı klasörü daha önce var olmamalı. Hazır 29 hisse / 65.413 satırlık çıktı:
[`output/yahoo_20261016T121656Z_dataset_v1_1/INCELEME.html`](output/yahoo_20261016T121656Z_dataset_v1_1/INCELEME.html).
HTML'yi yerel olarak tarayıcıda açın; GitHub HTML dosyalarını doğrudan çalıştırmaz.

- 24 özellik; 5/10/20 seanslık getiri hedefleri.
- Giriş t+1 açılışı, çıkış t+H kapanışı; dönem sınırını aşan etiketler çıkarılır.
- Göreli hacim sağlayıcının `volume` kolonunu kullanır; mutlak tarihsel hacim
  doğrulaması iddia edilmez. Split kaydı pencereyi sıfırlamaz.
- Erken tarih kesiminde henüz verisi/eligible dönemi olmayan hisseler gerekçeyle atlanır.
- Model girdileri yalnız manifestteki `feature_columns` listesidir; hedefler girdiye alınmaz.

Kullanım ve formüller: [docs/dataset_builder.md](docs/dataset_builder.md).
Kalite uyarıları korunur; bu çıktılar model eğitimi veya veri kabulü anlamına gelmez.
Maliyet ve sınıf eşikleri seçilmedi.

## Aşama 1 model deneyleri

```bash
uv run python train_baseline.py --deney 1
uv run python train_baseline.py --deney 2
uv run python train_baseline.py --deney 3
uv run python train_baseline.py --deney 4a
uv run python train_baseline.py --deney 4b
```

20 seanslık ufukta basit kıyaslar ve Ridge K1–K4 validation yıllarında karşılaştırılır.
Deney 1 ham özellikleri ve iki hedefi (hissenin kendi getirisi, QQQ'ya göre getiri),
Deney 2 gün içi sıralanmış özellikleri ve gün ortalamasından arındırılmış hedefi kullanır.
Deney 3 aynı özelliklerle betadan arındırılmış artık getiriyi hedefler. Deney 4a, Deney 1'in
kurulumunda Ridge yerine sabit ayarlı LightGBM kullanır (macOS'ta `brew install libomp` gerekir);
Deney 4b buna QQQ ve VIX'ten 6 piyasa durumu özelliği ekler.
2025+ test dönemi okunmaz. Rapor `output/experiments/<zaman>_<deney>_h20/RAPOR.md` altına yazılır.
Ayrıntılar: [docs/asama1_plan.md](docs/asama1_plan.md).

Yalnız yukarıdaki seçili snapshot ve hazır çıktı Git takibindedir. Sonraki indirmeler
ve deney çıktıları varsayılan olarak yok sayılır. Parquet dosyaları Git LFS gerektirmez;
en büyük dosya yaklaşık 16 MB'dir.
