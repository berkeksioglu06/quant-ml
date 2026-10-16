# quant-ml

Günlük piyasa verisini indirir, temizler ve araştırma için özellik/etiket tabloları üretir.

```bash
uv sync --group dev
uv run python market_data.py all
uv run streamlit run viz.py
```

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
Maliyet ve sınıf eşikleri seçilmedi; model henüz eğitilmedi.

Yalnız yukarıdaki seçili snapshot ve hazır çıktı Git takibindedir. Sonraki indirmeler
ve deney çıktıları varsayılan olarak yok sayılır. Parquet dosyaları Git LFS gerektirmez;
en büyük dosya yaklaşık 16 MB'dir.
