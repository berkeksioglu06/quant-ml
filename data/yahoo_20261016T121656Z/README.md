# Sabit veri kaydı

16 Ekim 2026 indirmesi; son tamamlanmış seans 15 Ekim. 47 seri.

- raw/: sağlayıcının fiyat Parquet dosyaları; içerik ve SHA-256 korunmuştur.
- raw/earnings/: sağlayıcının bilanço takvimi kayıtları; gerçekleşme garantisi değildir.
- clean/: orijinal kalite hattının temiz dosyaları; içerik değiştirilmedi.
- manifest.json: güncel market_data.py düzenine uyarlanmış manifest.
- issues.csv: ilk kalite raporunun bulguları.
- coverage.csv: güncel main koduyla yeniden üretilen ön sayım.
- provenance/: orijinal sağlayıcı manifesti, temizleme ayarları/logu ve ayrıntılı raporlar.

Bu kayıt yeni market_data.py ile yeniden temizlenmiş değildir. Aynı snapshot üzerinde
build çalıştırmak temiz dosyaları değiştirebilir; yeniden üretim için yeni snapshot
kullanın. İlk hattın temettü/SPAC vb. ayrıntılı kontrollerinin tümü market_data.py'ye
aktarılmadı. v1.1 özellik/etiket üreticisi mevcut temiz tablolardan çalışır.
