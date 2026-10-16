# v1.1

Bu sürüm v1 incelemesindeki iki düzeltmeyi ve genel uyarı ayrımını içerir.

1. `relative_volume_20` ve `volume_change_1` sağlayıcının `volume` kolonundan
   hesaplanır. `raw_volume` bağımlılığı ve split'te pencere sıfırlama kaldırıldı.
   Formüller, önceki 20 seans penceresi, geçerlilik maskesi ve sıfıra bölme
   davranışı değişmedi. Pencere boyunca tutarlı sağlayıcı düzeltmesi varsayılır;
   gerçek tarihsel pay adedi doğrulanmış sayılmaz.
2. Tarih kesiminde verisi veya eligible geçmişi olmayan hisse atlanır;
   `skipped_symbols` içinde makine tarafından okunabilir gerekçe saklanır.
   Hiç uygun hisse kalmazsa hata verilir. Eksik dosya/bütünlük hataları gizlenmez.
3. Yeni main düzenindeki `symbol="*"` kayıtları `global_quality_issues` altında
   tutulur ve inceleme raporunda ayrı gösterilir. Genel UYARI hisse durumuna
   kopyalanmaz; genel HATA üretimi durdurur.

Fiyat formülleri, giriş t+1 açılışı / çıkış t+H kapanışı, hedef formülleri,
purge kuralları ve model girdi listesi değişmedi. Main'e merge, model eğitimi,
maliyet veya eşik seçimi yapılmadı.
