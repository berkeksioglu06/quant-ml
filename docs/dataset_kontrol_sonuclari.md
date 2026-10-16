# Teslim kontrolü

16 Ekim 2026 · 1.1 inceleme sürümü

## Doğrulananlar

- **Main entegrasyonunda 90 otomatik test geçti.** Kayıt: `validation/dataset_builder_v1_1/main_quality_tests.xml`. İlk bağımsız paket incelemesindeki 69 testin kaydı da saklanır.
- Eski ve yeni klasör düzenlerinde aynı girdiyle aynı veri seti üretildi.
- 24 formülün tamamı bağımsız skaler hesaplarla sentetik veride karşılaştırıldı.
- Gelecek satırların değiştirilmesi geçmiş özellikleri değiştirmedi.
- Gerçek veride **110 tarih kesimi** tam/kesilmiş veri eşitliğini geçti.
- Gerçek veride **203 özellik hesabı** ve **261 etiket hesabı** bağımsız hesapla karşılaştırıldı.
- 4 dönem × 3 ufuk için **12 kronolojik sınır kontrolü** uygulandı.
- Eksik seans, bozuk fiyat, SPAC öncesi geçmiş, split, temettü, sıfır hacim,
  sıfıra bölme, kaynak bütünlüğü, üzerine yazma ve maliyet kuralları test edildi.
- Kaynak dosyalar değişmedi. Çıktı dosyalarının bütünlüğü ayrıca tekrar kontrol edildi.
- İnceleme raporu tarayıcıda açılıp görünümü kontrol edildi; yatay sayfa taşması yok.

## Gerçek çıktı

- Snapshot: `yahoo_20261016T121656Z`.
- Son veri günü: **15 Ekim 2026**.
- **29 hisse · 65.413 satır · 24 özellik · 3 getiri ufku**.
- NBIS verisi tabloda bulunur, eğitim adayı değildir.
- Mevcut kaynak raporundaki uyarılar korunur; eğitim onayı verilmez.
- Örnekte maliyet ve sınıf eşiği seçilmedi. Net getiri kolonlarının boş olması beklenir.

## v1 → v1.1 düzeltme denetimi

- Hacim oranları `volume` kullanır; split kaydında pencere sıfırlanmaz.
- K1/H5 `train_all_24`: **11.766 → 20.137**. Tüm 12 dönem/ufukta hem eğitim
  hem validation için all_24 ve price_only sayıları eşitlendi.
- Fiyat özellikleri, bütün hedefler, giriş/çıkış tarihleri ve purge sayıları v1 ile aynı.
- `labels.py` dosyası v1 ile bayt düzeyinde aynı; etiket formülleri değiştirilmedi.
- Genel UYARI kayıtları hisseye özel kalite durumundan ayrıldı; genel HATA durdurucu.
- İlk sürüm ZIP'i korunmuştur. Ayrıntılı karşılaştırma: `validation/dataset_builder_v1_1/regression.json`.

| Tarih kesimi | Hisse | Satır | Atlananlar |
|---|---:|---:|---|
| 2020-12-31 | 24 | 24.723 | RKLB, HIMS, SOFI, RIVN, NBIS |
| 2022-12-30 | 28 | 38.315 | NBIS |
| 2024-12-31 | 29 | 52.421 | Yok |

Her kesimde özellikler tam üretimin aynı günleriyle birebir eşleşti. Sonucu kesim
tarihinde bilinen etiketler de eşleşti; henüz bilinmeyen sonuçlar boş bırakıldı.

## İddia edilmeyenler

Henüz model eğitimi, kârlılık kanıtı, portföy backtesti veya yeni main'e merge yoktur.
Verinin güncel düzeltilmiş hali geçmişte yayımlanmış halinin aynısı olmak zorunda değildir.
Kaynak kalite kontrolleri `market_data.py` içine taşındı; yeni temizleme + kalite + özellik/etiket zinciri gerçek veride test edildi. 65.413 satır ve 50 kolonun tamamı incelenmiş tabloyla birebir eşleşti.
Bu paket seçilen temiz veriden türetilmiş tablolar üretir; yeni veri indirmez.
