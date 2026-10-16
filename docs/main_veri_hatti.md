# Main üzerinde birleşik veri hattı

## Akış

1. `uv run python market_data.py all`: yeni ham fiyatları ve şirketlerin bilanço
   takvim kayıtlarını indirir, SHA-256 ile kaydeder, tamamlanmış seansları temizler.
2. Aynı komut kalite raporlarını ve kapsama ön sayımını üretir.
3. `uv run python build_dataset.py --project . --snapshot <yeni_snapshot> --output output/<yeni_cikti>`:
   temiz tablolardan 24 özellik ve 5/10/20 seanslık getiri etiketlerini üretir.
4. `uv run python verify_dataset.py output/<yeni_cikti>`: çıktı bütünlüğünü doğrular.
5. `uv run streamlit run viz.py`: mevcut görselleştiriciyi açar.

Yeni snapshot adı ilk komutun sonunda yazdırılır. HATA bulguları özellik üretimini
durdurur; UYARI ve BILGI kayıtları incelenmek üzere korunur. Genel `*` uyarıları
bütün hisseleri uyarılı yapmaz. Veri boşlukları doldurulmaz, büyük hareketler silinmez.

## Taşınan kontroller ve sınırları

| Kontrol | Davranış |
|---|---|
| OHLCV | Eksik/sonsuz/geçersiz fiyat, eksik veya negatif hacim, fiyat aralığı ve donmuş satırlar raporlanır. |
| Takvim | İlk seans istenen başlangıç ve halka arz/kuruluş bilgisiyle karşılaştırılır. Son seans eksikse HATA; 20 seanslık kesintisiz boşluk da işlem gören varlıkta HATA. |
| SPAC / faaliyet kesimi | Kesim öncesi bölüm ve yaklaşık 10 dolar fiyat imzası ayrı raporlanır; `eligible=False` korunur. Bu imza şirket kimliğini kanıtlamaz. |
| Temettü | Adj Close/Close faktörü temettü ve sermaye kazancı dağıtımlarıyla karşılaştırılır. Açıklanamayan değişimler raporlanır. |
| Split | Beklenen olaylar yalnız istenen tarih aralığında aranır. Fiyat kırılması HATA; hacim oranı sadece inceleme işaretidir. Yöntem dış referansla doğrulanmadan mutlak tarihsel değer üretilmez. |
| Özel şirket işlemi | XLF'nin 2016-09-19 öncesi mutlak tarihsel fiyat/hacmi doğrulanmamış olarak boş tutulur. Sağlayıcı fiyat/hacmi değiştirilmez. |
| Bilanço | Gelecek tarih, geçmişte sonuç kaydı olan tarih ve geçmişte sonucu belirsiz tarih ayrılır. Eksik takvim fiyat indirmesini bozmaz; kalite raporuna girer. |
| Endeks ve büyük hareket | TNX ölçeği, VIX aralığı ve kayıtlı split olmadan split benzeri hareketler raporlanır. |

Bilanço takvimi şirketin tarihsel duyuru anını kanıtlamaz ve otomatik bir model
özelliği değildir. Hacim özellikleri sağlayıcının `volume` alanından hesaplanan
oranlardır; doğrulanmamış `raw_volume` yüzünden geçmiş pencereler silinmez.

## Arşiv ve eski dal

Repodaki 16 Ekim kaydının ham/temiz dosyaları ve ilk kalite raporları korunur.
Arşivdeki README ilk paketleme anını anlatır; mevcut kalite kodunun kapsamı bu
belgede tanımlıdır. Arşiv üzerinde `build` çalıştırılması engellenir. Aynı ham
veriyle yeni hattın testi ayrı geçici kayıtta yapılır.

`codex/data-pipeline` artık geliştirme hattı değildir. Yereldeki commit edilmemiş
çalışmalar korunmuştur; bu dal main'e topluca merge edilmez. Eski klasör düzenini
okuma desteği geçmiş kayıtları kullanmak içindir; eski veri hattı kodu geri getirilmez.

## Doğrulama

`uv run pytest -q` kalite hatalarını temsil eden sentetik örneklerin yanında,
repodaki 47 seriyi yeni main hattıyla ayrı geçici dizinde yeniden işler. Ardından
özellik/etiket üretir ve çıktının tamamını incelenmiş 65.413 satırlık tabloyla
karşılaştırır. GitHub Actions aynı testi Linux üzerinde de çalıştırır.
