# Veri kalite raporu — yahoo_20261016T121656Z
Kaynak: Yahoo Finance via yfinance Ticker.history · indirme 2026-10-16T12:16:56 UTC · istenen başlangıç 2016-01-01 · beklenen son seans 2026-10-15
Sürümler: yfinance 1.7.0, pandas 3.0.6, pyarrow 25.0.1, exchange_calendars 4.13.2, Python 3.12.14
İndirme seçenekleri: `{'auto_adjust': False, 'back_adjust': False, 'actions': True, 'repair': False, 'interval': '1d'}` — ham snapshot salt-okunur, temizleme ayrı katmanda.
Checksum doğrulaması (temizlemeden önce): 76 dosya doğrulandı, uyuşmayan 0, checksumsız 0.

> **Test dönemi notu:** Kalite kontrolleri (eksik seans, tutarlılık, sıçrama, likidite istatistikleri) 2025–2026 test dönemini de kapsar; veri kalitesi için bu inceleme gereklidir. Kural: bu rapordaki test dönemi bilgisi model, özellik, eşik veya evren seçiminde **kullanılmaz**.

**Durum:** 26 OK · 21 UYARI · 0 HATA (toplam 47 sembol)

## 1. Eğitimden önce çözülmesi gerekenler (HATA)
- Yok.

## 2. Sembol özeti
| Sembol | Rol | Durum | İndirme | Satır | Eligible | İlk | Kesim | Son | Eksik seans | Fazla tarih | H/L ihlal | 0 hacim | Split | Temettü | Büyük hareket |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RKLB | ana20,islem10 | OK | OK | 1479 | 1291 | 2020-11-24 | 2021-08-25 | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| HIMS | ana20,islem10 | UYARI | OK | 1782 | 1441 | 2019-09-13 | 2021-01-21 | 2026-10-15 | 0 | 0 | 0 | 63 | 0 | 0 | 7 |
| SOFI | ana20,islem10 | OK | OK | 1453 | 1351 | 2021-01-04 | 2021-06-01 | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| PLTR | ana20,islem10 | OK | OK | 1518 | 1518 | 2020-09-30 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| MU | ana20,islem10 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 21 | 0 |
| MRVL | ana20,islem10 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 2 |
| BE | ana20,islem10 | OK | OK | 2068 | 2068 | 2018-07-25 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| TSLA | ana20,islem10 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| AMD | ana20,islem10 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| NVDA | ana20,islem10 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 2 | 43 | 1 |
| ENPH | ana20 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| SHOP | ana20 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| ROKU | ana20 | UYARI | OK | 2274 | 2274 | 2017-09-28 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| XYZ | ana20 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| SNAP | ana20 | UYARI | OK | 2420 | 2420 | 2017-03-02 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| UPST | ana20 | OK | OK | 1464 | 1464 | 2020-12-16 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| SMCI | ana20 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 0 | 12 |
| PLUG | ana20 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| PTON | ana20 | UYARI | OK | 1773 | 1773 | 2019-09-26 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| RIVN | ana20 | OK | OK | 1237 | 1237 | 2021-11-10 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| AAPL | kontrol8 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| MSFT | kontrol8 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| GOOGL | kontrol8 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 10 | 0 |
| AMZN | kontrol8 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| MRNA | kontrol8 | OK | OK | 1974 | 1974 | 2018-12-07 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| JPM | kontrol8 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 44 | 0 |
| XOM | kontrol8 | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| WMT | kontrol8 | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| NBIS | nbis | OK | OK | 498 | 498 | 2024-10-21 | 2024-10-21 | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| SPY | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| QQQ | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| IWM | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| ^VIX | baglam | UYARI | OK | 2714 | 2714 | 2016-01-04 |  | 2026-10-15 | 0 | 2 | 0 |  | 0 |  | 40 |
| ^TNX | baglam | OK | OK | 2711 | 2711 | 2016-01-04 |  | 2026-10-15 | 1 | 0 | 0 |  | 0 |  | 4 |
| XLK | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| SMH | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 10 | 0 |
| IGV | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 17 | 0 |
| XLC | baglam | OK | OK | 2093 | 2093 | 2018-06-19 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 33 | 0 |
| XLY | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| XLF | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| XLE | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 44 | 0 |
| XLV | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 44 | 0 |
| XLP | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| XLI | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 43 | 0 |
| ITA | baglam | UYARI | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 1 | 43 | 0 |
| ICLN | baglam | OK | OK | 2712 | 2712 | 2016-01-04 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 23 | 0 |
| UFO | baglam | OK | OK | 1889 | 1889 | 2019-04-11 |  | 2026-10-15 | 0 | 0 | 0 | 0 | 0 | 26 | 0 |

## 3. Kurumsal işlemler
`auto_adjust=False` gerçek tarihsel fiyat garantisi vermez. Split günü fiyat oranı ve 20 seans hacim oranı yalnız **inceleme işaretidir** (`*_gibi`), doğrulama sayılmaz. Sağlayıcının split yöntemi fiyat ve hacim için ayrı ayrı, `config/data/reference_points.csv`'deki bağımsız kaynaklı gerçek değerlerle belirlenir. Yöntem doğrulanmadıkça `raw_close` / `raw_volume` son split'ten önceki satırlarda üretilmez (NaN); ≥5$ filtresi o dönemde uygulanamaz.

- **Fiyat yöntemi:** dogrulandi → `geri_duzeltilmis` · bilgi taşıyan referans noktası 4, semboller ['AAPL', 'AMZN', 'GOOGL', 'TSLA']
- **Hacim yöntemi:** dogrulanmadi (referans yok) · bilgi taşıyan referans noktası 0, semboller []
- Referans dosyası: `C:\Users\user\Projects\quant-ml\config\data\reference_points.csv` (10 satır)

| Sembol | date | ratio | close_ratio | open_ratio | fiyat_isaret | volume_ratio | hacim_isaret |
|---|---|---|---|---|---|---|---|
| TSLA | 2020-08-31 | 5 | 1.1257 | 1.0044 | duzeltilmis_gibi | 1.278 | duzeltilmis_gibi |
| TSLA | 2022-08-25 | 3 | 0.9965 | 1.0177 | duzeltilmis_gibi | 0.686 | duzeltilmis_gibi |
| NVDA | 2021-07-20 | 4 | 0.9911 | 0.9974 | duzeltilmis_gibi | 0.509 | duzeltilmis_gibi |
| NVDA | 2024-06-10 | 10 | 1.0075 | 0.9957 | duzeltilmis_gibi | 0.677 | duzeltilmis_gibi |
| SHOP | 2022-06-29 | 10 | 0.9436 | 0.9821 | duzeltilmis_gibi | 0.651 | duzeltilmis_gibi |
| SMCI | 2024-10-01 | 10 | 0.9738 | 1.0026 | duzeltilmis_gibi | 0.538 | duzeltilmis_gibi |
| AAPL | 2020-08-31 | 4 | 1.0339 | 1.0222 | duzeltilmis_gibi | 0.995 | duzeltilmis_gibi |
| GOOGL | 2022-07-18 | 20 | 0.9754 | 1.0077 | duzeltilmis_gibi | 0.785 | duzeltilmis_gibi |
| AMZN | 2022-06-06 | 20 | 1.0199 | 1.0237 | duzeltilmis_gibi | 0.707 | duzeltilmis_gibi |
| WMT | 2024-02-26 | 3 | 1.0185 | 1.0103 | duzeltilmis_gibi | 0.863 | duzeltilmis_gibi |
| XLK | 2025-12-05 | 2 | 1.0073 | 1.0046 | duzeltilmis_gibi | 0.473 | duzeltilmis_gibi |
| SMH | 2023-05-05 | 2 | 1.0211 | 1.0029 | duzeltilmis_gibi | 1.053 | duzeltilmis_gibi |
| IGV | 2024-03-07 | 5 | 1.0118 | 1.0087 | duzeltilmis_gibi | 0.916 | duzeltilmis_gibi |
| XLY | 2025-12-05 | 2 | 1.0055 | 1.0009 | duzeltilmis_gibi | 0.748 | duzeltilmis_gibi |
| XLF | 2016-09-19 | 1.231 | 1.0064 | 0.9996 | duzeltilmis_gibi | 0.872 | duzeltilmis_gibi |
| XLE | 2025-12-05 | 2 | 0.9959 | 0.9983 | duzeltilmis_gibi | 0.845 | duzeltilmis_gibi |
| ITA | 2020-12-07 | 2 | 0.9989 | 1.0028 | duzeltilmis_gibi | 0.586 | duzeltilmis_gibi |
- XLF: manifestoda olmayan split: 2016-09-19 oran 1.231 (manifestoya eklenmeli)

**Temettü / Adj Close tutarlılığı** — her ex-date'te Adj/Close faktör değişimi, `1 − temettü/önceki kapanış` ile karşılaştırılır.

| Sembol | Olay | Max faktör hatası | Uyumsuz olay | Temettüsüz Adj değişimi | Son Adj/Close |
|---|---|---|---|---|---|
| MU | 21 | 0 | 0 | 0 | 1 |
| MRVL | 43 | 0 | 0 | 0 | 1 |
| NVDA | 43 | 0 | 0 | 0 | 1 |
| AAPL | 43 | 0 | 0 | 0 | 1 |
| MSFT | 43 | 0 | 0 | 0 | 1 |
| GOOGL | 10 | 0 | 0 | 0 | 1 |
| JPM | 44 | 0 | 0 | 0 | 1 |
| XOM | 43 | 0 | 0 | 0 | 1 |
| WMT | 43 | 0 | 0 | 0 | 1 |
| SPY | 43 | 0 | 0 | 0 | 1 |
| QQQ | 43 | 0 | 0 | 0 | 1 |
| IWM | 43 | 0 | 0 | 0 | 1 |
| XLK | 43 | 0 | 0 | 0 | 1 |
| SMH | 10 | 0 | 0 | 0 | 1 |
| IGV | 17 | 0 | 0 | 0 | 1 |
| XLC | 33 | 0 | 0 | 0 | 1 |
| XLY | 43 | 0 | 0 | 0 | 1 |
| XLF | 43 | 0 | 0 | 0 | 1 |
| XLE | 44 | 0 | 0 | 0 | 1 |
| XLV | 44 | 0 | 0 | 0 | 1 |
| XLP | 43 | 0 | 0 | 0 | 1 |
| XLI | 43 | 0 | 0 | 0 | 1 |
| ITA | 43 | 0 | 0 | 0 | 1 |
| ICLN | 23 | 0 | 0 | 0 | 1 |
| UFO | 26 | 0 | 0 | 0 | 1 |

## 4. Geçmişin uygunluğu: SPAC, ticker değişimi, yeniden yapılanma, halka arz
Kesim öncesi satırlar silinmedi; temiz katmanda `eligible=False`. SPAC imzası: kesim öncesi gerçek fiyatın ~$10 bandında kalma oranı ve çok düşük günlük getiri.

| Sembol | Kesim | Öncesi satır | Öncesi aralık | Öncesi medyan $ | Öncesi $9–11.5 payı | Öncesi medyan |r| | Sonrası (60g) medyan |r| | Sonrası satır |
|---|---|---|---|---|---|---|---|---|
| RKLB | 2021-08-25 | 188 | 2020-11-24 → 2021-08-24 | 10.45 | 0.888 | 0.0093 | 0.0309 | 1291 |
| HIMS | 2021-01-21 | 341 | 2019-09-13 → 2021-01-20 | 10.05 | 0.906 | 0.002 | 0.0446 | 1441 |
| SOFI | 2021-06-01 | 102 | 2021-01-04 → 2021-05-28 | 18.22 | 0 | 0.0292 | 0.0259 | 1351 |
| NBIS | 2024-10-21 | 0 |  |  |  |  | 0.0442 | 498 |

**Beklenen ilk seans kontrolü** — her sembol için: halka arz/kuruluş tarihi, yoksa faaliyet kesimi, yoksa istenen başlangıç (2016). Veri daha geç başlıyorsa HATA.

| Sembol | Beklenen ilk seans | Dayanak | Veride ilk | Fark (gün) | Başta eksik seans | Yahoo firstTradeDate |
|---|---|---|---|---|---|---|
| RKLB | 2021-08-25 | faaliyet / kesim başlangıcı (veri en geç burada başlamalı) | 2020-11-24 | -274 | 0 |  |
| HIMS | 2021-01-21 | faaliyet / kesim başlangıcı (veri en geç burada başlamalı) | 2019-09-13 | -496 | 0 |  |
| SOFI | 2021-06-01 | faaliyet / kesim başlangıcı (veri en geç burada başlamalı) | 2021-01-04 | -148 | 0 |  |
| PLTR | 2020-09-30 | halka arz / kuruluş | 2020-09-30 | 0 | 0 |  |
| MU | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| MRVL | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| BE | 2018-07-25 | halka arz / kuruluş | 2018-07-25 | 0 | 0 |  |
| TSLA | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| AMD | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| NVDA | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ENPH | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| SHOP | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ROKU | 2017-09-28 | halka arz / kuruluş | 2017-09-28 | 0 | 0 |  |
| XYZ | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| SNAP | 2017-03-02 | halka arz / kuruluş | 2017-03-02 | 0 | 0 |  |
| UPST | 2020-12-16 | halka arz / kuruluş | 2020-12-16 | 0 | 0 |  |
| SMCI | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| PLUG | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| PTON | 2019-09-26 | halka arz / kuruluş | 2019-09-26 | 0 | 0 |  |
| RIVN | 2021-11-10 | halka arz / kuruluş | 2021-11-10 | 0 | 0 |  |
| AAPL | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| MSFT | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| GOOGL | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| AMZN | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| MRNA | 2018-12-07 | halka arz / kuruluş | 2018-12-07 | 0 | 0 |  |
| JPM | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XOM | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| WMT | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| NBIS | 2024-10-21 | faaliyet / kesim başlangıcı (veri en geç burada başlamalı) | 2024-10-21 | 0 | 0 |  |
| SPY | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| QQQ | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| IWM | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ^VIX | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ^TNX | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLK | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| SMH | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| IGV | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLC | 2018-06-19 | halka arz / kuruluş | 2018-06-19 | 0 | 0 |  |
| XLY | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLF | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLE | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLV | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLP | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| XLI | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ITA | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| ICLN | 2016-01-04 | istenen başlangıç | 2016-01-04 | 0 | 0 |  |
| UFO | 2019-04-01 | yaklaşık kuruluş ayı | 2019-04-11 | 10 | 0 |  |

**XYZ ← SQ (ticker değişimi 2025-01-21)**

- XYZ: 2016-01-04 → 2026-10-15 (2712 satır)
- SQ: Yahoo'da veri yok
- Değişim ±7 gün içinde en büyük |getiri|: 4.27% (ticker değişimi fiyatta kırılma yaratmamalı)
- Onay (universe.json `legacy_merge_approved`): False · Karar: **eski sembol verisi yok**

**Manifesto notları**

- RKLB: SPAC: Vector Acquisition (VACQ) birleşmesi; RKLB olarak işlem 2021-08-25. Öncesi boş SPAC dönemi.
- HIMS: SPAC: Oaktree Acquisition Corp (OAC) birleşmesi; HIMS olarak işlem 2021-01-21.
- SOFI: SPAC: Social Capital Hedosophia V (IPOE) birleşmesi; SOFI olarak işlem 2021-06-01.
- PLTR: Doğrudan listeleme 2020-09-30.
- BE: Halka arz 2018-07-25.
- ROKU: Halka arz 2017-09-28.
- XYZ: Block: SQ -> XYZ ticker değişimi 2025-01-21. Geçmişin XYZ altında kesintisiz olup olmadığı kontrol edilir.
- SNAP: Halka arz 2017-03-02.
- UPST: Halka arz 2020-12-16.
- PTON: Halka arz 2019-09-26.
- RIVN: Halka arz 2021-11-10.
- MRNA: Halka arz 2018-12-07.
- NBIS: Eski Yandex N.V.; işlem 2022-02-28'de durdu, NBIS olarak 2024-10-21'de yeniden başladı. Yandex geçmişi birleştirilmez.

## 5. Takvim ve eksikler
ABD (XNYS) seans takvimine göre, sembolün ilk–son (eligible) aralığında. Eksikler doldurulmadı. Endekslerde (^VIX, ^TNX) farklı yayın takvimi BİLGI düzeyinde.

| Sembol | Eksik | En uzun boşluk | Örnek eksik | Fazla | Örnek fazla |
|---|---|---|---|---|---|
| ^VIX | 0 | 0 |  | 2 | 2026-05-25, 2026-09-07 |
| ^TNX | 1 | 1 | 2016-11-11 | 0 |  |

## 6. Büyük hareketler (|kapanış getirisi| > %25) — silinmedi
| Sembol | date | ret | gap | split | div | vol_x |
|---|---|---|---|---|---|---|
| MRNA | 2026-08-19 | 1.7697 | 0.8428 | 0 | 0 | 39.06 |
| ^VIX | 2018-02-05 | 1.156 | 0.0653 | 0 | 0 |  |
| UPST | 2021-03-18 | 0.8932 | 0.3742 | 0 | 0 | 15.06 |
| ^VIX | 2024-12-18 | 0.7404 | -0.0189 | 0 | 0 |  |
| PLUG | 2017-04-05 | 0.7308 | 0.8308 | 0 | 0 | 31.28 |
| ^VIX | 2024-08-05 | 0.649 | 0 | 0 | 0 |  |
| ^VIX | 2021-01-27 | 0.6164 | 0.0348 | 0 | 0 |  |
| BE | 2024-11-15 | 0.5919 | 0.5776 | 0 | 0 | 10.1 |
| SNAP | 2022-02-04 | 0.5882 | 0.4396 | 0 | 0 | 11.22 |
| SOFI | 2021-01-07 | 0.5792 | 0.0066 | 0 | 0 | 49.75 |
| UPST | 2022-05-10 | -0.5642 | -0.5594 | 0 | 0 | 8.12 |
| ROKU | 2017-11-09 | 0.5494 | 0.3137 | 0 | 0 | 17.99 |
| ^VIX | 2021-11-26 | 0.5404 | 0.4327 | 0 | 0 |  |
| AMD | 2016-04-22 | 0.5229 | 0.2176 | 0 | 0 | 15.82 |
| ^VIX | 2025-04-04 | 0.5093 | 0.0033 | 0 | 0 |  |
| ^TNX | 2020-03-10 | 0.499 | 0.3106 | 0 | 0 |  |
| NBIS | 2025-09-09 | 0.4942 | 0.5173 | 0 | 0 | 9.52 |
| ^VIX | 2016-06-24 | 0.4933 | 0.5107 | 0 | 0 |  |
| ^VIX | 2020-06-11 | 0.4795 | 0.1045 | 0 | 0 |  |
| SNAP | 2018-02-07 | 0.4758 | 0.2198 | 0 | 0 | 15.67 |
| ^VIX | 2020-02-24 | 0.4655 | 0.3027 | 0 | 0 |  |
| ^VIX | 2017-05-17 | 0.4638 | 0.1164 | 0 | 0 |  |
| UPST | 2024-11-08 | 0.4602 | 0.1767 | 0 | 0 | 6.23 |
| BE | 2020-03-24 | 0.4447 | 0.0977 | 0 | 0 | 1.81 |
| ^VIX | 2017-08-10 | 0.4437 | 0.0414 | 0 | 0 |  |
| ^VIX | 2018-10-10 | 0.4395 | 0.005 | 0 | 0 |  |
| SNAP | 2022-05-24 | -0.4308 | -0.3551 | 0 | 0 | 9.01 |
| ^VIX | 2020-03-16 | 0.4299 | 0 | 0 | 0 |  |
| BE | 2019-08-13 | -0.425 | -0.2525 | 0 | 0 | 16.54 |
| ENPH | 2020-02-19 | 0.4244 | 0.1827 | 0 | 0 | 6.95 |
| BE | 2020-03-19 | 0.4235 | 0.0391 | 0 | 0 | 1.54 |
| ^VIX | 2020-02-27 | 0.4209 | 0.0504 | 0 | 0 |  |
| SMCI | 2018-10-04 | -0.4112 | -0.2916 | 0 | 0 | 44.86 |
| HIMS | 2026-03-09 | 0.4079 | 0.4749 | 0 | 0 | 5.71 |
| PLUG | 2023-11-10 | -0.4047 | -0.3761 | 0 | 0 | 9.7 |
| ^VIX | 2020-03-12 | 0.4002 | 0.1403 | 0 | 0 |  |
| ^VIX | 2016-09-09 | 0.3989 | 0.0008 | 0 | 0 |  |
| ^VIX | 2026-06-05 | 0.3968 | 0.0305 | 0 | 0 |  |
| ^VIX | 2019-08-05 | 0.3964 | 0.1334 | 0 | 0 |  |
| ^VIX | 2025-04-03 | 0.3956 | 0.2264 | 0 | 0 |  |

_… toplam 160 olay; tamamı `buyuk_hareketler.csv`._

## 7. Bağlam serileri
- **^VIX**: aralık 9.14–82.69, son 1 yıl medyan 17.09
- **^TNX**: aralık 0.50–5.31, son 1 yıl medyan 4.34; ölçek: yüzde (ör. 4.25 = %4.25)

Bağlam ETF'lerinin kapsaması (ısınma = ilk 200 seans):

| symbol | roles | ilk_tarih | isinma_bitis | 2016_basindan_tam | K1 | K2 | K3 | K4 | TEST |
|---|---|---|---|---|---|---|---|---|---|
| SPY | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| QQQ | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| IWM | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| ^VIX | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| ^TNX | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLK | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| SMH | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| IGV | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLC | baglam | 2018-06-19 | 2019-04-05 | False | kısmi | kısmi | kısmi | kısmi | var |
| XLY | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLF | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLE | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLV | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLP | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| XLI | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| ITA | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| ICLN | baglam | 2016-01-04 | 2016-10-18 | True | tam | tam | tam | tam | var |
| UFO | baglam | 2019-04-11 | 2020-01-28 | False | kısmi | kısmi | kısmi | kısmi | var |

## 8. Bilanço: sağlayıcı takvim kayıtları (Yahoo `get_earnings_dates`)
Bunlar **sağlayıcı takvim kaydıdır**; gerçekleşme, ilan saati ve önceden duyurulma zamanı bu kaynakla doğrulanmaz. Sınıflar: `planlanan` (indirme anından sonra), `sonuç kaydı var` (geçmiş tarih + sağlayıcıda raporlanan EPS; yine doğrulanmamış), `durumu belirsiz` (geçmiş tarih, sonuç bilgisi yok — gerçekleşmiş sayılmaz). "Bilançoya kalan gün" özelliği, tarihsel duyuru zamanı doğrulanmadan kullanılmaz.

| Sembol | Durum | Kayıt | En eski | Sonuç kaydı var | Durumu belirsiz | Planlanan | ~Beklenen çeyrek | Saat bilgisi |
|---|---|---|---|---|---|---|---|---|
| RKLB | var | 21 | 2021-11-15 | 20 | 0 | 1 | 24 | True |
| HIMS | var | 24 | 2021-03-18 | 23 | 0 | 1 | 28 | True |
| SOFI | var | 22 | 2021-08-12 | 21 | 0 | 1 | 23 | True |
| PLTR | var | 25 | 2020-11-12 | 24 | 0 | 1 | 24 | True |
| MU | var | 100 | 2002-03-21 | 99 | 0 | 1 | 43 | True |
| MRVL | var | 100 | 2002-02-28 | 99 | 0 | 1 | 43 | True |
| BE | var | 33 | 2018-11-06 | 32 | 0 | 1 | 33 | True |
| TSLA | var | 65 | 2010-11-09 | 63 | 1 | 1 | 43 | True |
| AMD | var | 100 | 2002-01-16 | 98 | 1 | 1 | 43 | True |
| NVDA | var | 100 | 2002-05-22 | 71 | 28 | 1 | 43 | True |
| ENPH | var | 59 | 2012-05-10 | 58 | 0 | 1 | 43 | True |
| SHOP | var | 46 | 2015-07-30 | 34 | 11 | 1 | 43 | True |
| ROKU | var | 37 | 2017-11-08 | 35 | 1 | 1 | 36 | True |
| XYZ | var | 44 | 2016-03-10 | 43 | 0 | 1 | 43 | True |
| SNAP | var | 39 | 2017-05-10 | 37 | 1 | 1 | 38 | True |
| UPST | var | 24 | 2021-03-17 | 23 | 0 | 1 | 23 | True |
| SMCI | var | 76 | 2007-05-09 | 73 | 2 | 1 | 43 | True |
| PLUG | var | 100 | 2002-02-26 | 99 | 0 | 1 | 43 | True |
| PTON | var | 29 | 2019-11-05 | 27 | 1 | 1 | 28 | True |
| RIVN | var | 21 | 2021-12-16 | 20 | 0 | 1 | 20 | True |
| AAPL | var | 100 | 2002-04-17 | 87 | 12 | 1 | 43 | True |
| MSFT | var | 100 | 2002-01-17 | 99 | 0 | 1 | 43 | True |
| GOOGL | var | 89 | 2004-10-21 | 87 | 1 | 1 | 43 | True |
| AMZN | var | 100 | 2002-04-23 | 91 | 8 | 1 | 43 | True |
| MRNA | var | 32 | 2019-03-13 | 31 | 0 | 1 | 31 | True |
| JPM | var | 100 | 2002-04-17 | 99 | 0 | 1 | 43 | True |
| XOM | var | 100 | 2002-01-23 | 99 | 0 | 1 | 43 | True |
| WMT | var | 100 | 2002-05-14 | 98 | 1 | 1 | 43 | True |
| NBIS | var | 53 | 2011-07-28 | 52 | 0 | 1 | 8 | True |

## 9. Likidite ve oynaklık (bilgi; seçim için kullanılmadı)
| Sembol | medyan_dolar_hacim_1y_musd | yillik_vol_1y | gercek_fiyat_dogrulanamayan_orani | gercek_fiyat_lt5_orani_dogrulanan_icinde | gercek_fiyat_lt5_son_tarih |
|---|---|---|---|---|---|
| RKLB | 1459.7 | 0.92 | 0 | 0.31 | 2024-08-08 |
| HIMS | 478.2 | 0.881 | 0 | 0.078 | 2022-11-07 |
| SOFI | 1211.2 | 0.568 | 0 | 0.034 | 2023-05-19 |
| PLTR | 6030.5 | 0.586 | 0 | 0 |  |
| MU | 20257.2 | 0.801 | 0 | 0 |  |
| MRVL | 3450.4 | 0.777 | 0 | 0 |  |
| BE | 2166.9 | 1.103 | 0 | 0.032 | 2020-04-03 |
| TSLA | 22725.9 | 0.452 | 0 | 0 |  |
| AMD | 11288.7 | 0.671 | 0 | 0.044 | 2016-07-05 |
| NVDA | 30021.4 | 0.372 | 0 | 0 |  |
| ENPH | 204.9 | 0.799 | 0 | 0.242 | 2019-01-03 |
| SHOP | 1075 | 0.587 | 0 | 0 |  |
| ROKU | 287.4 | 0.402 | 0 | 0 |  |
| XYZ | 389.3 | 0.462 | 0 | 0 |  |
| SNAP | 230.7 | 0.575 | 0 | 0.026 | 2026-07-31 |
| UPST | 145.3 | 0.657 | 0 | 0 |  |
| SMCI | 1064.9 | 0.949 | 0 | 0 |  |
| PLUG | 172.9 | 0.788 | 0 | 0.68 | 2026-10-15 |
| PTON | 52.3 | 0.613 | 0 | 0.151 | 2026-10-08 |
| RIVN | 418.3 | 0.687 | 0 | 0 |  |
| AAPL | 12629.9 | 0.245 | 0 | 0 |  |
| MSFT | 12719.1 | 0.325 | 0 | 0 |  |
| GOOGL | 9495.2 | 0.312 | 0 | 0 |  |
| AMZN | 9821.4 | 0.337 | 0 | 0 |  |
| MRNA | 360.7 | 1.284 | 0 | 0 |  |
| JPM | 2629 | 0.223 | 0 | 0 |  |
| XOM | 2219 | 0.26 | 0 | 0 |  |
| WMT | 2360.2 | 0.264 | 0 | 0 |  |
| NBIS | 2341.6 | 1.068 | 0 | 0 |  |
| SPY | 42561.7 | 0.127 | 0 | 0 |  |
| QQQ | 30297.5 | 0.197 | 0 | 0 |  |
| IWM | 8078.9 | 0.183 | 0 | 0 |  |
| XLK | 1937.1 | 0.263 | 0 | 0 |  |
| SMH | 3677.9 | 0.395 | 0 | 0 |  |
| IGV | 1095.7 | 0.32 | 0 | 0 |  |
| XLC | 639.6 | 0.161 | 0 | 0 |  |
| XLY | 982.3 | 0.192 | 0 | 0 |  |
| XLF | 2034.6 | 0.147 | 0.066 | 0 |  |
| XLE | 1917.8 | 0.22 | 0 | 0 |  |
| XLV | 1667.3 | 0.161 | 0 | 0 |  |
| XLP | 1067.4 | 0.147 | 0 | 0 |  |
| XLI | 1502.1 | 0.172 | 0 | 0 |  |
| ITA | 167.2 | 0.231 | 0 | 0 |  |
| ICLN | 97.3 | 0.308 | 0 | 0 |  |
| UFO | 19 | 0.41 | 0 | 0 |  |

## 10. Kapsama matrisi (hisse × katman) — ÖN SAYIM
**Takvim bazlı ön sayımdır.** Etiket ve purge gerçek barlarla hesaplanmadan kesin eğitim uygunluğu veya katman başına uygun şirket sayısı ilan edilmez. `YETER` = ön sayımda ≥ eşik.

Hücre: `durum eğitim_örneği(H=20) / validation_sinyal_günü`. Isınma 200 seans, purge: etiket sonu (t+20) training sonunu geçemez. Ön eşik ≥ 252 örnek. Sayım XNYS seans takvimi üzerindedir; t+1…t+20 seanslarından biri eksik/bozuksa örnek düşer (`kapsama_uzun.csv` → `bosluk_kaybi_*`). TEST sütunu yalnız sinyal günü sayısıdır.

| symbol | K1 | K2 | K3 | K4 | TEST |
|---|---|---|---|---|---|
| RKLB | YOK 0 / 0 | YOK 0 / 141 | AZ 121 / 250 | YETER 371 / 252 | 448 sinyal |
| HIMS | YOK 0 / 40 | AZ 20 / 251 | YETER 271 / 250 | YETER 521 / 252 | 448 sinyal |
| SOFI | YOK 0 / 0 | YOK 0 / 201 | AZ 181 / 250 | YETER 431 / 252 | 448 sinyal |
| PLTR | YOK 0 / 117 | AZ 97 / 251 | YETER 348 / 250 | YETER 598 / 252 | 448 sinyal |
| MU | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| MRVL | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| BE | YETER 395 / 252 | YETER 647 / 251 | YETER 898 / 250 | YETER 1148 / 252 | 448 sinyal |
| TSLA | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| AMD | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| NVDA | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| ENPH | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| SHOP | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| ROKU | YETER 601 / 252 | YETER 853 / 251 | YETER 1104 / 250 | YETER 1354 / 252 | 448 sinyal |
| XYZ | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| SNAP | YETER 747 / 252 | YETER 999 / 251 | YETER 1250 / 250 | YETER 1500 / 252 | 448 sinyal |
| UPST | YOK 0 / 63 | AZ 43 / 251 | YETER 294 / 250 | YETER 544 / 252 | 448 sinyal |
| SMCI | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| PLUG | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| PTON | AZ 100 / 252 | YETER 352 / 251 | YETER 603 / 250 | YETER 853 / 252 | 448 sinyal |
| RIVN | YOK 0 / 0 | YOK 0 / 87 | AZ 67 / 250 | YETER 317 / 252 | 448 sinyal |
| AAPL | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| MSFT | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| GOOGL | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| AMZN | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| MRNA | YETER 301 / 252 | YETER 553 / 251 | YETER 804 / 250 | YETER 1054 / 252 | 448 sinyal |
| JPM | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| XOM | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| WMT | YETER 1039 / 252 | YETER 1291 / 251 | YETER 1542 / 250 | YETER 1792 / 252 | 448 sinyal |
| NBIS | EGITIM DISI 0 / 0 | EGITIM DISI 0 / 0 | EGITIM DISI 0 / 0 | EGITIM DISI 0 / 0 | 298 sinyal |

**Katman özeti**

| katman | train_end | Ana 20 | Kontrol 8 | İşlem 10 | Odaklı-20 örnek (H20) | Geniş-28 örnek (H20) | İşlem-10 val sinyal günü |
|---|---|---|---|---|---|---|---|
| K1 | 2020-12-31 | 13/20 | 8/8 | 6/10 | 12133 | 19707 | 1669 |
| K2 | 2021-12-31 | 14/20 | 8/8 | 6/10 | 15761 | 25351 | 2350 |
| K3 | 2022-12-31 | 17/20 | 8/8 | 8/10 | 20188 | 31786 | 2500 |
| K4 | 2023-12-31 | 20/20 | 8/8 | 10/10 | 25557 | 39155 | 2520 |

## 11. Tüm bulgular (sembol bazında)
HATA ve UYARI düzeyindekiler; BILGI satırları `bulgular.csv` içinde.

_Seansı bitmemiş son gün 1 sembolde temiz katmandan çıkarıldı._

**HIMS** (UYARI)
- [UYARI] `DONUK` 25 satır önceki günle birebir aynı (donmuş veri?)
- [UYARI] `ZERO_VOL` 63 işlem gününde hacim 0

**TSLA** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1673 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**AMD** (UYARI)
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**NVDA** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2122 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 28 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**SHOP** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1633 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 11 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**ROKU** (UYARI)
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**SNAP** (UYARI)
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**SMCI** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2200 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 2 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**PTON** (UYARI)
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**AAPL** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1173 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 12 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**GOOGL** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1645 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**AMZN** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1617 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 8 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**WMT** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2049 satırda raw_volume üretilmedi
- [UYARI] `BILANCO_BELIRSIZ` 1 geçmiş tarihli kayıtta sonuç bilgisi yok; gerçekleşmiş sayılmadı

**^VIX** (UYARI)
- [UYARI] `FAZLA_TARIH` 2 satır ABD seansı olmayan günde: ['2026-05-25', '2026-09-07']

**XLK** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2496 satırda raw_volume üretilmedi

**SMH** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1847 satırda raw_volume üretilmedi

**IGV** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2057 satırda raw_volume üretilmedi

**XLY** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2496 satırda raw_volume üretilmedi

**XLF** (UYARI)
- [UYARI] `KURUMSAL_ISLEM_DOGRULANMADI` 2016-09-19 öncesi 179 satırda tarihsel fiyat/hacim üretilmedi: XLRE pay dağıtımı; sağlayıcının 1.231 split kaydı sıradan hisse bölünmesi değildir. Olay öncesi tarihsel fiyat/hacim ayrıca doğrulanmalıdır. Kaynak: https://www.miaxglobal.com/sites/default/files/alert-files/XLF_Distribution_39575.pdf
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 179 satırda raw_volume üretilmedi

**XLE** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 2496 satırda raw_volume üretilmedi

**ITA** (UYARI)
- [UYARI] `RAW_HACIM_YOK` sağlayıcının hacim split yöntemi dış referansla doğrulanmadı (referans yok); son split öncesi 1241 satırda raw_volume üretilmedi
