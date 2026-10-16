# Quant ML — veri üreticisi

**1.1 · Yerel inceleme sürümü · 16 Ekim 2026**

Önce **`output/yahoo_20261016T121656Z_dataset_v1_1/INCELEME.html`** dosyasını çift tıklayıp açın.
Gerçek kayıt üzerinden üretilen tabloları ve kontrolleri internet olmadan inceleyebilirsiniz.

## Paket ne yapar?

Mevcut piyasa verisi projenizdeki **temiz günlük fiyat/hacim dosyalarını** okur.
Her hisse ve gün için 24 model girdisi ile 5, 10, 20 seanslık gerçekleşmiş getiri
hedeflerini üretir. Ham veriyi indiren veya temizleyen hattın yerine geçmez.

- Eski `data/clean/<snapshot>/` ve yeni `data/<snapshot>/clean/` düzenlerini tanır.
- Eski `quant_ml.data` paketine veya `market_data.py` dosyasının içe aktarılmasına bağlı değildir.
- Kaynak veriye yazmaz. İnternet kullanmaz. Var olan çıktı klasörünün üzerine yazmaz.
- Eksik, bozuk, doğrulanmamış veya geleceği bilinmeyen değerleri doldurmaz.
- Model eğitmez, işlem önermez, kârlılık iddiasında bulunmaz.
- Araç güncel main yapısına ayrı modül olarak eklenmiştir.

## Paketin içeriği

| Dosya/klasör | Amaç |
|---|---|
| `build_dataset.py` | Özellik ve etiket üretiminin tek giriş noktası |
| `dataset_builder/` | Kaynak uyarlama, hesaplama, raporlama ve doğrulama kodları |
| `output/yahoo_20261016T121656Z_dataset_v1_1/INCELEME.html` | İlk açılacak insan tarafından okunabilir rapor |
| `output/yahoo_20261016T121656Z_dataset_v1_1/dataset.parquet` | Bütün özellikler, hedefler, tarihler ve kalite durumları |
| `output/yahoo_20261016T121656Z_dataset_v1_1/features.parquet` | Gelecek hedefleri içermeyen özellik tablosu |
| `output/yahoo_20261016T121656Z_dataset_v1_1/latest.csv` | Son günün bütün hisseleri |
| `output/yahoo_20261016T121656Z_dataset_v1_1/sample_200.csv` | Hızlı inceleme için son 200 satır |
| `output/yahoo_20261016T121656Z_dataset_v1_1/coverage.csv` | Hisse başına hazır/eksik özellik sayıları |
| `output/yahoo_20261016T121656Z_dataset_v1_1/fold_counts.csv` | Eğitim/validation sayıları ve dönem sınırında çıkarılan örnekler |
| `output/yahoo_20261016T121656Z_dataset_v1_1/partition_membership.parquet` | Her dönem/ufuk için hangi hisse/günün kullanılabileceği |
| `output/yahoo_20261016T121656Z_dataset_v1_1/manifest.json` | Sürüm, ayarlar, model girdi listesi ve dosya bütünlük kayıtları |
| `tests/` | Formül, zaman sırası, veri bozukluğu ve iki kaynak düzeninin testleri |
| `validation/dataset_builder_v1_1/` | Test sonuç kaydı ve gerçek veri üzerinde bağımsız denetimler |
| `verify_dataset.py` | Üretilen dosyaları tekrar kontrol eder |

## Çalıştırma

Python **3.12 veya üstü** gerekir. Mevcut projenizin sanal ortamını kullanabilirsiniz.
Yeni bir ortam kullanıyorsanız önce proje bağımlılıklarını kurun:

```powershell
uv sync --group dev
```

Repo kökünde, proje yolunu kendinize göre değiştirerek çalıştırın:

```powershell
python build_dataset.py --project "C:\Users\user\Projects\quant-ml" --snapshot yahoo_20261016T121656Z --output "C:\Users\user\Projects\quant-ml\output\yeni_veri_seti"
```

Çıktı klasörü henüz var olmamalıdır. Paket hangi kaynak düzeninin kullanılacağını
otomatik seçer. Aynı snapshot her iki düzende de varsa `--layout main` veya
`--layout legacy` ile açıkça seçin. Yeni main üzerinde kaynak veri indirmesi ve
temizlemesi önceden `market_data.py` ile tamamlanmış olmalıdır.

Yalnız bir tarihe kadar bilinen satırları kullanmak için:

```powershell
python build_dataset.py --project "C:\Users\user\Projects\quant-ml" --snapshot yahoo_20261016T121656Z --as-of 2024-12-31 --output "C:\Users\user\Projects\quant-ml\output\2024_kesimi"
```

Kesim tarihinde hiç verisi veya eligible dönemi olmayan hisseler atlanır.
`manifest.json → skipped_symbols` içinde sembol ve gerekçe saklanır. Hiç uygun
hisse kalmazsa hata verilir. Eksik dosya veya bütünlük hatası atlanmaz.

Paket, snapshot'ın indirme zamanında henüz tamamlanmamış seansı almaz.
Son çıktının en yeni günü kaynak kaydının son tamamlanmış günüyle sınırlıdır.

## Maliyet ve sınıflar

**Varsayılan:** yalnız brüt sürekli getiri hedefi üretilir. Net getiri kolonları
boş kalır. İşlem maliyetleri ve yukarı/aşağı/nötr eşikleri sizin adınıza seçilmez.

Maliyetler biliniyorsa `--buy-bps` ve `--sell-bps` birlikte verilir.
1 baz puan (bps) = %0.01. Bu değerler her taraftaki toplam oransal maliyettir.
Net long getiri:

`(1 + brüt getiri) × (1 − satış_bps/10000) / (1 + alış_bps/10000) − 1`

İsteğe bağlı `--neutral-threshold` ondalık getiri birimindedir; örneğin 0.01 = %1.
Bu seçenek maliyetler açıkça verilmişse kullanılabilir. Eşik üstü +1, eksi eşik
altı −1, iki eşik arası 0 olur. Eşiğe tam eşit değer nötrdür.
−1 sınıfı aşağı yönlü hareketi belirtir; short işleminin kârı anlamına gelmez.
Bu parametrelerin örnek değerleri bir strateji tavsiyesi değildir.

Maliyet ve sınıf seçeneklerinin yazım biçimi:

```text
--buy-bps ALIS_MALIYETI --sell-bps SATIS_MALIYETI --neutral-threshold NOTR_ESIK
```

Bu ayarlar son test dönemi kullanılarak seçilmemelidir. Gerçek piyasa spread'i,
fiyat boşluğu, gerçekleşme ve short/ödünç alma maliyeti bu basit long modelinin dışındadır.

## 24 model girdisi

Tüm getiri ve mesafe oranları ondalıktır; 0.05 = %5. Pencereler takvim günü değil
XNYS işlem seansıdır. Volatilite yıllıklaştırılmıştır.

| Kolon grubu | Sayı | Hesap |
|---|---:|---|
| `return_1/5/10/20` | 4 | Düzeltilmiş kapanış / w seans önceki düzeltilmiş kapanış − 1 |
| `sma_distance_10/20/60` | 3 | Kapanış / bugünü içeren hareketli ortalama − 1 |
| `sma_slope_10/20/60_5` | 3 | Bugünkü ortalama / 5 seans önceki ortalama − 1 |
| `volatility_10/20` | 2 | w günlük log getirinin örnek standart sapması (ddof=1) × √252 |
| `range_ratio` | 1 | (High − Low) / Close |
| `gap_return` | 1 | Düzeltilmiş Open / önceki düzeltilmiş Close − 1 |
| `intraday_return` | 1 | Close / Open − 1 |
| `atr_14_ratio` | 1 | 14 seanslık basit ortalama true range / kapanış; Wilder ATR değil |
| `relative_spy/qqq_5/20` | 4 | Hisse getirisi − aynı tarihlerde benchmark getirisi |
| `market_spy/qqq_return_20` | 2 | Benchmark'ın 20 seans getirisi |
| `relative_volume_20` | 1 | Sağlayıcı volume / bugünden önceki 20 seansın sağlayıcı hacmi ortalaması |
| `volume_change_1` | 1 | Sağlayıcı volume / önceki seansın sağlayıcı hacmi − 1 |

Fiyatlar `adj_close / close` ile aynı düzeltilmiş ölçeğe alınır. Boyutsuz oranlar,
geçmişe aynı katsayıyla uygulanan fiyat düzeltmesinden etkilenmez. Bununla birlikte
sağlayıcının geçmiş veriyi sonradan revize etmesi için point-in-time garanti verilmez.

Hacim özellikleri yalnız sağlayıcının `volume` kolonundan hesaplanır; split
kaydı pencereyi sıfırlamaz. Bu iki oran, hesaplama penceresi boyunca tutarlı
split-düzeltilmiş ölçek varsayar. Bu, mutlak tarihsel hisse adedinin doğrulandığı
anlamına gelmez. `raw_volume` model girdisi değildir; ileride mutlak pay adedi
eşiği gerekirse ayrıca doğrulanmalıdır. Geçersiz, negatif veya eksik sağlayıcı
hacmi kullanılmaz; sıfır paydalı oranlar boş kalır.

Yeni main düzenindeki `symbol="*"` uyarıları `global_quality_issues` altında ve
raporda ayrı tutulur; tek başına bütün hisseleri UYARI yapmaz. Genel HATA
üretimi durdurur; hisseye özel uyarılar kendi hisselerinde korunur.

## Hedefler ve dönem sınırları

- Özellikler **t günü kapanışından sonra** bilinebilir. Aynı kapanıştan işlem varsayılmaz.
- Giriş **t+1 açılışı**, çıkış **t+H kapanışı**; H = evren ayarlarındaki 5/10/20 seans.
- Giriş fiyatı da çıkışla aynı düzeltilmiş fiyat ölçeğindedir.
- t ve t+1…t+H günlerinin tamamında geçerli, eligible fiyat gerekir.
- Bir eksik/bozuk seans varsa ilgili hedef boş kalır; ardışık mevcut satırlar üzerinden atlanmaz.
- Son H günün geleceği bilinmediği için hedefleri boş kalır; bu veri kaybı değildir.
- Eğitim etiketi eğitim sonunu, validation etiketi validation sonunu aşamaz.
- `partition_membership.parquet` yalnız bu kuralları geçen fiyat adayı satırlarını listeler.
- Bütün 24 girdi kullanılacaksa ayrıca `volume_features_ready=True` gerekir.

## Modelde neler kullanılmalı?

**X girdilerini yalnız `manifest.json` içindeki `feature_columns` listesinden seçin.**
`target_*`, `label_*`, `entry_date`, kalite ve adaylık bilgileri model girdisi değildir.
Eksik değerleri tek bir otomatik kuralla doldurmayın; eksik veri politikasını eğitim
dönemi üzerinde ayrı belirleyin. Ölçekleyiciyi ve doldurma istatistiklerini yalnız
eğitim verisinde öğrenin.

`price_sample_candidate`: yeterli geçmişi olan, 22 fiyat/piyasa özelliği hazır,
eğitim evreni üyesi ve OK/UYARI durumundaki satır. Isınma süresi evren ayarlarından
okunur (mevcut ayar: 200 önceki geçerli seans). SPAC öncesi geçmiş sayılmaz.
NBIS tabloda bulunur ama eğitim adayı değildir. UYARI ayrıca incelenmelidir.
`training_approved=False` bilinçli olarak korunur; adaylık nihai eğitim onayı değildir.

## Testleri ve çıktıyı tekrar kontrol etme

```powershell
uv sync --group dev
python -m pytest -q -c pytest.ini -p no:cacheprovider
python verify_dataset.py output/yahoo_20261016T121656Z_dataset_v1_1
```

Gerçek veride geçmiş hesapların gelecekteki satırlardan etkilenmediğini tekrar sınamak için:

```powershell
python validate_dataset.py --project "C:\Users\user\Projects\quant-ml" --snapshot yahoo_20261016T121656Z --report "kontrol_sonucu.json"
```

## Bu sürümün sınırı

Bu, veri/özellik/etiket üreticisinin inceleme sürümüdür. Eski ayrıntılı kalite kontrollerinin main'e geri eklenmesi, model eğitimi,
portföy backtesti ve gerçek zamanlı servis bu paketin parçası değildir.
Kaynak kalite raporuna güvenerek çalışır; ham kaynağın ekonomik doğruluğunu yeniden
kanıtlamaz. İki klasör düzeni için uyumluluk testleri, main'e merge yapıldığı anlamına gelmez.
