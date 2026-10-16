# Geniş evren listeleri (16 Ekim 2026)

`genis_evren.py` ile üretildi. Snapshot: `genis_yahoo_20261016T191133Z`, son tamamlanmış seans 2026-10-15.
Ham ve günlük veri (`data/genis_yahoo_*`) Git'e girmez. Bu klasörde yalnız listeler var.

Kural ([kural.json](kural.json)): her gün, yalnız o güne kadarki veriyle, gerçek fiyat > 5$,
60 seans medyan dolar hacmi > 10 M$ ve en az 252 seans geçmiş.

| Dosya | Satır | İçerik |
|---|---:|---|
| `evrene_girenler.csv` | 2.957 | 2016'dan beri en az bir gün evrene girenler; ilk/son giriş tarihi ve evrende gün sayısı |
| `bugun_evren_20261015.csv` | 2.278 | Son gün evrendekiler; fiyat, medyan dolar hacmi, oynaklık (bilgi) |

Aday havuzu 5.600 hisseydi; hiçbir gün kuralı geçemeyen ~2.640 hisse listelenmedi
(`genis_evren.py symbols` havuzu yeniden üretir).
| `EVREN.md` | | Özet rapor: yıllara göre evren büyüklüğü, mevcut 29 hissenin durumu |

İndirilemeyen: CTY (Yahoo boş tablo). Temizlik sonrası boş kalan: TCXB.

Bilinen sınırlar:
- Aday havuzu bugün işlem gören hisseler. Batmış veya satın alınmış hisseler yok (survivorship).
  Örnek: CFLT, CYBR, EA, EXAS, LC, PSTG, QRVO ve ZI bugünkü dizinde yok.
- SPAC'larda Yahoo birleşme öncesi kabuk dönemini de geçmiş olarak verir. Bu dönem 252 seanslık
  geçmiş sayımına giriyor (ör. HIMS, ASTS, VRT birleşmeden hemen sonra evrene giriyor). Düzeltme bekliyor.

Yeniden üretmek için:

```bash
uv run python genis_evren.py symbols
uv run python genis_evren.py download
uv run python genis_evren.py membership
```
