# Geniş evren referans script'leri

Geniş evren G1 hazırlığı için referans. 16 Ekim 2026'da 2.957 hisselik veriyi bu script'ler üretti.
Repo dışı yerel klasöre (`data/genis_2957_<tarih>/`, Git'e girmez) yazarlar. Kalıcı hali
`genis_evren.py`'ye `clean` ve `dataset` komutları olarak taşınacak (hazırlık belgesi, bölüm 1).

| Dosya | Ne yapar | Çalıştırma |
|---|---|---|
| `ref_build_clean.py` | Ham kopya + temiz günlük tablo + `in_universe` | `python ref_build_clean.py <repo> <hedef_klasör>` |
| `ref_build_dataset.py` | 24 özellik, 5/10/20 hedef, K1–K4 bölmeleri | `python ref_build_dataset.py <repo> <hedef_klasör> [N]` |
| `ref_crosscheck.py` | 28 hisselik eski veri setiyle satır satır karşılaştırma | `python ref_crosscheck.py <repo> <hedef_klasör>` |
| `beklenen_fold_counts.csv` | Tutması gereken fold × ufuk × bölme sayıları | — |
