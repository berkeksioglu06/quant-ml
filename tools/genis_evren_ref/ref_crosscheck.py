"""Geniş veri setindeki özellikler, orijinal 28 hisselik veri setiyle aynı satırlarda tutuyor mu?"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO, DATA = sys.argv[1], sys.argv[2]
ORIG = Path(REPO) / "output" / "yahoo_20261016T121656Z_dataset_v1_1"
orig = pd.read_parquet(ORIG / "dataset.parquet")
F = json.loads((ORIG / "manifest.json").read_text(encoding="utf-8"))["feature_columns"]
syms = set(orig.symbol)
wide = pd.concat(
    [
        pd.read_parquet(p, columns=["date", "symbol", *F, "target_return_20"])
        for p in glob.glob(f"{DATA}/features/part-*.parquet")
    ]
)
wide = wide[wide.symbol.isin(syms)]
m = orig.merge(wide, on=["date", "symbol"], suffixes=("_o", "_w"))
print(
    f"ortak satır: {len(m)} / orijinal {len(orig)} | ortak hisse {m.symbol.nunique()} / {len(syms)}"
)
rows = []
for c in [*F, "target_return_20"]:
    a, b = m[f"{c}_o"], m[f"{c}_w"]
    both = a.notna() & b.notna()
    same = np.isclose(a[both], b[both], rtol=1e-9, atol=1e-12)
    rows.append(
        {
            "kolon": c,
            "ikisinde_dolu": int(both.sum()),
            "ayni_oran": float(same.mean()),
            "max_fark": float((a[both] - b[both]).abs().max()),
            "sadece_orijinalde_dolu": int((a.notna() & b.isna()).sum()),
            "sadece_geniste_dolu": int((a.isna() & b.notna()).sum()),
        }
    )
r = pd.DataFrame(rows)
print(r.to_string(index=False))
bad = m.loc[~np.isclose(m.return_20_o, m.return_20_w, equal_nan=True)]
print("return_20 farklı satırların hisse dağılımı:", bad.symbol.value_counts().head(10).to_dict())
