"""2.957 hisse için ham kopya + temiz günlük tablo (repo dışı, data/ altında; Git'e girmez)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

GENIS = Path(sys.argv[1])  # worktree: genis_evren.py ve snapshot burada
TARGET = Path(sys.argv[2])  # ana proje data/genis_2957_<tarih>
sys.path.insert(0, str(GENIS))
import genis_evren as g
import market_data as md

snap = GENIS / "data" / "genis_yahoo_20261016T191133Z"
manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
last_session = pd.Timestamp(manifest["last_complete_session"])
ever = pd.read_csv(GENIS / "evren" / "evrene_girenler.csv")
symbols = ever["symbol"].tolist()
member = pd.read_parquet(snap / "membership.parquet")
member = member.loc[member["symbol"].isin(symbols)]

(TARGET / "raw").mkdir(parents=True, exist_ok=False)
(TARGET / "clean").mkdir()
sub = {
    "source": manifest["source"],
    "created_at": manifest["created_at"],
    "last_complete_session": manifest["last_complete_session"],
    "start": manifest["start"],
    "options": manifest["options"],
    "rule": g.RULE,
    "from_snapshot": snap.name,
    "symbols": {},
}
by_symbol = {s: x.set_index("date") for s, x in member.groupby("symbol")}
for i, sym in enumerate(symbols, 1):
    rec = manifest["symbols"][sym]
    src = snap / "raw" / f"{md.safe(sym)}.parquet"
    if md.sha256(src) != rec["sha256"]:
        raise SystemExit(f"checksum uyuşmuyor: {sym}")
    dst = TARGET / "raw" / src.name
    shutil.copy2(src, dst)
    daily, issues = md.to_daily(pd.read_parquet(dst), last_session)
    sessions = md.CAL.sessions_in_range(daily.index.min(), daily.index.max()).tz_localize(None)
    daily["is_session"] = daily.index.isin(sessions)
    daily["eligible"] = True  # gerçek işlem satırları; aday olup olmama in_universe'te
    daily["raw_close"] = daily["close"] * daily["split_factor"]
    m = by_symbol[sym][["dollar_volume_med", "history_sessions", "volatility", "in_universe"]]
    daily = daily.join(m)
    daily["in_universe"] = daily["in_universe"].fillna(False).astype(bool)
    daily.to_parquet(TARGET / "clean" / src.name)
    sub["symbols"][sym] = rec | {
        "clean_rows": len(daily),
        "clean_sha256": md.sha256(TARGET / "clean" / src.name),
        "issues": [list(x) for x in issues],
    }
    if i % 500 == 0:
        print(f"[{i}/{len(symbols)}]", flush=True)
member.to_parquet(TARGET / "membership.parquet", index=False)
(TARGET / "manifest.json").write_text(json.dumps(sub, indent=2, default=str), encoding="utf-8")
shutil.copy2(GENIS / "evren" / "evrene_girenler.csv", TARGET / "evrene_girenler.csv")
shutil.copy2(GENIS / "evren" / "bugun_evren_20261015.csv", TARGET / "bugun_evren_20261015.csv")
(TARGET / "README.md").write_text(
    "# Geniş evren verisi (2.957 hisse) — Git'e girmez\n\n"
    f"Kaynak snapshot: `{snap.name}` (son seans {manifest['last_complete_session']}).\n\n"
    "- `raw/`: yfinance tablosu, kaynak snapshot'tan kopya (sha256 manifest'te).\n"
    "- `clean/`: `market_data.to_daily` sonrası günlük tablo + `is_session`, `eligible` (bütün gerçek\n"
    "  işlem satırları), `raw_close` (split'e göre düzeltilmemiş), kural ölçüleri ve `in_universe`.\n"
    "  Eğitim adayı = `in_universe` True olan satırlar. Özellik pencereleri bütün satırlarla hesaplanır.\n"
    "- `membership.parquet`: date × symbol kural ölçüleri.\n\n"
    "Bilinen sınırlar: survivorship (bugün işlem görenler); SPAC kabuk dönemi geçmiş sayımına giriyor.\n",
    encoding="utf-8",
)
print("tamam:", len(symbols), "hisse →", TARGET)
