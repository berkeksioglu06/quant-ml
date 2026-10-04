#!/usr/bin/env python3
"""Quant ML veri etabı.

  uv run python -m quant_ml.data all                    # indir + temizle + kalite + kapsama (önerilen)
  uv run python -m quant_ml.data download [--symbols RKLB NVDA] [--no-earnings]
  uv run python -m quant_ml.data build [--snapshot yahoo_20261004T150000Z]   # mevcut snapshot üzerinden yeniden üret

Çıktılar data/ altında:
  raw/<snapshot>/prices/*.parquet     sağlayıcı kopyası, değiştirilmez (salt-okunur)
  raw/<snapshot>/earnings/*.parquet   Yahoo bilanço tarihleri (denetim için)
  raw/<snapshot>/manifest.json        kaynak kaydı: zaman, seçenekler, sürümler, satır/tarih, sha256
  clean/<snapshot>/*.parquet          temiz katman + clean_log.json
  reports/<snapshot>/kalite_raporu.md kalite raporu + kapsama matrisi (+ CSV/JSON ekleri)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .clean import build_clean
from .common import ROOT, Paths, latest_snapshot, load_universe
from .coverage import coverage
from .download import download_snapshot
from .quality import run_quality
from .report import write_report


def build(data_dir: Path, snapshot: str | None, universe=None):
    snap = snapshot or latest_snapshot(data_dir).name
    paths = Paths(data_dir, snap)
    print(f"→ temiz katman: {paths.clean}")
    build_clean(paths, universe)
    print("→ kalite kontrolleri")
    q = run_quality(paths, universe)
    print("→ kapsama matrisi")
    cov = coverage(paths, universe)
    write_report(paths, q, cov, universe)
    R = q["results"]
    print(f"\nRapor: {paths.reports / 'kalite_raporu.md'}")
    print(
        f"OK {sum(r['durum'] == 'OK' for r in R.values())} · UYARI {sum(r['durum'] == 'UYARI' for r in R.values())} "
        f"· HATA {sum(r['durum'] == 'HATA' for r in R.values())}"
    )
    for s, r in R.items():
        for i in r["issues"]:
            if i["severity"] == "HATA":
                print(f"  HATA {s:<6} {i['code']}: {i['msg'][:120]}")
    print("\n" + cov["summary"].to_string(index=False))
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["all", "download", "build"])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--start")
    ap.add_argument("--snapshot")
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0, help="semboller arası bekleme (sn)")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    u = load_universe()

    if a.cmd in ("all", "download"):
        snap = download_snapshot(
            data_dir, a.symbols, a.start, earnings=not a.no_earnings, pause=a.pause, universe=u
        )
        a.snapshot = snap.name
    if a.cmd in ("all", "build"):
        build(data_dir, a.snapshot, u)


if __name__ == "__main__":
    main()
