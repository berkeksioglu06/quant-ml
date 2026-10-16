"""2.957 hisse için 24 özellik + 5/10/20 seans hedefleri + K1–K4 bölmeleri (Git'e girmez).

Aynı tanım: dataset_builder.features.make_features ve labels.make_labels (v1.1).
Aday = price_sample_candidate (200 seans ısınma, 22 fiyat özelliği hazır) VE o gün in_universe.
Çıktı: <data>/features/part-*.parquet (bütün satırlar, 2025+ dahil; araştırma kodu test
başlangıcında keser), partition_membership.parquet, manifest_features.json.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(sys.argv[1])  # dataset_builder ve universe.json (worktree)
DATA = Path(sys.argv[2])  # genis_2957_<tarih>
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
sys.path.insert(0, str(REPO))

from dataset_builder.features import (
    FEATURES,
    PRICE_FEATURES,
    VOLUME_FEATURES,
    make_features,
)
from dataset_builder.labels import make_labels, partition_mask

UNIVERSE = json.loads((REPO / "universe.json").read_text(encoding="utf-8"))
HORIZONS = tuple(UNIVERSE["horizons"])
WARMUP = UNIVERSE["warmup_sessions"]
BENCH_DIR = REPO / "data" / "yahoo_20261016T121656Z" / "clean"


def one(sym):
    bench = {b: pd.read_parquet(BENCH_DIR / f"{b}.parquet") for b in ("SPY", "QQQ")}
    frame = pd.read_parquet(DATA / "clean" / f"{sym.replace('^', '_')}.parquet")
    if frame.index.name != "date":
        frame = frame.set_index("date")
    frame = frame.loc[frame["is_session"]]
    feats = make_features(
        frame, bench, sym, training_member=True, quality_status="OK", warmup_sessions=WARMUP
    )
    labels = make_labels(frame, sym, HORIZONS)
    out = feats.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
    inu = frame["in_universe"].reindex(out["date"]).fillna(False).astype(bool).to_numpy()
    out["in_universe"] = inu
    out["price_sample_candidate"] = out["price_sample_candidate"] & out["in_universe"]
    parts = []
    for fold in UNIVERSE["folds"]:
        for h in HORIZONS:
            for name, (a, b) in (
                ("train", (UNIVERSE["start"], fold["train_end"])),
                ("validation", (fold["val_start"], fold["val_end"])),
            ):
                mask = partition_mask(out, h, a, b)
                sel = out.loc[mask, ["date", "symbol", f"label_end_{h}"]].rename(
                    columns={f"label_end_{h}": "label_end"}
                )
                parts.append(sel.assign(fold=fold["name"], horizon=h, partition=name))
    return sym, out, pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    symbols = pd.read_csv(DATA / "evrene_girenler.csv")["symbol"].tolist()[:LIMIT]
    feat_dir = DATA / ("features" if LIMIT is None else "features_deneme")
    feat_dir.mkdir(exist_ok=False)
    members, errors, counts, buf, part = [], [], [], [], 0

    def flush():
        global part, buf
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(
                feat_dir / f"part-{part:03d}.parquet", index=False
            )
            part, buf = part + 1, []

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(one, s): s for s in symbols}
        for i, fut in enumerate(futures, 1):
            s = futures[fut]
            try:
                sym, out, mem = fut.result()
            except Exception as exc:  # noqa: BLE001 - tek hisse raporlanır
                errors.append({"symbol": s, "error": repr(exc)[:300]})
                continue
            buf.append(out)
            members.append(mem)
            counts.append(
                {
                    "symbol": sym,
                    "rows": len(out),
                    "candidates": int(out.price_sample_candidate.sum()),
                    "volume_ready": int(out.volume_features_ready.sum()),
                }
            )
            if len(buf) >= 100:
                flush()
            if i % 250 == 0:
                print(f"[{i}/{len(symbols)}]", flush=True)
    flush()
    mem = pd.concat(members, ignore_index=True)
    mem.to_parquet(
        DATA / ("partition_membership.parquet" if LIMIT is None else "partition_deneme.parquet"),
        index=False,
    )
    pd.DataFrame(counts).to_csv(DATA / "features_kapsam.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(DATA / "features_hatalar.csv", index=False)
    fold_counts = (
        mem.groupby(["fold", "horizon", "partition"])
        .agg(satir=("symbol", "size"), hisse=("symbol", "nunique"))
        .reset_index()
    )
    fold_counts.to_csv(DATA / "fold_counts.csv", index=False)
    (DATA / "manifest_features.json").write_text(
        json.dumps(
            {
                "builder": "dataset_builder v1.1 make_features / make_labels",
                "features": FEATURES,
                "price_features": PRICE_FEATURES,
                "volume_features": VOLUME_FEATURES,
                "horizons": HORIZONS,
                "warmup_sessions": WARMUP,
                "folds": UNIVERSE["folds"],
                "test_start": UNIVERSE["test"]["start"],
                "benchmarks": str(BENCH_DIR),
                "candidate": "price_sample_candidate (ısınma + 22 fiyat özelliği) VE in_universe",
                "symbols": len(counts),
                "errors": len(errors),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("hata:", len(errors))
    print(fold_counts.to_string(index=False))
