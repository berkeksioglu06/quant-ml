"""Hazır veri setinden K1–K4 eğitim/validation tablolarını kurar.

İki hedef üretir:
- ``target_return_H``: hissenin t+1 açılışından t+H kapanışına getirisi (veri setinden).
- ``target_vs_qqq_H``: aynı pencerede hisse getirisi − QQQ getirisi.

QQQ getirisi ``dataset_builder.labels.make_labels`` ile aynı kuralla hesaplanır.
Satır seçimi ``partition_membership.parquet`` üzerinden yapılır; purge kuralı oradadır.
Test başlangıcı ve sonrasındaki satırlar okunduktan hemen sonra atılır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder.labels import make_labels
from dataset_builder.source import discover

BENCHMARK = "QQQ"


@dataclass(frozen=True)
class Panel:
    rows: pd.DataFrame
    features: list[str]
    targets: dict[str, str]
    folds: list[dict]
    horizon: int
    test_start: pd.Timestamp
    snapshot: str
    dataset_dir: Path
    dataset_manifest: dict

    def split(self, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        part = self.rows.loc[self.rows["fold"].eq(fold)]
        train = part.loc[part["partition"].eq("train")].reset_index(drop=True)
        val = part.loc[part["partition"].eq("validation")].reset_index(drop=True)
        if train.empty or val.empty:
            raise ValueError(f"{fold}: boş eğitim veya validation")
        return train, val


def benchmark_returns(project, snapshot, horizon, symbol=BENCHMARK):
    """Benchmark için hisselerle aynı giriş/çıkış kuralında H seanslık getiri."""
    source = discover(project, snapshot, "main")
    labels = make_labels(source.load(symbol), symbol, (horizon,))
    labels = labels.loc[labels[f"label_ready_{horizon}"]]
    return labels.set_index("date")[[f"label_end_{horizon}", f"target_return_{horizon}"]].rename(
        columns={
            f"label_end_{horizon}": "benchmark_label_end",
            f"target_return_{horizon}": "benchmark_return",
        }
    )


def load_panel(dataset_dir, project, horizon=20) -> Panel:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if horizon not in manifest["horizons"]:
        raise ValueError(f"Veri setinde {horizon} seanslık ufuk yok")
    features = list(manifest["feature_columns"])
    test_start = pd.Timestamp(manifest["test_start"])
    raw_target = f"target_return_{horizon}"
    columns = ["date", "symbol", *features, "volume_features_ready"]
    columns += [f"label_end_{horizon}", raw_target]
    data = pd.read_parquet(dataset_dir / "dataset.parquet", columns=columns)
    # Test dönemi kapalı: sonraki hiçbir adım bu satırları görmez.
    data = data.loc[data["date"] < test_start]

    membership = pd.read_parquet(dataset_dir / "partition_membership.parquet")
    membership = membership.loc[membership["horizon"].eq(horizon)]
    if membership["label_end"].max() >= test_start:
        raise ValueError("Bölme kayıtları test dönemine taşıyor")
    rows = membership.merge(data, on=["date", "symbol"], how="left", validate="many_to_one")
    if rows[raw_target].isna().any():
        raise ValueError("Bölme kaydında hedefi olmayan satır var")
    if not rows["label_end"].eq(rows[f"label_end_{horizon}"]).all():
        raise ValueError("Bölme kaydı ile veri setindeki etiket sonu uyuşmuyor")
    # Bütün 24 girdi kullanıldığı için hacim özellikleri de hazır olmalı.
    rows = rows.loc[rows["volume_features_ready"].eq(True)]
    if not np.isfinite(rows[features].to_numpy()).all():
        raise ValueError("Model girdilerinde eksik veya sonsuz değer var")

    bench = benchmark_returns(project, manifest["snapshot"], horizon)
    rows = rows.join(bench, on="date")
    if rows["benchmark_return"].isna().any():
        missing = rows.loc[rows["benchmark_return"].isna(), "date"].unique()[:5]
        raise ValueError(f"{BENCHMARK} getirisi eksik tarihler: {list(missing)}")
    if not rows["benchmark_label_end"].eq(rows["label_end"]).all():
        raise ValueError(f"{BENCHMARK} ve hisse etiket pencereleri uyuşmuyor")
    excess_target = f"target_vs_qqq_{horizon}"
    rows[excess_target] = rows[raw_target] - rows["benchmark_return"]

    keep = ["date", "symbol", "fold", "partition", "label_end", *features]
    keep += [raw_target, excess_target, "benchmark_return"]
    rows = rows[keep].sort_values(["fold", "partition", "date", "symbol"]).reset_index(drop=True)
    folds = [f for f in manifest["folds"] if f["name"] in set(rows["fold"])]
    return Panel(
        rows=rows,
        features=features,
        targets={"raw": raw_target, "vs_qqq": excess_target},
        folds=folds,
        horizon=horizon,
        test_start=test_start,
        snapshot=manifest["snapshot"],
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
    )
