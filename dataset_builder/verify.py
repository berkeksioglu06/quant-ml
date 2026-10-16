"""Üretilmiş paketi kaynak veri olmadan tekrar denetle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256


def verify_output(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "review_ready":
        raise ValueError("Üretim başarı kaydı yok")
    for name, digest in manifest["output_sha256"].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Çıktı bütünlüğü doğrulanamadı: {name}")
    dataset = pd.read_parquet(directory / "dataset.parquet")
    features = pd.read_parquet(directory / "features.parquet")
    columns = manifest["feature_columns"]
    if len(columns) != 24 or len(set(columns)) != 24:
        raise ValueError("24 tekil model girdisi gerekli")
    if len(dataset) != manifest["rows"] or dataset.duplicated(["date", "symbol"]).any():
        raise ValueError("Satır sayısı veya hisse/tarih tekilliği bozuk")
    if any(c.startswith(("target_", "label_")) or c == "entry_date" for c in features):
        raise ValueError("Özellik dosyasına gelecek bilgi karışmış")
    pd.testing.assert_frame_equal(
        features[["date", "symbol", *columns]], dataset[["date", "symbol", *columns]]
    )
    if np.isinf(
        dataset[
            columns + manifest["gross_target_columns"] + manifest["net_long_target_columns"]
        ].to_numpy()
    ).any():
        raise ValueError("Sonsuz değer")
    for h in manifest["horizons"]:
        ready = dataset[f"label_ready_{h}"]
        if not dataset.loc[ready, f"target_return_{h}"].notna().all():
            raise ValueError("Hazır etikette hedef boş")
        if dataset.loc[~ready, f"target_return_{h}"].notna().any():
            raise ValueError("Hazır olmayan etikette hedef üretilmiş")
    membership = pd.read_parquet(directory / "partition_membership.parquet")
    folds = {f["name"]: f for f in manifest["folds"]}
    for (fold, horizon, partition), group in membership.groupby(["fold", "horizon", "partition"]):
        settings = folds[fold]
        start = manifest["universe_start"] if partition == "train" else settings["val_start"]
        end = settings["train_end"] if partition == "train" else settings["val_end"]
        if not group.date.between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError("Sinyal tarihi dönem dışında")
        if group.label_end.gt(pd.Timestamp(end)).any():
            raise ValueError("Etiket dönem sınırını aşıyor")
        if partition == "train" and group.date.ge(pd.Timestamp(manifest["test_start"])).any():
            raise ValueError("Son test dönemi eğitime karışmış")
    return {
        "status": "verified",
        "files": len(manifest["output_sha256"]),
        "rows": len(dataset),
        "symbols": int(dataset.symbol.nunique()),
    }
