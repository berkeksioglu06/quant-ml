"""Salt okunur kaynak -> tek sürümlü özellik/etiket çıktısı."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256
from .features import FEATURES, PRICE_FEATURES, VOLUME_FEATURES, make_features
from .labels import Costs, make_labels, partition_mask
from .report import write_review
from .source import discover


def build(
    project,
    snapshot,
    output,
    *,
    layout="auto",
    as_of=None,
    costs: Costs | None = None,
    neutral_threshold=None,
):
    source = discover(project, snapshot, layout)
    output = Path(output).resolve()
    # Kaynak veri ağacına, kaynakların üzerine veya proje köküne çıktı yazma.
    data_root = (source.project / "data").resolve()
    if (
        output == source.project
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("Çıktı kaynak data klasöründen ayrı bir klasör olmalı")
    if output.exists():
        raise ValueError("Çıktı klasörü zaten var; yeni bir çıktı adı seçin")
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        if pd.isna(as_of) or as_of.tzinfo is not None or as_of != as_of.normalize():
            raise ValueError("as_of YYYY-MM-DD biçiminde geçerli tarih olmalı")
    if neutral_threshold is not None:
        if not np.isfinite(neutral_threshold) or neutral_threshold < 0:
            raise ValueError("Nötr eşik sonlu ve sıfır veya pozitif olmalı")
        if costs is None:
            raise ValueError("Sınıf üretmek için alış/satış maliyetleri açıkça belirtilmeli")
    universe = source.universe
    equities = [s for s in universe["symbols"] if s["asset"] == "equity"]
    if not equities:
        raise ValueError("Evren içinde hisse yok")
    horizons = tuple(universe["horizons"])
    if not universe["folds"]:
        raise ValueError("En az bir kronolojik değerlendirme dönemi gerekli")
    if len({f["name"] for f in universe["folds"]}) != len(universe["folds"]):
        raise ValueError("Dönem adları tekil olmalı")
    for fold in universe["folds"]:
        if not (
            pd.Timestamp(universe["start"])
            <= pd.Timestamp(fold["train_end"])
            < pd.Timestamp(fold["val_start"])
            <= pd.Timestamp(fold["val_end"])
            < pd.Timestamp(universe["test"]["start"])
        ):
            raise ValueError(
                f"Kronolojik dönemler çakışıyor veya son teste taşıyor: {fold['name']}"
            )
    benchmarks = {b: source.load(b, as_of) for b in ("SPY", "QQQ")}
    frames, checks, skipped_symbols = [], [], []
    for info in equities:
        symbol = info["symbol"]
        frame = source.load(symbol, as_of)
        if frame.empty or not frame.eligible.any():
            skipped_symbols.append(
                {
                    "symbol": symbol,
                    "reason": "no_data_on_or_before_cutoff"
                    if frame.empty
                    else "no_eligible_data_on_or_before_cutoff",
                }
            )
            continue
        features = make_features(
            frame,
            benchmarks,
            symbol,
            training_member=bool(set(info["roles"]) & {"ana20", "kontrol8", "genis"}),
            quality_status=source.statuses[symbol],
            warmup_sessions=universe.get("warmup_sessions", 200),
        )
        labels = make_labels(frame, symbol, horizons, costs=costs)
        joined = features.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
        if joined.entry_date.isna().any():
            raise ValueError(f"{symbol}: özellik ve etiket günleri eşleşmedi")
        # Bağımsız skaler kontrol: ilk/orta/son olgun etiket, her hisse ve ufuk.
        for h in horizons:
            ready = joined.loc[joined[f"label_ready_{h}"]]
            positions = [0, len(ready) // 2, -1] if len(ready) else []
            for row in ready.iloc[positions].to_dict("records"):
                enter = frame.loc[row["entry_date"]]
                exit_price = frame.loc[row[f"label_end_{h}"], "adj_close"]
                expected = exit_price / (enter.open * enter.adj_close / enter.close) - 1
                error = abs(row[f"target_return_{h}"] - expected)
                if error > 1e-10:
                    raise ValueError(f"Bağımsız getiri kontrolü başarısız: {symbol}")
                checks.append(
                    {
                        "symbol": symbol,
                        "date": row["date"],
                        "horizon": h,
                        "actual": row[f"target_return_{h}"],
                        "reference": expected,
                    }
                )
            if neutral_threshold is not None:
                target = joined[f"target_long_net_{h}"]
                classes = pd.Series(pd.NA, index=joined.index, dtype="Int8")
                known = target.notna()
                classes.loc[known] = 0
                classes.loc[target.gt(neutral_threshold)] = 1
                classes.loc[target.lt(-neutral_threshold)] = -1
                joined[f"target_class_{h}"] = classes
        frames.append(joined)
    if not frames:
        raise ValueError("Seçilen tarihte uygun verisi olan hiç hisse kalmadı")
    dataset = (
        pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    )
    targets = [f"target_return_{h}" for h in horizons]
    net_targets = [f"target_long_net_{h}" for h in horizons]
    class_targets = [f"target_class_{h}" for h in horizons] if neutral_threshold is not None else []
    if dataset.duplicated(["date", "symbol"]).any():
        raise ValueError("Çift hisse/tarih satırı")
    if np.isinf(dataset[FEATURES + targets + net_targets].to_numpy()).any():
        raise ValueError("Sonsuz değer üretildi")
    if not dataset.price_features_ready.any():
        raise ValueError("Hiçbir satırda fiyat özellikleri tamamlanmadı")
    latest = dataset.loc[dataset.date.eq(dataset.date.max())]
    if latest[[f"label_ready_{h}" for h in horizons]].any().any():
        raise ValueError("Son günün geleceği biliniyormuş gibi etiketlendi")
    folds = []
    membership = []
    for fold in universe["folds"]:
        for h in horizons:
            train = partition_mask(dataset, h, universe["start"], fold["train_end"])
            val = partition_mask(dataset, h, fold["val_start"], fold["val_end"])
            all_features = dataset.volume_features_ready
            before = (
                dataset.date.between(
                    pd.Timestamp(universe["start"]), pd.Timestamp(fold["train_end"])
                )
                & dataset.price_sample_candidate
                & dataset[f"label_ready_{h}"]
            )
            folds.append(
                {
                    "fold": fold["name"],
                    "horizon": h,
                    "train_price_only": int(train.sum()),
                    "train_all_24": int((train & all_features).sum()),
                    "validation_price_only": int(val.sum()),
                    "validation_all_24": int((val & all_features).sum()),
                    "purged_train": int((before & ~train).sum()),
                }
            )
            for name, mask in (("train", train), ("validation", val)):
                selected = dataset.loc[mask, ["date", "symbol", f"label_end_{h}"]].copy()
                selected = selected.rename(columns={f"label_end_{h}": "label_end"})
                selected["fold"], selected["horizon"], selected["partition"] = fold["name"], h, name
                membership.append(selected)
    coverage = (
        dataset.groupby("symbol")
        .agg(
            rows=("date", "size"),
            first=("date", "min"),
            last=("date", "max"),
            valid_prices=("price_valid", "sum"),
            price_ready=("price_features_ready", "sum"),
            volume_ready=("volume_features_ready", "sum"),
            price_candidates=("price_sample_candidate", "sum"),
            quality=("quality_status", "first"),
        )
        .reset_index()
    )
    availability = []
    for symbol, part in dataset.groupby("symbol"):
        for h in horizons:
            row = {"symbol": symbol, "horizon": h}
            row.update(part[f"label_status_{h}"].value_counts().to_dict())
            availability.append(row)
    source.check_unchanged()
    output.mkdir(parents=True)
    dataset.to_parquet(output / "dataset.parquet", index=False)
    feature_table = dataset[
        [
            "date",
            "symbol",
            *FEATURES,
            "price_valid",
            "warmup_ready",
            "price_features_ready",
            "volume_features_ready",
            "price_sample_candidate",
            "quality_status",
        ]
    ]
    feature_table.to_parquet(output / "features.parquet", index=False)
    latest.to_csv(output / "latest.csv", index=False, encoding="utf-8-sig")
    dataset.tail(200).to_csv(output / "sample_200.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(output / "fold_counts.csv", index=False, encoding="utf-8-sig")
    pd.concat(membership, ignore_index=True).to_parquet(
        output / "partition_membership.parquet", index=False
    )
    pd.DataFrame(availability).fillna(0).to_csv(
        output / "label_availability.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(checks).to_csv(output / "numeric_checks.csv", index=False, encoding="utf-8-sig")
    dataset[FEATURES].isna().mean().rename("missing_fraction").to_csv(
        output / "missingness.csv", encoding="utf-8-sig"
    )
    manifest = {
        "version": "1.1-review",
        "status": "review_ready",
        "training_approved": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot": snapshot,
        "layout": source.layout,
        "as_of": str(as_of.date()) if as_of is not None else None,
        "rows": len(dataset),
        "symbols": int(dataset.symbol.nunique()),
        "skipped_symbols": skipped_symbols,
        "global_quality_issues": source.global_issues,
        "last_date": str(dataset.date.max().date()),
        "feature_columns": FEATURES,
        "price_feature_columns": PRICE_FEATURES,
        "volume_feature_columns": VOLUME_FEATURES,
        "gross_target_columns": targets,
        "net_long_target_columns": net_targets,
        "class_target_columns": class_targets,
        "horizons": horizons,
        "costs": asdict(costs) if costs is not None else None,
        "neutral_threshold": neutral_threshold,
        "numeric_checks_passed": len(checks),
        "partition_checks": len(folds),
        "folds": universe["folds"],
        "test_start": universe["test"]["start"],
        "universe_start": universe["start"],
        "source_sha256": source.source_files,
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "rules": {
            "entry": "t+1 open",
            "exit": "t+H close",
            "warmup_prior_valid_sessions": universe.get("warmup_sessions", 200),
            "volume": "provider volume ratios; assumes consistent split-adjusted scale across each window; no split reset; not absolute historical share counts",
            "price": "dimensionless adjusted OHLC ratios; post-close information",
        },
    }
    write_review(output, manifest, coverage, pd.DataFrame(folds), latest)
    manifest["output_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    source.check_unchanged()
    # Başarı manifesti en son yazılır; bu dosya yoksa üretim tamamlanmamıştır.
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
