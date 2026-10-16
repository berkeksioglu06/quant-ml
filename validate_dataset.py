"""Gerçek kayıtta geçmişe bilgi sızmasını ve bağımsız formül sonuçlarını denetle."""

import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd

from dataset_builder.features import make_features
from dataset_builder.source import discover


def validate(project, snapshot, destination):
    source = discover(project, snapshot)
    benchmark = {name: source.load(name) for name in ("SPY", "QQQ")}
    prefix_checks, scalar_checks = 0, 0
    for info in source.universe["symbols"]:
        if info["asset"] != "equity":
            continue
        symbol = info["symbol"]
        frame = source.load(symbol)
        kwargs = {
            "training_member": bool(set(info["roles"]) & {"ana20", "kontrol8"}),
            "quality_status": source.statuses[symbol],
            "warmup_sessions": source.universe.get("warmup_sessions", 200),
        }
        full = make_features(frame, benchmark, symbol, **kwargs).set_index("date")
        for cutoff in ("2020-12-31", "2022-12-30", "2024-12-31", "2026-06-30"):
            part = frame.loc[:cutoff]
            if part.empty or not part.eligible.any():
                continue
            expected = make_features(
                part, {k: v.loc[:cutoff] for k, v in benchmark.items()}, symbol, **kwargs
            ).set_index("date")
            pd.testing.assert_frame_equal(full.loc[:cutoff], expected)
            prefix_checks += 1
        row = full.iloc[-1]
        date = full.index[-1]
        pos = frame.index.get_loc(date)
        for window in (1, 5, 10, 20):
            expected = frame.adj_close.iloc[pos] / frame.adj_close.iloc[pos - window] - 1
            if not math.isclose(row[f"return_{window}"], expected, abs_tol=1e-12, rel_tol=1e-10):
                raise ValueError(f"Skaler getiri kontrolü farklı: {symbol}")
            scalar_checks += 1
        for window in (10, 20, 60):
            expected = (
                frame.adj_close.iloc[pos]
                / statistics.mean(frame.adj_close.iloc[pos - window + 1 : pos + 1].to_list())
                - 1
            )
            if not math.isclose(
                row[f"sma_distance_{window}"], expected, abs_tol=1e-12, rel_tol=1e-10
            ):
                raise ValueError(f"Skaler ortalama kontrolü farklı: {symbol}")
            scalar_checks += 1
    source.check_unchanged()
    result = {
        "status": "passed",
        "snapshot": snapshot,
        "prefix_checks": prefix_checks,
        "independent_feature_checks": scalar_checks,
        "note": "Satır bazında geleceğe erişim denetimidir; sağlayıcı revizyonlarına karşı point-in-time garantisi değildir.",
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate(args.project, args.snapshot, args.report)
