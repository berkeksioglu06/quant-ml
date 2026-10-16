"""Deney 1: H seans, iki hedef, dört kıyas + Ridge, K1–K4."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder.common import sha256

from .data import Panel
from .metrics import daily_rank_ic, error_metrics, summarize_ic, top_minus_all
from .models import ALPHA_GRID, RidgeModel, baseline_models

TOP_K = 5


def make_models(features):
    return [*baseline_models(), RidgeModel(features)]


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def evaluate(panel: Panel):
    """Bütün hedef × dönem × model kombinasyonlarını eğitir ve değerlendirir."""
    lags = panel.horizon - 1
    fold_rows, daily, preds, coefs, alphas, exposures = [], [], [], [], [], []
    for target_key, target in panel.targets.items():
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            for model in make_models(panel.features):
                model.fit(train, target)
                pred = model.predict(val)
                frame = val[["date", "symbol"]].assign(pred=pred, y=val[target].to_numpy())
                row = {
                    "target": target_key,
                    "fold": name,
                    "model": model.name,
                    "kind": model.kind,
                    "n_train": len(train),
                    "n_val": len(val),
                    "val_days": int(val["date"].nunique()),
                }
                if model.kind == "forecast":
                    row |= error_metrics(frame["y"], frame["pred"])
                if np.ptp(pred) > 0:
                    ic = daily_rank_ic(frame)
                    spread = top_minus_all(frame, TOP_K)
                    row |= summarize_ic(ic, lags)
                    row["top5_minus_all"] = float(spread.mean())
                    daily.append(
                        pd.concat([ic, spread], axis=1)
                        .reset_index()
                        .assign(target=target_key, fold=name, model=model.name)
                    )
                fold_rows.append(row)
                preds.append(frame.assign(target=target_key, fold=name, model=model.name))
                if isinstance(model, RidgeModel):
                    coefs.append(
                        model.coefficients()
                        .rename("coef")
                        .rename_axis("feature")
                        .reset_index()
                        .assign(target=target_key, fold=name)
                    )
                    alphas.append(
                        {
                            "target": target_key,
                            "fold": name,
                            "alpha": model.alpha_,
                            "alpha_at_grid_edge": model.alpha_
                            in (min(ALPHA_GRID), max(ALPHA_GRID)),
                            **{f"inner_mse_{a:g}": m for a, m in model.inner_mse_.items()},
                        }
                    )
                    exposures.append(
                        val[panel.features]
                        .corrwith(pd.Series(pred, index=val.index))
                        .rename("corr")
                        .rename_axis("feature")
                        .reset_index()
                        .assign(target=target_key, fold=name)
                    )
    return {
        "folds": pd.DataFrame(fold_rows),
        "daily": pd.concat(daily, ignore_index=True),
        "predictions": pd.concat(preds, ignore_index=True),
        "coefficients": pd.concat(coefs, ignore_index=True),
        "alphas": pd.DataFrame(alphas),
        "exposures": pd.concat(exposures, ignore_index=True),
        "feature_ic": feature_ic(panel),
    }


def feature_ic(panel: Panel):
    """Her özelliğin tek başına günlük rank IC'si; eğitim ve validation ayrı.

    Yalnız açıklama içindir, model seçiminde kullanılmaz. Aynı gün QQQ getirisi bütün
    hisselerde aynı olduğundan günlük rank IC iki hedefte birebir aynıdır.
    """
    target = panel.targets["raw"]
    rows = []
    for fold in panel.folds:
        for partition, frame in zip(("train", "validation"), panel.split(fold["name"])):
            for feature in panel.features:
                ic = daily_rank_ic(frame[["date"]].assign(pred=frame[feature], y=frame[target]))
                rows.append(
                    {
                        "fold": fold["name"],
                        "partition": partition,
                        "feature": feature,
                        "ic_mean": float(ic.mean()),
                    }
                )
    return pd.DataFrame(rows)


def pooled(result, panel: Panel):
    """K1–K4 validation yıllarını birlikte özetler ve dönem tutarlılığını sayar."""
    lags = panel.horizon - 1
    folds, daily, preds = result["folds"], result["daily"], result["predictions"]
    rows = []
    for (target, model), part in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "kind": part["kind"].iloc[0]}
        p = preds.loc[preds["target"].eq(target) & preds["model"].eq(model)]
        if row["kind"] == "forecast":
            row |= error_metrics(p["y"], p["pred"])
        d = daily.loc[daily["target"].eq(target) & daily["model"].eq(model)]
        if not d.empty:
            row |= summarize_ic(d.set_index("date")["rank_ic"], lags)
            row["top5_minus_all"] = float(d["top_minus_all"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)

    consistency = []
    for target in panel.targets:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        ridge = f.loc["ridge"]
        rec = {"target": target, "folds": len(ridge)}
        for base in ("sifir", "egitim_ortalamasi"):
            rec[f"mse_beats_{base}"] = int((ridge["mse"] < f.loc[base, "mse"]).sum())
        rec["ic_positive"] = int((ridge["ic_mean"] > 0).sum())
        for base in ("momentum_20", "geri_donus_5"):
            rec[f"ic_beats_{base}"] = int((ridge["ic_mean"] > f.loc[base, "ic_mean"]).sum())
        consistency.append(rec)

    context = (
        panel.rows.loc[panel.rows["partition"].eq("validation")]
        .groupby("fold")
        .agg(
            val_start=("date", "min"),
            val_end=("date", "max"),
            symbols=("symbol", "nunique"),
            stock_mean=(panel.targets["raw"], "mean"),
            excess_mean=(panel.targets["vs_qqq"], "mean"),
        )
    )
    qqq = (
        panel.rows.drop_duplicates(["fold", "partition", "date"])
        .loc[lambda x: x["partition"].eq("validation")]
        .groupby("fold")["benchmark_return"]
        .mean()
        .rename("qqq_mean")
    )
    context = context.join(qqq).reset_index()
    return summary, pd.DataFrame(consistency), context


def run(panel: Panel, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Çıktı klasörü zaten var: {output}")
    result = evaluate(panel)
    summary, consistency, context = pooled(result, panel)

    output.mkdir(parents=True)
    result["folds"].to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "pooled_metrics.csv", index=False)
    consistency.to_csv(output / "consistency.csv", index=False)
    context.to_csv(output / "period_context.csv", index=False)
    result["coefficients"].to_csv(output / "ridge_coefficients.csv", index=False)
    result["alphas"].to_csv(output / "ridge_alpha.csv", index=False)
    result["exposures"].to_csv(output / "ridge_exposure.csv", index=False)
    result["feature_ic"].to_csv(output / "feature_ic.csv", index=False)
    result["daily"].to_parquet(output / "daily_ic.parquet", index=False)
    result["predictions"].to_parquet(output / "predictions.parquet", index=False)

    dataset_manifest = panel.dataset_dir / "manifest.json"
    config = {
        "experiment": "asama1_deney1_ridge",
        "run_id": output.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "snapshot": panel.snapshot,
        "dataset_dir": str(panel.dataset_dir),
        "dataset_version": panel.dataset_manifest["version"],
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "horizon": panel.horizon,
        "targets": panel.targets,
        "benchmark": "QQQ",
        "features": panel.features,
        "folds": panel.folds,
        "test_start": str(panel.test_start.date()),
        "test_period_read": False,
        "models": [m.name for m in make_models(panel.features)],
        "ridge": {
            "alpha_grid": list(ALPHA_GRID),
            "alpha_selection": "eğitimin son 252 seansı iç validation, purge'lü, MSE",
            "feature_winsor_quantile": 0.01,
            "target_winsor_quantile": 0.01,
        },
        "rank_ic": {"method": "günlük Spearman", "min_names": 5, "nw_lags": panel.horizon - 1},
        "top_k": TOP_K,
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from .report import write_report

    write_report(output, config, result, summary, consistency, context)
    _register(output, config, summary)
    return config, summary, consistency


def _register(output, config, summary):
    """Denenen her kombinasyon output/experiments/runs.csv dosyasına eklenir."""
    registry = output.parent / "runs.csv"
    ridge = summary.loc[summary["model"].eq("ridge")].set_index("target")
    record = {
        "run_id": config["run_id"],
        "created_at_utc": config["created_at_utc"],
        "experiment": config["experiment"],
        "git_commit": config["git_commit"],
        "git_dirty": config["git_dirty"],
        "snapshot": config["snapshot"],
        "horizon": config["horizon"],
        "models": " ".join(config["models"]),
    }
    for target in config["targets"]:
        record[f"ridge_ic_{target}"] = round(float(ridge.loc[target, "ic_mean"]), 5)
        record[f"ridge_r2_{target}"] = round(float(ridge.loc[target, "r2_oos"]), 5)
    new = not registry.exists()
    with registry.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if new:
            writer.writeheader()
        writer.writerow(record)
