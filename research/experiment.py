"""Aşama 1 deneyleri: aynı K1–K4 çerçevesinde kıyaslar + Ridge.

Deney 1: ham özellikler, hissenin kendi getirisi ve QQQ'ya göre getiri.
Deney 2: gün içi sıraya çevrilmiş özellikler, gün içi ortalamadan arındırılmış getiri;
Deney 1 Ridge'i referans satır olarak aynı tablolarda gösterilir.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder.common import sha256

from .data import BETA_WINDOW, REGIME_THRESHOLD, REGIMES, Panel
from .metrics import (
    daily_rank_ic,
    error_metrics,
    summarize_ic,
    top_k_mean,
    top_minus_all,
)
from .models import ALPHA_GRID, RidgeModel, baseline_models

TOP_K = 5
MAIN_MODEL = "ridge"
ERROR_BASELINES = ("sifir", "egitim_ortalamasi")
# Hüküm kuralı yalnız bu iki skor kıyasına bakar; diğerleri tanı kıyaslarıdır.
RANK_BASELINES = ("momentum_20", "geri_donus_5")
RISK_BASELINES = ("en_oynak_20", "yuksek_beta")
EVAL_COLUMNS = ["date", "symbol", "beta_252", "volatility_20", "beta_adj_return", "regime"]
REGISTRY_FIELDS = [
    "run_id",
    "created_at_utc",
    "experiment",
    "git_commit",
    "git_dirty",
    "snapshot",
    "horizon",
    "feature_set",
    "models",
    "target",
    "ridge_ic",
    "ridge_ic_t_nw",
    "ridge_r2",
]


@dataclass(frozen=True)
class Experiment:
    key: str
    name: str
    slug: str
    title: str
    targets: tuple[str, ...]
    feature_set: str
    reference: bool = False


EXPERIMENTS = {
    "1": Experiment(
        key="1",
        name="asama1_deney1_ridge",
        slug="ridge",
        title="Deney 1 — Ridge basit kıyasları geçiyor mu?",
        targets=("raw", "vs_qqq"),
        feature_set="raw",
    ),
    "2": Experiment(
        key="2",
        name="asama1_deney2_ridge_xs",
        slug="ridge_xs",
        title="Deney 2 — Gün içi hedef ve sıralanmış özelliklerle Ridge",
        targets=("xs",),
        feature_set="xs_rank",
        reference=True,
    ),
}


def make_models(experiment: Experiment, panel: Panel):
    features = panel.features if experiment.feature_set == "raw" else panel.xs_features
    models = [*baseline_models(), RidgeModel(features)]
    if experiment.reference:
        models.append(
            RidgeModel(panel.features, name="ridge_deney1", kind="score", target_key="raw")
        )
    return models


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _daily_diagnostics(frame):
    """Günlük rank IC, beta-düzeltmeli IC, ilk k hissenin getirisi/riski ve rejim."""
    beta_rows = frame.dropna(subset=["beta_adj_return"])
    return pd.concat(
        [
            daily_rank_ic(frame),
            daily_rank_ic(beta_rows, target_col="beta_adj_return").rename("rank_ic_beta"),
            top_minus_all(frame, TOP_K),
            top_k_mean(frame, "beta_252", TOP_K).rename("top_beta"),
            top_k_mean(frame, "volatility_20", TOP_K).rename("top_vol"),
            frame.groupby("date")["regime"].first(),
        ],
        axis=1,
    ).rename_axis("date")


def evaluate(panel: Panel, experiment: Experiment):
    """Deneyin bütün hedef × dönem × model kombinasyonlarını eğitir ve değerlendirir."""
    lags = panel.horizon - 1
    fold_rows, daily, preds, coefs, alphas, exposures = [], [], [], [], [], []
    for target_key in experiment.targets:
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            for model in make_models(experiment, panel):
                fit_target = panel.targets[getattr(model, "target_key", None) or target_key]
                model.fit(train, fit_target)
                pred = model.predict(val)
                ok = np.isfinite(pred)
                frame = val.loc[ok, EVAL_COLUMNS].assign(
                    pred=pred[ok], y=val.loc[ok, target].to_numpy()
                )
                row = {
                    "target": target_key,
                    "fold": name,
                    "model": model.name,
                    "kind": model.kind,
                    "n_train": len(train),
                    "n_val": len(val),
                    "coverage": float(ok.mean()),
                    "val_days": int(val["date"].nunique()),
                }
                if model.kind == "forecast":
                    row |= error_metrics(frame["y"], frame["pred"])
                if np.ptp(frame["pred"].to_numpy()) > 0:
                    d = _daily_diagnostics(frame)
                    row |= summarize_ic(d["rank_ic"], lags)
                    row["ic_beta_mean"] = float(d["rank_ic_beta"].mean())
                    row["top5_minus_all"] = float(d["top_minus_all"].mean())
                    daily.append(
                        d.reset_index().assign(target=target_key, fold=name, model=model.name)
                    )
                fold_rows.append(row)
                preds.append(
                    frame[["date", "symbol", "pred", "y"]].assign(
                        target=target_key, fold=name, model=model.name
                    )
                )
                if model.name == MAIN_MODEL:
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
                        val[model.features]
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

    Yalnız açıklama içindir, model seçiminde kullanılmaz. Günlük rank IC, hedefe aynı
    gün bütün hisselerde aynı olan bir sabit eklenmesinden (QQQ, gün ortalaması) ve
    özelliğin gün içi sıraya çevrilmesinden etkilenmez.
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


def pooled(result, panel: Panel, experiment: Experiment):
    """K1–K4 validation yıllarını birlikte özetler ve dönem tutarlılığını sayar."""
    lags = panel.horizon - 1
    folds, daily, preds = result["folds"], result["daily"], result["predictions"]
    rows = []
    for (target, model), part in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "kind": part["kind"].iloc[0]}
        p = preds.loc[preds["target"].eq(target) & preds["model"].eq(model)]
        row["coverage"] = len(p) / part["n_val"].sum()
        if row["kind"] == "forecast":
            row |= error_metrics(p["y"], p["pred"])
        d = daily.loc[daily["target"].eq(target) & daily["model"].eq(model)]
        if not d.empty:
            row |= summarize_ic(d.set_index("date")["rank_ic"], lags)
            by_regime = d.groupby("regime")["rank_ic"].mean()
            row |= {f"ic_{r}": float(by_regime.get(r, np.nan)) for r in REGIMES}
            row["ic_beta_mean"] = float(d["rank_ic_beta"].mean())
            row["ic_beta_t_nw"] = summarize_ic(d.set_index("date")["rank_ic_beta"], lags)["ic_t_nw"]
            row["top5_minus_all"] = float(d["top_minus_all"].mean())
            row["top5_beta"] = float(d["top_beta"].mean())
            row["top5_vol"] = float(d["top_vol"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)

    consistency = []
    for target in experiment.targets:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        ridge = f.loc[MAIN_MODEL]
        rec = {"target": target, "folds": len(ridge)}
        for base in ERROR_BASELINES:
            rec[f"mse_beats_{base}"] = int((ridge["mse"] < f.loc[base, "mse"]).sum())
        rec["ic_positive"] = int((ridge["ic_mean"] > 0).sum())
        for base in (*RANK_BASELINES, *RISK_BASELINES):
            rec[f"ic_beats_{base}"] = int((ridge["ic_mean"] > f.loc[base, "ic_mean"]).sum())
        consistency.append(rec)

    val = panel.rows.loc[panel.rows["partition"].eq("validation")]
    days = val.drop_duplicates(["fold", "date"])
    context = val.groupby("fold").agg(
        val_start=("date", "min"),
        val_end=("date", "max"),
        symbols=("symbol", "nunique"),
        stock_mean=(panel.targets["raw"], "mean"),
        excess_mean=(panel.targets["vs_qqq"], "mean"),
        beta_coverage=("beta_252", lambda s: s.notna().mean()),
    )
    context = context.join(days.groupby("fold")["benchmark_return"].mean().rename("qqq_mean"))
    regimes = days.groupby(["fold", "regime"]).size().unstack(fill_value=0)
    context = context.join(regimes.reindex(columns=list(REGIMES), fill_value=0).add_prefix("days_"))
    universe = val.groupby(["fold", "date"])[["beta_252", "volatility_20"]].mean()
    context = context.join(universe.groupby("fold").mean().add_prefix("universe_"))
    return summary, pd.DataFrame(consistency), context.reset_index()


def run(panel: Panel, output: Path, experiment: Experiment = EXPERIMENTS["1"]):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Çıktı klasörü zaten var: {output}")
    result = evaluate(panel, experiment)
    summary, consistency, context = pooled(result, panel, experiment)

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

    models = make_models(experiment, panel)
    main = next(m for m in models if m.name == MAIN_MODEL)
    config = {
        "experiment": experiment.name,
        "experiment_key": experiment.key,
        "title": experiment.title,
        "run_id": output.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "snapshot": panel.snapshot,
        "dataset_dir": str(panel.dataset_dir),
        "dataset_version": panel.dataset_manifest["version"],
        "dataset_manifest_sha256": sha256(panel.dataset_dir / "manifest.json"),
        "horizon": panel.horizon,
        "targets": {k: panel.targets[k] for k in experiment.targets},
        "benchmark": "QQQ",
        "feature_set": experiment.feature_set,
        "features": main.features,
        "folds": panel.folds,
        "test_start": str(panel.test_start.date()),
        "test_period_read": False,
        "models": [m.name for m in models],
        "reference_model": "ridge_deney1" if experiment.reference else None,
        "ridge": {
            "alpha_grid": list(ALPHA_GRID),
            "alpha_selection": "eğitimin son 252 seansı iç validation, purge'lü, MSE",
            "feature_winsor_quantile": 0.01,
            "target_winsor_quantile": 0.01,
        },
        "rank_ic": {"method": "günlük Spearman", "min_names": 5, "nw_lags": panel.horizon - 1},
        "diagnostics": {
            "beta": f"QQQ'ya karşı günlük getirilerle {BETA_WINDOW} seans, yalnız t'ye kadar",
            "regime_threshold": REGIME_THRESHOLD,
        },
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
    if registry.exists():
        with registry.open(encoding="utf-8") as stream:
            if next(csv.reader(stream), None) != REGISTRY_FIELDS:
                raise ValueError(f"{registry} kolonları beklenen biçimde değil")
    ridge = summary.loc[summary["model"].eq(MAIN_MODEL)].set_index("target")
    new = not registry.exists()
    with registry.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        if new:
            writer.writeheader()
        for target in config["targets"]:
            writer.writerow(
                {
                    "run_id": config["run_id"],
                    "created_at_utc": config["created_at_utc"],
                    "experiment": config["experiment"],
                    "git_commit": config["git_commit"],
                    "git_dirty": config["git_dirty"],
                    "snapshot": config["snapshot"],
                    "horizon": config["horizon"],
                    "feature_set": config["feature_set"],
                    "models": " ".join(config["models"]),
                    "target": target,
                    "ridge_ic": round(float(ridge.loc[target, "ic_mean"]), 5),
                    "ridge_ic_t_nw": round(float(ridge.loc[target, "ic_t_nw"]), 3),
                    "ridge_r2": round(float(ridge.loc[target, "r2_oos"]), 5),
                }
            )
