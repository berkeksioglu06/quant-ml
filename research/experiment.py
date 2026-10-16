"""Aşama 1 deneyleri: aynı K1–K4 çerçevesinde kıyaslar + ana model.

Deney 1: Ridge, ham özellikler, hissenin kendi getirisi ve QQQ'ya göre getiri.
Deney 2: gün içi sıraya çevrilmiş özellikler, gün içi ortalamadan arındırılmış getiri;
Deney 1 Ridge'i referans satır olarak aynı tablolarda gösterilir.
Deney 3: Deney 2 ile aynı; tek fark hedef: hisse − beta_t × QQQ, gün içi arındırılmış.
Yalnız betası olan satırlar; Ridge skorunun beta-nötrlenmiş hali ek satırdır.
Deney 4a: Deney 1 ile aynı (hissenin kendi getirisi, ham 24 özellik); tek fark model:
sabit ayarlı LightGBM.
Deney 4b: 4a ile aynı; tek fark 6 piyasa durumu özelliği. Yalnız bu özellikleri eksiksiz
olan satırlar; 4a ayarlı LightGBM aynı satırlarda "lightgbm_4a" referansı olarak eğitilir.
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

from .data import (
    BETA_MIN_OBS,
    BETA_WINDOW,
    REGIME_THRESHOLD,
    REGIMES,
    Panel,
    market_coverage,
)
from .metrics import (
    daily_rank_ic,
    error_metrics,
    neutralize,
    newey_west_tstat,
    summarize_ic,
    top_k_mean,
    top_minus_all,
)
from .models import (
    ALPHA_GRID,
    LGBM_EARLY_STOPPING,
    LGBM_MAX_TREES,
    LGBM_PARAMS,
    LGBM_SEED,
    TREE_CURVE,
    LGBMModel,
    RidgeModel,
    baseline_models,
)

TOP_K = 5
MAIN_MODEL = "ridge"
ERROR_BASELINES = ("sifir", "egitim_ortalamasi")
# Hüküm kuralı yalnız bu iki skor kıyasına bakar; diğerleri tanı kıyaslarıdır.
RANK_BASELINES = ("momentum_20", "geri_donus_5")
RISK_BASELINES = ("en_oynak_20", "yuksek_beta")
REFERENCE_MODEL = "ridge_deney1"
LGBM_REFERENCE = "lightgbm_4a"
EVAL_COLUMNS = ["date", "symbol", "beta_252", "volatility_20", "beta_adj_return", "regime"]
NEUTRAL_MODEL = "ridge_beta_notr"
PD_FEATURE = "volatility_20"
PD_CONDITION = "market_qqq_return_20"
PD_MARKET_CONDITION = "market_qqq_sma200_dist"
PD_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
# İlk sürümdeki kolonlar korunur (eski kayıtlar ve okuyan script'ler için); ana model
# kolonları sona eklenir. ridge_* yalnız ana model Ridge ise doludur.
LEGACY_REGISTRY_FIELDS = [
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
REGISTRY_FIELDS = [*LEGACY_REGISTRY_FIELDS, "main_model", "main_ic", "main_ic_t_nw", "main_r2"]


@dataclass(frozen=True)
class Experiment:
    key: str
    name: str
    slug: str
    title: str
    targets: tuple[str, ...]
    feature_set: str
    reference: bool = False
    universe: str = "all"
    beta_neutral_row: bool = False
    criteria: str | None = None
    main_model: str = MAIN_MODEL
    lgbm_reference: bool = False
    label: str | None = None


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
    "3": Experiment(
        key="3",
        name="asama1_deney3_ridge_resid",
        slug="ridge_resid",
        title="Deney 3 — Risk-nötr sıralama: betadan arındırılmış hedefle Ridge",
        targets=("resid",),
        feature_set="xs_rank",
        reference=True,
        universe="beta",
        beta_neutral_row=True,
        criteria="risk_notr",
    ),
    "4a": Experiment(
        key="4a",
        name="asama1_deney4a_lightgbm",
        slug="lgbm",
        title="Deney 4a — LightGBM, hissenin kendi getirisi",
        targets=("raw",),
        feature_set="raw",
        reference=True,
        criteria="lgbm_4a",
        main_model="lightgbm",
    ),
    "4b": Experiment(
        key="4b",
        name="asama1_deney4b_lightgbm_market",
        slug="lgbm_market",
        title="Deney 4b — LightGBM + piyasa durumu özellikleri",
        targets=("raw",),
        feature_set="raw_market",
        reference=True,
        universe="market",
        criteria="lgbm_4a",
        main_model="lightgbm",
        lgbm_reference=True,
        label="LightGBM 4b",
    ),
}


def model_features(experiment: Experiment, panel: Panel):
    if experiment.feature_set == "raw_market":
        return [*panel.features, *panel.market_features]
    return panel.features if experiment.feature_set == "raw" else panel.xs_features


def pd_conditions(experiment: Experiment):
    if experiment.feature_set == "raw_market":
        return [PD_MARKET_CONDITION, PD_CONDITION]
    return [PD_CONDITION]


def make_models(experiment: Experiment, panel: Panel):
    features = model_features(experiment, panel)
    if experiment.main_model == "lightgbm":
        main = LGBMModel(features)
    else:
        main = RidgeModel(features)
    models = [*baseline_models(), main]
    if experiment.reference:
        # Referans aynı hedefe eğitildiyse getiri tahmini olarak da karşılaştırılır.
        kind = "forecast" if experiment.targets == ("raw",) else "score"
        models.append(RidgeModel(panel.features, name=REFERENCE_MODEL, kind=kind, target_key="raw"))
    if experiment.lgbm_reference:
        models.append(LGBMModel(panel.features, name=LGBM_REFERENCE))
    return models


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _daily_diagnostics(frame):
    """Günlük rank IC'ler, skorun risk maruziyeti, ilk k hissenin getirisi/riski ve rejim."""
    beta_rows = frame.dropna(subset=["beta_252"])
    return pd.concat(
        [
            daily_rank_ic(frame),
            daily_rank_ic(frame, target_col="y_raw").rename("rank_ic_raw"),
            daily_rank_ic(beta_rows, target_col="beta_adj_return").rename("rank_ic_beta"),
            daily_rank_ic(beta_rows, target_col="beta_252").rename("corr_beta"),
            daily_rank_ic(frame, target_col="volatility_20").rename("corr_vol"),
            top_minus_all(frame, TOP_K),
            top_k_mean(frame, "beta_252", TOP_K).rename("top_beta"),
            top_k_mean(frame, "volatility_20", TOP_K).rename("top_vol"),
            frame.groupby("date")["regime"].first(),
        ],
        axis=1,
    ).rename_axis("date")


def _score(val, pred, target, raw, lags, meta):
    """Bir skorun validation ölçüleri; eksik skorlu satırlar değerlendirmeden düşer."""
    ok = np.isfinite(pred)
    frame = val.loc[ok, EVAL_COLUMNS].assign(
        pred=pred[ok],
        y=val.loc[ok, target].to_numpy(),
        y_raw=val.loc[ok, raw].to_numpy(),
    )
    row = meta | {
        "n_val": len(val),
        "coverage": float(ok.mean()),
        "val_days": int(val["date"].nunique()),
    }
    if meta["kind"] == "forecast":
        row |= error_metrics(frame["y"], frame["pred"])
    days = None
    if np.ptp(frame["pred"].to_numpy()) > 0:
        d = _daily_diagnostics(frame)
        row |= summarize_ic(d["rank_ic"], lags)
        row["ic_raw_mean"] = float(d["rank_ic_raw"].mean())
        row["ic_beta_mean"] = float(d["rank_ic_beta"].mean())
        row["top5_minus_all"] = float(d["top_minus_all"].mean())
        days = d.reset_index().assign(target=meta["target"], fold=meta["fold"], model=meta["model"])
    scored = frame[["date", "symbol", "pred", "y"]].assign(
        target=meta["target"], fold=meta["fold"], model=meta["model"]
    )
    return row, days, scored


def beta_neutral(val, pred):
    """Eğitim değişmez; yalnız değerlendirmede skor her gün betaya karşı nötrlenir."""
    has_beta = val["beta_252"].notna().to_numpy()
    out = np.full(len(val), np.nan)
    frame = val.loc[has_beta, ["date", "beta_252"]].assign(pred=pred[has_beta])
    out[has_beta] = neutralize(frame, "beta_252").to_numpy()
    return out


def partial_dependence(model, frame, feature=PD_FEATURE, condition=PD_CONDITION):
    """``feature`` sabit değerlere çekildiğinde ortalama tahmin; ``condition`` > 0 ve < 0 ayrı.

    Izgara bütün satırlardaki ``feature`` kantilleridir; iki alt kümede aynı değerler kullanılır.
    Boş alt küme (ör. dönem boyunca QQQ hep SMA200 üstünde) atlanır.
    """
    grid = frame[feature].quantile(list(PD_QUANTILES))
    rows = []
    for subset, mask in (("pozitif", frame[condition] > 0), ("negatif", frame[condition] < 0)):
        part = frame.loc[mask]
        if part.empty:
            continue
        for quantile, value in grid.items():
            pred = model.predict(part.assign(**{feature: value}))
            rows.append(
                {
                    "subset": subset,
                    "quantile": float(quantile),
                    "value": float(value),
                    "pd": float(pred.mean()),
                    "rows": int(mask.sum()),
                }
            )
    columns = ["subset", "quantile", "value", "pd", "rows"]
    return pd.DataFrame(rows, columns=columns).assign(condition=condition)


def _frame(items):
    return pd.concat(items, ignore_index=True) if items else None


def evaluate(panel: Panel, experiment: Experiment):
    """Deneyin bütün hedef × dönem × model kombinasyonlarını eğitir ve değerlendirir."""
    lags = panel.horizon - 1
    main_name = experiment.main_model
    fold_rows, daily, preds, exposures = [], [], [], []
    coefs, alphas, trees, importance, dependence, curves = [], [], [], [], [], []
    raw = panel.targets["raw"]
    for target_key in experiment.targets:
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            for model in make_models(experiment, panel):
                fit_target = panel.targets[getattr(model, "target_key", None) or target_key]
                model.fit(train, fit_target)
                pred = model.predict(val)
                scores = [(model.name, model.kind, pred)]
                if model.name == main_name and experiment.beta_neutral_row:
                    scores.append((NEUTRAL_MODEL, "score", beta_neutral(val, pred)))
                for model_name, kind, values in scores:
                    meta = {
                        "target": target_key,
                        "fold": name,
                        "model": model_name,
                        "kind": kind,
                        "n_train": len(train),
                    }
                    row, days, scored = _score(val, values, target, raw, lags, meta)
                    fold_rows.append(row)
                    preds.append(scored)
                    if days is not None:
                        daily.append(days)
                keys = {"target": target_key, "fold": name}
                if isinstance(model, LGBMModel):
                    trees.append(
                        keys
                        | {
                            "model": model.name,
                            "n_trees": model.n_trees_,
                            "at_max_trees": model.n_trees_ == model.max_trees,
                            "inner_mse": model.inner_mse_,
                        }
                    )
                    importance.append(
                        model.importance()
                        .rename("gain_share")
                        .rename_axis("feature")
                        .reset_index()
                        .assign(**keys, model=model.name)
                    )
                    if model.features == panel.features:
                        # 4a ayarlarının early stopping incelemesi; seçimi değiştirmez.
                        counts = sorted({*TREE_CURVE, model.n_trees_})
                        curves.append(
                            model.tree_curve(train, fit_target, counts).assign(
                                **keys, model=model.name, chosen=model.n_trees_
                            )
                        )
                if model.name != main_name:
                    continue
                exposures.append(
                    val[model.features]
                    .corrwith(pd.Series(pred, index=val.index))
                    .rename("corr")
                    .rename_axis("feature")
                    .reset_index()
                    .assign(**keys)
                )
                if isinstance(model, RidgeModel):
                    coefs.append(
                        model.coefficients()
                        .rename("coef")
                        .rename_axis("feature")
                        .reset_index()
                        .assign(**keys)
                    )
                    alphas.append(
                        keys
                        | {
                            "alpha": model.alpha_,
                            "alpha_at_grid_edge": model.alpha_
                            in (min(ALPHA_GRID), max(ALPHA_GRID)),
                            **{f"inner_mse_{a:g}": m for a, m in model.inner_mse_.items()},
                        }
                    )
                if isinstance(model, LGBMModel):
                    for condition in pd_conditions(experiment):
                        dependence.append(
                            partial_dependence(model, val, condition=condition).assign(**keys)
                        )
    return {
        "folds": pd.DataFrame(fold_rows),
        "daily": pd.concat(daily, ignore_index=True),
        "predictions": pd.concat(preds, ignore_index=True),
        "exposures": pd.concat(exposures, ignore_index=True),
        "coefficients": _frame(coefs),
        "alphas": pd.DataFrame(alphas) if alphas else None,
        "trees": pd.DataFrame(trees) if trees else None,
        "importance": _frame(importance),
        "partial_dependence": _frame(dependence),
        "tree_curve": _frame(curves),
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
        row["val_days"] = int(part["val_days"].sum())
        if not d.empty:
            row |= summarize_ic(d.set_index("date")["rank_ic"], lags)
            # Bilgi: skorun sabit olduğu (sıralama üretmediği) günler 0 sayılırsa ortalama.
            row["ic_zero_filled"] = float(d["rank_ic"].sum() / row["val_days"])
            by_regime = d.groupby("regime")["rank_ic"].mean()
            row |= {f"ic_{r}": float(by_regime.get(r, np.nan)) for r in REGIMES}
            down = d.loc[d["regime"].eq("asagi")].set_index("date")["rank_ic"]
            row["ic_asagi_t_nw"] = newey_west_tstat(down, lags)
            row["ic_raw_mean"] = float(d["rank_ic_raw"].mean())
            row["ic_beta_mean"] = float(d["rank_ic_beta"].mean())
            row["ic_beta_t_nw"] = summarize_ic(d.set_index("date")["rank_ic_beta"], lags)["ic_t_nw"]
            row["corr_beta"] = float(d["corr_beta"].mean())
            row["corr_vol"] = float(d["corr_vol"].mean())
            row["top5_minus_all"] = float(d["top_minus_all"].mean())
            row["top5_beta"] = float(d["top_beta"].mean())
            row["top5_vol"] = float(d["top_vol"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)

    consistency = []
    for target in experiment.targets:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        main = f.loc[experiment.main_model]
        rec = {"target": target, "folds": len(main)}
        for base in ERROR_BASELINES:
            rec[f"mse_beats_{base}"] = int((main["mse"] < f.loc[base, "mse"]).sum())
        rec["ic_positive"] = int((main["ic_mean"] > 0).sum())
        for base in (*RANK_BASELINES, *RISK_BASELINES):
            rec[f"ic_beats_{base}"] = int((main["ic_mean"] > f.loc[base, "ic_mean"]).sum())
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
    if panel.excluded is not None:
        excluded = panel.excluded.pivot_table(index="fold", columns="partition", values="excluded")
        context = context.join(excluded.add_prefix("excluded_"))
    return summary, pd.DataFrame(consistency), context.reset_index()


def run(panel: Panel, output: Path, experiment: Experiment = EXPERIMENTS["1"]):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Çıktı klasörü zaten var: {output}")
    coverage = None
    if experiment.universe == "beta":
        panel = panel.with_beta()
    elif experiment.universe == "market":
        # Kapsam bütün uygun satırlarda sayılır; deney yalnız ortak satırlarda koşar.
        coverage = market_coverage(panel.rows, panel.market_features)
        panel = panel.with_market()
    result = evaluate(panel, experiment)
    summary, consistency, context = pooled(result, panel, experiment)

    output.mkdir(parents=True)
    result["folds"].to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "pooled_metrics.csv", index=False)
    consistency.to_csv(output / "consistency.csv", index=False)
    context.to_csv(output / "period_context.csv", index=False)
    prefix = "lgbm" if experiment.main_model == "lightgbm" else "ridge"
    result["exposures"].to_csv(output / f"{prefix}_exposure.csv", index=False)
    files = {
        "coefficients": "ridge_coefficients.csv",
        "alphas": "ridge_alpha.csv",
        "trees": "lgbm_trees.csv",
        "importance": "lgbm_importance.csv",
        "partial_dependence": "lgbm_partial_dependence.csv",
        "tree_curve": "lgbm_tree_curve.csv",
    }
    for key, file in files.items():
        if result[key] is not None:
            result[key].to_csv(output / file, index=False)
    if coverage is not None:
        coverage[0].to_csv(output / "market_coverage.csv", index=False)
        coverage[1].to_csv(output / "market_coverage_detail.csv", index=False)
    result["feature_ic"].to_csv(output / "feature_ic.csv", index=False)
    result["daily"].to_parquet(output / "daily_ic.parquet", index=False)
    result["predictions"].to_parquet(output / "predictions.parquet", index=False)

    models = make_models(experiment, panel)
    main = next(m for m in models if m.name == experiment.main_model)
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
        "membership": panel.membership_file,
        "membership_dropped_rows": panel.membership_dropped,
        "horizon": panel.horizon,
        "targets": {k: panel.targets[k] for k in experiment.targets},
        "benchmark": "QQQ",
        "main_model": experiment.main_model,
        "main_label": experiment.label,
        "feature_set": experiment.feature_set,
        "market_features": (
            panel.market_features if experiment.feature_set == "raw_market" else None
        ),
        "features": main.features,
        "folds": panel.folds,
        "test_start": str(panel.test_start.date()),
        "test_period_read": False,
        "models": [m.name for m in models],
        "reference_model": REFERENCE_MODEL if experiment.reference else None,
        "lgbm_reference_model": LGBM_REFERENCE if experiment.lgbm_reference else None,
        "universe": experiment.universe,
        "excluded_rows": (
            panel.excluded.to_dict("records") if panel.excluded is not None else None
        ),
        "beta_neutral_row": NEUTRAL_MODEL if experiment.beta_neutral_row else None,
        "market_coverage": (
            {
                "summary": coverage[0].to_dict("records"),
                "detail": coverage[1]
                .astype({"first_missing": str, "last_missing": str})
                .to_dict("records"),
            }
            if coverage is not None
            else None
        ),
        "criteria": experiment.criteria,
        "ridge": {
            "alpha_grid": list(ALPHA_GRID),
            "alpha_selection": "eğitimin son 252 seansı iç validation, purge'lü, MSE",
            "feature_winsor_quantile": 0.01,
            "target_winsor_quantile": 0.01,
        },
        "rank_ic": {"method": "günlük Spearman", "min_names": 5, "nw_lags": panel.horizon - 1},
        "diagnostics": {
            "beta": (
                f"QQQ'ya karşı günlük getirilerle {BETA_WINDOW} seans (en az {BETA_MIN_OBS} "
                "gözlem), yalnız t'ye kadar"
            ),
            "regime_threshold": REGIME_THRESHOLD,
        },
        "top_k": TOP_K,
    }
    if experiment.main_model == "lightgbm":
        config["lightgbm"] = {
            "params": LGBM_PARAMS,
            "max_trees": LGBM_MAX_TREES,
            "early_stopping_rounds": LGBM_EARLY_STOPPING,
            "tree_selection": "eğitimin son 252 seansı iç validation, purge'lü, l2",
            "seed": LGBM_SEED,
            "deterministic": True,
            "force_col_wise": True,
            "target_winsor_quantile": 0.01,
            "partial_dependence": {
                "feature": PD_FEATURE,
                "conditions": [f"{c} > 0 / < 0" for c in pd_conditions(experiment)],
                "quantiles": list(PD_QUANTILES),
                "rows": "validation",
            },
            "tree_curve": {
                "counts": list(TREE_CURVE),
                "model": "4a ayarları (24 özellik)",
                "rows": "iç bölme: eğitimin son 252 seansı, purge'lü",
            },
        }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from .report import write_report

    write_report(output, config, result, summary, consistency, context)
    _register(output, config, summary)
    return config, summary, consistency


def _migrate_registry(registry):
    """İlk sürüm kayıtlarına ana model kolonlarını ekler; hiçbir satır silinmez."""
    with registry.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream), None)
    if header == REGISTRY_FIELDS:
        return
    if header != LEGACY_REGISTRY_FIELDS:
        raise ValueError(f"{registry} kolonları beklenen biçimde değil")
    with registry.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row |= {
            "main_model": MAIN_MODEL,
            "main_ic": row["ridge_ic"],
            "main_ic_t_nw": row["ridge_ic_t_nw"],
            "main_r2": row["ridge_r2"],
        }
    with registry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _register(output, config, summary):
    """Denenen her kombinasyon output/experiments/runs.csv dosyasına eklenir."""
    registry = output.parent / "runs.csv"
    if registry.exists():
        _migrate_registry(registry)
    main_name = config["main_model"]
    main = summary.loc[summary["model"].eq(main_name)].set_index("target")
    new = not registry.exists()
    with registry.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        if new:
            writer.writeheader()
        for target in config["targets"]:
            values = {
                "ic": round(float(main.loc[target, "ic_mean"]), 5),
                "ic_t_nw": round(float(main.loc[target, "ic_t_nw"]), 3),
                "r2": round(float(main.loc[target, "r2_oos"]), 5),
            }
            ridge = values if main_name == MAIN_MODEL else dict.fromkeys(values, "")
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
                    **{f"ridge_{k}": v for k, v in ridge.items()},
                    "main_model": main_name,
                    **{f"main_{k}": v for k, v in values.items()},
                }
            )
