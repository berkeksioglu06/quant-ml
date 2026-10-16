"""LightGBM tanısı: "model hiçbir şey öğrenmiyor" sorusu için debug adımları.

Deney 4a/4b çerçevesinde (aynı satırlar, K1–K4, test dönemi kapalı) beş soru:

1. Öğrenme eğrisi: ağaç sayısı arttıkça eğitim ve validation ölçüleri ne yapıyor?
   Eğitim IC'si yükselip validation yükselmiyorsa sorun "az epoch" değildir.
2. Etiket karıştırma: hedef karıştırılınca eğitim uyumu gerçek hedeftekine yakınsa
   model gürültüye uyuyor demektir (sinyal yok ya da çok zayıf).
3. Hiperparametre taraması (yalnız bilgi; seçim yapılmaz): hiçbir ayar K1–K4'te
   kıyasları geçmiyorsa sorun "ayar" değildir.
4. Early stopping ölçüsü: iç bölmede l2 ile seçilen ağaç sayısı ve IC ile seçilen
   ağaç sayısı validation'da ne veriyor?
5. Özellik kümesi: yalnız piyasa (gün bazlı) özellikleri, yalnız hisse özellikleri,
   gün içi sıralanmış özellikler + gün içi hedef. R²'nin kaynağı zamanlama mı,
   sıralama mı?

Çalıştırma: ``uv run python -m research.diag_lgbm``. Çıktı
``output/experiments/<zaman>_diag_lgbm_h20/`` altına CSV'ler ve ``TANI.md``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import load_panel
from .metrics import daily_rank_ic, error_metrics
from .models import LGBM_PARAMS, LGBM_SEED, clipped_target, purged_inner_split

DEFAULT_DATASET = Path("output/yahoo_20261016T185837Z_dataset_v1_1")
CURVE = (1, 5, 10, 25, 50, 100, 200, 400, 1000)
GRID = {
    "min_child_samples": (20, 200, 1000),
    "num_leaves": (7, 31),
    "learning_rate": (0.03, 0.1),
}
GRID_TREES = 300
SHUFFLE_TREES = 200
SHUFFLE_REPEATS = 3


def _regressor(params, n_trees, n_jobs):
    return lgb.LGBMRegressor(
        **params,
        n_estimators=n_trees,
        random_state=LGBM_SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _matrix(frame, features):
    return pd.DataFrame(frame[features].to_numpy(dtype=float), columns=features)


def _ic(frame, pred, target):
    ic = daily_rank_ic(frame[["date"]].assign(pred=pred, y=frame[target]))
    n_days = frame["date"].nunique()
    return {
        "ic": float(ic.mean()) if len(ic) else np.nan,
        "ic_days": len(ic),
        "const_day_share": 1 - len(ic) / n_days,
        "ic_zero_filled": float(ic.sum() / n_days) if n_days else np.nan,
    }


def _score(frame, pred, target, prefix):
    out = {f"{prefix}_{k}": v for k, v in _ic(frame, pred, target).items()}
    err = error_metrics(frame[target], pred)
    out[f"{prefix}_r2"] = err["r2_oos"]
    return out


def feature_sets(panel):
    stock_only = [f for f in panel.features if not f.startswith("market_")]
    return {
        # Deney 4a: 24 ham özellik, hissenin kendi getirisi.
        "raw24": (list(panel.features), "raw"),
        # Deney 4b: 24 + 6 piyasa durumu özelliği.
        "raw30_market": ([*panel.features, *panel.market_features], "raw"),
        # Yalnız gün bazlı özellikler: sıralama üretemez, R² varsa zamanlamadandır.
        "market_only8": (
            [f for f in panel.features if f.startswith("market_")] + list(panel.market_features),
            "raw",
        ),
        # Yalnız hissenin kendi özellikleri (gün bazlı olanlar çıkarıldı).
        "stock_only22": (stock_only, "raw"),
        # Deney 2 kurulumu: gün içi sıralanmış 22 özellik + gün ortalamasından arındırılmış hedef.
        "xs22_xs_target": (list(panel.xs_features), "xs"),
    }


def learning_curve(panel, sets, n_jobs, log):
    """Early stopping yok; sabit ağaç sayılarında eğitim ve validation ölçüleri."""
    rows = []
    for set_name, (features, target_key) in sets.items():
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            t0 = time.time()
            model = _regressor(LGBM_PARAMS, max(CURVE), n_jobs).fit(
                _matrix(train, features), clipped_target(train, target, 0.01)
            )
            for n in CURVE:
                p_tr = model.predict(_matrix(train, features), num_iteration=n)
                p_va = model.predict(_matrix(val, features), num_iteration=n)
                rows.append(
                    {"set": set_name, "target": target_key, "fold": name, "n_trees": n}
                    | _score(train, p_tr, target, "train")
                    | _score(val, p_va, target, "val")
                )
            log(f"  öğrenme eğrisi {set_name} {name}: {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


def shuffle_test(panel, sets, n_jobs, log):
    """Hedef karıştırılınca eğitim uyumu ne kadar düşüyor? (gerçek − karıştırılmış)."""
    rows = []
    rng = np.random.default_rng(LGBM_SEED)
    for set_name in ("raw24", "xs22_xs_target"):
        if set_name not in sets:
            continue
        features, target_key = sets[set_name]
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            X_tr, X_va = _matrix(train, features), _matrix(val, features)
            y_real = clipped_target(train, target, 0.01)
            model = _regressor(LGBM_PARAMS, SHUFFLE_TREES, n_jobs).fit(X_tr, y_real)
            rows.append(
                {"set": set_name, "fold": name, "label": "gercek", "repeat": 0}
                | _score(train, model.predict(X_tr), target, "train")
                | _score(val, model.predict(X_va), target, "val")
            )
            for r in range(SHUFFLE_REPEATS):
                # Tam karıştırma: hedef ile satır arasındaki her bağ kopar.
                shuffled = train.assign(**{target: rng.permutation(train[target].to_numpy())})
                model = _regressor(LGBM_PARAMS, SHUFFLE_TREES, n_jobs).fit(
                    X_tr, clipped_target(shuffled, target, 0.01)
                )
                rows.append(
                    {"set": set_name, "fold": name, "label": "karistirilmis", "repeat": r + 1}
                    | _score(shuffled, model.predict(X_tr), target, "train")
                    | _score(val, model.predict(X_va), target, "val")
                )
            log(f"  karıştırma {set_name} {name} bitti")
    return pd.DataFrame(rows)


def hp_grid(panel, sets, n_jobs, log):
    """Küçük ızgara, sabit ağaç sayısı; yalnız 'ayar sorunu mu' sorusu için."""
    rows = []
    combos = list(itertools.product(*GRID.values()))
    for set_name in ("raw24", "xs22_xs_target"):
        if set_name not in sets:
            continue
        features, target_key = sets[set_name]
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            X_tr, X_va = _matrix(train, features), _matrix(val, features)
            y_tr = clipped_target(train, target, 0.01)
            t0 = time.time()
            for values in combos:
                params = LGBM_PARAMS | dict(zip(GRID, values))
                if params["num_leaves"] > 2 ** params["max_depth"]:
                    params["max_depth"] = -1  # 31 yaprak için derinlik sınırı kalkar
                model = _regressor(params, GRID_TREES, n_jobs).fit(X_tr, y_tr)
                rows.append(
                    {"set": set_name, "fold": name}
                    | dict(zip(GRID, values))
                    | _score(train, model.predict(X_tr), target, "train")
                    | _score(val, model.predict(X_va), target, "val")
                )
            log(f"  ızgara {set_name} {name}: {len(combos)} ayar, {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def early_stopping_metric(panel, sets, n_jobs, log):
    """İç bölmede l2 ile ve IC ile seçilen ağaç sayısı; ikisinin validation sonucu."""
    rows = []
    counts = sorted({*CURVE, 2, 3, 150, 300, 600, 800})
    for set_name in ("raw24", "xs22_xs_target"):
        if set_name not in sets:
            continue
        features, target_key = sets[set_name]
        target = panel.targets[target_key]
        for fold in panel.folds:
            name = fold["name"]
            train, val = panel.split(name)
            inner_train, inner_val = purged_inner_split(train)
            inner = _regressor(LGBM_PARAMS, max(counts), n_jobs).fit(
                _matrix(inner_train, features), clipped_target(inner_train, target, 0.01)
            )
            X_in = _matrix(inner_val, features)
            curve = []
            for n in counts:
                p = inner.predict(X_in, num_iteration=n)
                curve.append(
                    {"n_trees": n, "inner_l2": error_metrics(inner_val[target], p)["mse"]}
                    | {f"inner_{k}": v for k, v in _ic(inner_val, p, target).items()}
                )
            curve = pd.DataFrame(curve)
            by_l2 = int(curve.loc[curve["inner_l2"].idxmin(), "n_trees"])
            by_ic = int(curve.loc[curve["inner_ic_zero_filled"].idxmax(), "n_trees"])
            full = _regressor(LGBM_PARAMS, max(by_l2, by_ic), n_jobs).fit(
                _matrix(train, features), clipped_target(train, target, 0.01)
            )
            X_va = _matrix(val, features)
            for rule, n in (("l2", by_l2), ("rank_ic", by_ic)):
                rows.append(
                    {"set": set_name, "fold": name, "rule": rule, "n_trees": n}
                    | _score(val, full.predict(X_va, num_iteration=n), target, "val")
                )
            log(f"  early stopping {set_name} {name}: l2→{by_l2} ağaç, IC→{by_ic} ağaç")
    return pd.DataFrame(rows)


def single_feature_ic(panel):
    """Her özelliğin tek başına günlük rank IC'si, eğitim ve validation ayrı (tavan)."""
    target = panel.targets["raw"]
    rows = []
    for fold in panel.folds:
        for partition, frame in zip(("train", "validation"), panel.split(fold["name"])):
            for feature in panel.features:
                if feature.startswith("market_"):
                    continue
                ic = daily_rank_ic(frame[["date"]].assign(pred=frame[feature], y=frame[target]))
                rows.append(
                    {
                        "fold": fold["name"],
                        "partition": partition,
                        "feature": feature,
                        "ic": float(ic.mean()),
                    }
                )
    return pd.DataFrame(rows)


def target_structure(panel):
    """Hedef varyansının gün-arası payı ve kesit büyüklüğü."""
    target = panel.targets["raw"]
    rows = []
    for fold in panel.folds:
        for partition, frame in zip(("train", "validation"), panel.split(fold["name"])):
            y = frame[target]
            day_mean = y.groupby(frame["date"]).transform("mean")
            rows.append(
                {
                    "fold": fold["name"],
                    "partition": partition,
                    "rows": len(frame),
                    "days": frame["date"].nunique(),
                    "names_per_day": len(frame) / frame["date"].nunique(),
                    "target_var": float(y.var()),
                    "between_day_share": float(day_mean.var() / y.var()),
                    "target_mean": float(y.mean()),
                }
            )
    return pd.DataFrame(rows)


def _fmt(x, pct=False):
    if pd.isna(x):
        return "—"
    return f"{x:+.1%}" if pct else f"{x:+.3f}"


def write_report(output, panel, structure, fic, curve, shuffle, grid, es, elapsed):
    L = []
    L.append("# LightGBM tanısı — model neden öğrenmiyor? (H = 20, K1–K4, test kapalı)\n")
    L.append(
        f"Süre {elapsed / 60:.1f} dk. Ayarlar Deney 4a ile aynı; early stopping yok, ağaç sayıları sabit.\n"
    )

    L.append("## 0. Hedefin yapısı\n")
    L.append("| Dönem | Bölme | Satır | Gün | Hisse/gün | Hedef var | Gün-arası pay | Ort |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in structure.itertuples():
        L.append(
            f"| {r.fold} | {r.partition} | {r.rows} | {r.days} | {r.names_per_day:.1f} | "
            f"{r.target_var:.4f} | {r.between_day_share:.0%} | {r.target_mean:+.1%} |"
        )
    L.append("")
    L.append(
        "Gün-arası pay: hedef varyansının gün ortalamalarına (piyasa yönüne) düşen kısmı. "
        "MSE hedefi bu kısmı da küçültmeye çalışır; rank IC ise yalnız gün içi sıralamaya bakar.\n"
    )

    L.append("## 1. Tek özellik tavanı: eğitimde ve validation'da işaret\n")
    piv = fic.pivot_table(index="feature", columns=["partition", "fold"], values="ic")
    tr = piv["train"].mean(axis=1)
    va = piv["validation"].mean(axis=1)
    tab = pd.DataFrame({"train_ic": tr, "val_ic": va}).sort_values("val_ic")
    L.append("| Özellik | Eğitim IC (K1–K4 ort) | Validation IC (K1–K4 ort) | İşaret |")
    L.append("|---|---:|---:|---|")
    for f, r in tab.iterrows():
        flip = "TERS" if np.sign(r.train_ic) != np.sign(r.val_ic) and abs(r.val_ic) > 0.03 else ""
        L.append(f"| `{f}` | {_fmt(r.train_ic)} | {_fmt(r.val_ic)} | {flip} |")
    L.append("")

    L.append(
        f"Satır: {len(panel.rows)}; üyelik maskesi: {panel.membership_file or 'yok'}"
        f" (düşen {panel.membership_dropped}).\n"
    )
    if curve is None:
        L.append("## 2. Öğrenme eğrisi — koşulmadı\n")
        curve = pd.DataFrame(columns=["set"])
    else:
        L.append("## 2. Öğrenme eğrisi (early stopping yok)\n")
    L.append(
        "Her özellik kümesi için ağaç sayısına göre eğitim ve validation rank IC'si (K1–K4 "
        "ortalaması; sabit skorlu günler 0 sayılır) ve R². Eğitim IC'si artıp validation "
        "artmıyorsa 'az epoch' değildir.\n"
    )
    for set_name, g in curve.groupby("set", sort=False):
        agg = g.groupby("n_trees").agg(
            train_ic=("train_ic_zero_filled", "mean"),
            val_ic=("val_ic_zero_filled", "mean"),
            train_r2=("train_r2", "mean"),
            val_r2=("val_r2", "mean"),
            const=("val_const_day_share", "mean"),
        )
        L.append(f"**{set_name}** (hedef: {g['target'].iloc[0]})\n")
        L.append("| Ağaç | Eğitim IC | Val IC | Eğitim R² | Val R² | Val sabit gün |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        for n, r in agg.iterrows():
            L.append(
                f"| {n} | {_fmt(r.train_ic)} | {_fmt(r.val_ic)} | {_fmt(r.train_r2, True)} | "
                f"{_fmt(r.val_r2, True)} | {r.const:.0%} |"
            )
        per_fold = g[g.n_trees.eq(200)].set_index("fold")["val_ic_zero_filled"]
        L.append(
            "\n200 ağaçta dönem bazında val IC: "
            + ", ".join(f"{k} {_fmt(v)}" for k, v in per_fold.items())
            + "\n"
        )

    if shuffle is None:
        L.append("## 3. Etiket karıştırma — koşulmadı\n")
        shuffle = pd.DataFrame(columns=["set", "fold", "label"])
    L.append("## 3. Etiket karıştırma (200 ağaç)\n")
    L.append(
        "Hedef eğitim içinde rastgele karıştırıldı. Eğitim uyumu gerçek hedeftekine yakınsa "
        "modelin eğitimde 'öğrendiği' şey büyük ölçüde gürültüdür.\n"
    )
    L.append("| Küme | Dönem | Etiket | Eğitim IC | Eğitim R² | Val IC |")
    L.append("|---|---|---|---:|---:|---:|")
    agg = shuffle.groupby(["set", "fold", "label"], sort=False).mean(numeric_only=True)
    for (s, f, lab), r in agg.iterrows():
        L.append(
            f"| {s} | {f} | {lab} | {_fmt(r.train_ic_zero_filled)} | {_fmt(r.train_r2, True)} | "
            f"{_fmt(r.val_ic_zero_filled)} |"
        )
    L.append("")

    if grid is None:
        L.append("## 4. Hiperparametre taraması — koşulmadı\n")
        grid = pd.DataFrame(columns=["set"])
    L.append("## 4. Hiperparametre taraması (300 ağaç; yalnız bilgi, seçim yapılmaz)\n")
    # Momentum kıyası (return_20) bölüm 1'deki tek özellik tablosundan: K1–K4 validation
    # IC ortalaması. Evrene göre değişir; sabit bir sayı yazılmaz.
    momentum = float(
        fic.loc[fic["partition"].eq("validation") & fic["feature"].eq("return_20"), "ic"].mean()
    )
    L.append(
        "Her ayar için K1–K4 validation rank IC ortalaması ve pozitif dönem sayısı. "
        f"Kıyas: momentum_20 (return_20) validation IC ortalaması {_fmt(momentum)}.\n"
    )
    for set_name, g in grid.groupby("set", sort=False):
        agg = (
            g.groupby(list(GRID))
            .agg(
                val_ic=("val_ic_zero_filled", "mean"),
                pos=("val_ic_zero_filled", lambda s: int((s > 0).sum())),
                train_ic=("train_ic_zero_filled", "mean"),
                val_r2=("val_r2", "mean"),
            )
            .reset_index()
            .sort_values("val_ic", ascending=False)
        )
        L.append(f"**{set_name}**\n")
        L.append("| min_child | leaves | lr | Eğitim IC | Val IC | Pozitif dönem | Val R² |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in agg.itertuples():
            L.append(
                f"| {r.min_child_samples} | {r.num_leaves} | {r.learning_rate} | "
                f"{_fmt(r.train_ic)} | {_fmt(r.val_ic)} | {r.pos}/4 | {_fmt(r.val_r2, True)} |"
            )
        L.append(
            f"\nEn iyi/en kötü val IC: {_fmt(agg.val_ic.max())} / {_fmt(agg.val_ic.min())}; "
            f"{int((agg.val_ic > momentum).sum())}/{len(agg)} ayar momentumu geçiyor.\n"
        )

    if es is None:
        L.append("## 5. Early stopping ölçüsü — koşulmadı\n")
        es = pd.DataFrame(columns=["set"])
    L.append("## 5. Early stopping ölçüsü: l2 mi, IC mi?\n")
    L.append("| Küme | Dönem | Kural | Ağaç | Val IC | Val R² | Val sabit gün |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for r in es.itertuples():
        L.append(
            f"| {r.set} | {r.fold} | {r.rule} | {r.n_trees} | {_fmt(r.val_ic_zero_filled)} | "
            f"{_fmt(r.val_r2, True)} | {r.val_const_day_share:.0%} |"
        )
    L.append("")
    L.append(
        "Dosyalar: `target_structure.csv`, `single_feature_ic.csv`, `learning_curve.csv`, "
        "`shuffle_test.csv`, `hp_grid.csv`, `early_stopping_metric.csv`.\n"
    )
    (output / "TANI.md").write_text("\n".join(L), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="LightGBM tanısı (K1–K4; test dönemi kapalı)")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1, help="LightGBM iş parçacığı")
    parser.add_argument("--membership", type=Path, help="Günlük evren üyeliği parquet'i")
    parser.add_argument(
        "--sets",
        help="virgülle özellik kümeleri (varsayılan hepsi): raw24,raw30_market,market_only8,"
        "stock_only22,xs22_xs_target",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--steps",
        default="all",
        help="virgülle: structure,feature,curve,shuffle,grid,es (varsayılan hepsi)",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    dataset = args.dataset if args.dataset.is_absolute() else project / args.dataset
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or project / "output/experiments" / f"{stamp}_diag_lgbm_h{args.horizon}"
    output.mkdir(parents=True, exist_ok=False)
    steps = set(args.steps.split(",")) if args.steps != "all" else None

    def log(msg):
        print(msg, flush=True)

    def want(step):
        return steps is None or step in steps

    t0 = time.time()
    panel = load_panel(dataset, project, args.horizon, membership=args.membership).with_market()
    sets = feature_sets(panel)
    if args.sets:
        chosen = args.sets.split(",")
        if unknown := set(chosen) - set(sets):
            parser.error(f"bilinmeyen küme: {sorted(unknown)}")
        sets = {k: v for k, v in sets.items() if k in chosen}
    log(f"panel: {len(panel.rows)} satır, {panel.rows['symbol'].nunique()} hisse")

    structure = target_structure(panel)
    structure.to_csv(output / "target_structure.csv", index=False)
    fic = single_feature_ic(panel)
    fic.to_csv(output / "single_feature_ic.csv", index=False)
    log("yapı ve tek özellik IC'si yazıldı")

    curve = shuffle = grid = es = None
    if want("curve"):
        log("öğrenme eğrisi...")
        curve = learning_curve(panel, sets, args.jobs, log)
        curve.to_csv(output / "learning_curve.csv", index=False)
    if want("shuffle"):
        log("etiket karıştırma...")
        shuffle = shuffle_test(panel, sets, args.jobs, log)
        shuffle.to_csv(output / "shuffle_test.csv", index=False)
    if want("grid"):
        log("hiperparametre ızgarası...")
        grid = hp_grid(panel, sets, args.jobs, log)
        grid.to_csv(output / "hp_grid.csv", index=False)
    if want("es"):
        log("early stopping ölçüsü...")
        es = early_stopping_metric(panel, sets, args.jobs, log)
        es.to_csv(output / "early_stopping_metric.csv", index=False)

    write_report(output, panel, structure, fic, curve, shuffle, grid, es, time.time() - t0)
    (output / "config.json").write_text(
        json.dumps(
            {
                "params": LGBM_PARAMS,
                "curve": CURVE,
                "grid": GRID,
                "grid_trees": GRID_TREES,
                "shuffle_trees": SHUFFLE_TREES,
                "snapshot": panel.snapshot,
                "membership": panel.membership_file,
                "membership_dropped_rows": panel.membership_dropped,
                "test_period_read": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"bitti {time.time() - t0:.0f}s → {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
