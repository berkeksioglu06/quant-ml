"""Aşama 1 raporu: 'Ridge basit kıyasları geçiyor mu, hangi hedefte, hangi dönemde?'"""

from __future__ import annotations

import math

from .data import REGIMES
from .experiment import ERROR_BASELINES, MAIN_MODEL, RANK_BASELINES, RISK_BASELINES

TARGET_NAMES = {
    "raw": "Hissenin kendi getirisi",
    "vs_qqq": "QQQ'ya göre getiri",
    "xs": "Gün içi ortalamadan arındırılmış getiri",
}
MODEL_NAMES = {
    "sifir": "Sıfır",
    "egitim_ortalamasi": "Eğitim ortalaması",
    "momentum_20": "Momentum (return_20)",
    "geri_donus_5": "Geri dönüş (−return_5)",
    "en_oynak_20": "En oynak (volatility_20)",
    "yuksek_beta": "Yüksek beta (beta_252)",
    "ridge": "**Ridge**",
    "ridge_deney1": "Ridge · Deney 1 (referans)",
}
REGIME_NAMES = {"yukari": "QQQ ↑", "yatay": "Yatay", "asagi": "QQQ ↓"}


def _missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x, digits=3):
    return "—" if _missing(x) else f"{x:+.{digits}f}"


def pct(x, digits=2):
    return "—" if _missing(x) else f"{100 * x:+.{digits}f}%"


def plain(x, digits=2):
    return "—" if _missing(x) else f"{x:.{digits}f}"


def para(*parts):
    return "".join(parts)


def table(header, rows, left=1):
    align = ["---"] * left + ["---:"] * (len(header) - left)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _rank_models(summary):
    present = set(summary["model"])
    order = (*RANK_BASELINES, *RISK_BASELINES, MAIN_MODEL, "ridge_deney1")
    return [m for m in order if m in present]


def _universe(context):
    days = context[[f"days_{r}" for r in REGIMES]].sum(axis=1)
    return {
        "beta": float((context["universe_beta_252"] * days).sum() / days.sum()),
        "vol": float((context["universe_volatility_20"] * days).sum() / days.sum()),
        "days": {r: int(context[f"days_{r}"].sum()) for r in REGIMES},
    }


def verdict(target, folds, summary, consistency):
    """Kural tabanlı kısa hüküm; eşikler raporda açıkça yazılıdır."""
    s = summary.loc[summary["target"].eq(target)].set_index("model")
    c = consistency.loc[consistency["target"].eq(target)].iloc[0]
    f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
    n = int(c["folds"])

    best_error = max(ERROR_BASELINES, key=lambda m: s.loc[m, "r2_oos"])
    error_wins = min(c[f"mse_beats_{m}"] for m in ERROR_BASELINES)
    ridge_r2 = s.loc[MAIN_MODEL, "r2_oos"]
    if ridge_r2 > s.loc[best_error, "r2_oos"] and error_wins >= n - 1:
        error = "geçiyor"
    elif ridge_r2 < s.loc[best_error, "r2_oos"] and error_wins <= 1:
        error = "geçmiyor"
    else:
        error = "karışık"

    best_rank = max(RANK_BASELINES, key=lambda m: s.loc[m, "ic_mean"])
    rank_wins = min(c[f"ic_beats_{m}"] for m in RANK_BASELINES)
    ridge_ic = s.loc[MAIN_MODEL, "ic_mean"]
    if (
        ridge_ic > s.loc[best_rank, "ic_mean"]
        and ridge_ic > 0
        and rank_wins >= n - 1
        and c["ic_positive"] >= n - 1
    ):
        rank = "geçiyor"
    elif ridge_ic < s.loc[best_rank, "ic_mean"] and rank_wins <= 1:
        rank = "geçmiyor"
    else:
        rank = "karışık"

    names = sorted({fold for _, fold in f.index})
    error_folds = [
        k
        for k in names
        if f.loc[(MAIN_MODEL, k), "mse"] < min(f.loc[(m, k), "mse"] for m in ERROR_BASELINES)
    ]
    rank_folds = [
        k
        for k in names
        if f.loc[(MAIN_MODEL, k), "ic_mean"]
        > max(0, *(f.loc[(m, k), "ic_mean"] for m in RANK_BASELINES))
    ]
    return {
        "error": error,
        "rank": rank,
        "error_folds": error_folds,
        "rank_folds": rank_folds,
        "ridge_r2": ridge_r2,
        "best_error": best_error,
        "best_error_r2": s.loc[best_error, "r2_oos"],
        "ridge_ic": ridge_ic,
        "ridge_t": s.loc[MAIN_MODEL, "ic_t_nw"],
        "best_rank": best_rank,
        "best_rank_ic": s.loc[best_rank, "ic_mean"],
    }


def _short_answer(config, folds, summary, consistency, context, exposures):
    names = [f["name"] for f in config["folds"]]
    universe = _universe(context)
    lines = []
    for target in config["targets"]:
        v = verdict(target, folds, summary, consistency)
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        lines += [
            f"**{TARGET_NAMES[target]}**",
            para(
                f"- Tahmin hatası: Ridge **{v['error']}**. K1–K4 R²: Ridge {pct(v['ridge_r2'])}, ",
                f"en iyi sabit kıyas ({MODEL_NAMES[v['best_error']]}) {pct(v['best_error_r2'])}. ",
                "İki sabit kıyası birden geçtiği dönemler: ",
                f"{', '.join(v['error_folds']) or 'yok'}.",
            ),
            para(
                f"- Günlük sıralama: Ridge **{v['rank']}**. K1–K4 rank IC: Ridge ",
                f"{num(v['ridge_ic'])} (Newey-West t = {num(v['ridge_t'], 2)}), en iyi skor ",
                f"kıyası ({MODEL_NAMES[v['best_rank']]}) {num(v['best_rank_ic'])}. ",
                "Momentum ve geri dönüşü birlikte geçip pozitif kaldığı dönemler: ",
                f"{', '.join(v['rank_folds']) or 'yok'}.",
            ),
            para(
                "- Rejim ve risk: rank IC QQQ ↑ / yatay / QQQ ↓ = ",
                " / ".join(num(s.loc[MAIN_MODEL, f"ic_{r}"]) for r in REGIMES),
                f"; beta-düzeltmeli IC {num(s.loc[MAIN_MODEL, 'ic_beta_mean'])}. ",
                f"Seçilen ilk {config['top_k']} hissenin ortalama betası ",
                f"{plain(s.loc[MAIN_MODEL, 'top5_beta'])}, volatilitesi ",
                f"{plain(s.loc[MAIN_MODEL, 'top5_vol'])} ",
                f"(evren: {plain(universe['beta'])} / {plain(universe['vol'])}).",
            ),
        ]
        if "ridge_deney1" in s.index:
            ref = s.loc["ridge_deney1"]
            better = "üstünde" if v["ridge_ic"] > ref["ic_mean"] else "altında"
            lines.append(
                para(
                    f"- Deney 1 Ridge'i (referans) K1–K4 rank IC {num(ref['ic_mean'])} ",
                    f"(t = {num(ref['ic_t_nw'], 2)}); bu deneyin Ridge'i referansın {better}.",
                )
            )
        lines.append("")
    lead = exposures.assign(a=exposures["corr"].abs()).groupby("feature")["a"].mean()
    lead = lead.sort_values(ascending=False)
    lines += [
        para(
            "Ridge skorunun validation yıllarındaki ortalama mutlak korelasyonu en yüksek "
            "girdiler: ",
            ", ".join(f"`{x}` ({lead[x]:.2f})" for x in lead.index[:3]),
            ". Ayrıntı: bölüm 6.",
        ),
        "",
        para(
            "Hüküm kuralı: K1–K4 birleşik ölçüde en iyi kıyastan iyi ve en az ",
            f"{len(names) - 1}/{len(names)} dönemde iki kıyası da geçiyorsa *geçiyor*; ",
            "birleşik ölçüde kötü ve en fazla 1 dönemde geçiyorsa *geçmiyor*; arası *karışık*. ",
            "Hata için kıyaslar sıfır ve eğitim ortalaması, sıralama için momentum ve geri dönüş.",
        ),
        "",
    ]
    return lines


def _error_section(config, folds, summary):
    names = [f["name"] for f in config["folds"]]
    lines = [
        "## 1. Tahmin hatası",
        "",
        para(
            "R²_oos = 1 − Σ(y − ŷ)² / Σy². Sıfır tahmini tanım gereği 0'dır; pozitif değer ",
            "sıfırdan iyi demektir. Skor kıyasları ve referans satırı getiri tahmini olmadığı ",
            "için bu tabloda yoktur.",
        ),
        "",
    ]
    for target in config["targets"]:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        rows = [
            [
                MODEL_NAMES[m],
                *[pct(f.loc[(m, k), "r2_oos"]) for k in names],
                pct(s.loc[m, "r2_oos"]),
                f"{s.loc[m, 'mse']:.5f}",
            ]
            for m in (*ERROR_BASELINES, MAIN_MODEL)
        ]
        header = ["Model", *names, "K1–K4", "MSE K1–K4"]
        lines += [f"**{TARGET_NAMES[target]}**", "", table(header, rows), ""]
    return lines


def _rank_section(config, folds, summary):
    names = [f["name"] for f in config["folds"]]
    h, k_top = config["horizon"], config["top_k"]
    lines = [
        "## 2. Günlük sıralama başarısı",
        "",
        para(
            "Her gün hisseler arasında tahmin ile gerçekleşen getiri arasındaki Spearman ",
            "korelasyonu (rank IC), sonra günlerin ortalaması. Ardışık günlerin ",
            f"{h} seanslık etiketleri örtüştüğü için t-istatistiği Newey-West ({h - 1} gecikme) ",
            "ile hesaplandı. Sıfır ve eğitim ortalaması sabit olduğu için sıralama üretmez. ",
            f"Son sütun: en yüksek skorlu {k_top} hissenin {h} seanslık ortalama getirisi eksi ",
            "bütün hisselerin ortalaması (maliyetsiz; portföy getirisi değildir).",
        ),
        "",
    ]
    for target in config["targets"]:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        rows = [
            [
                MODEL_NAMES[m],
                *[num(f.loc[(m, k), "ic_mean"]) for k in names],
                num(s.loc[m, "ic_mean"]),
                num(s.loc[m, "ic_t_nw"], 2),
                f"{100 * s.loc[m, 'ic_pos_frac']:.0f}%",
                pct(s.loc[m, "top5_minus_all"]),
            ]
            for m in _rank_models(s.reset_index())
        ]
        header = ["Model", *names, "K1–K4", "NW t", "IC>0 gün", f"Top{k_top}−tümü"]
        lines += [f"**{TARGET_NAMES[target]}**", "", table(header, rows), ""]
    lines += [
        para(
            "Not: Günlük rank IC, hedefe aynı gün bütün hisselerde aynı olan bir sabit ",
            "eklenmesinden etkilenmez (QQQ getirisi, gün ortalaması). Bu yüzden skor kıyaslarının ",
            "değerleri bütün hedeflerde ve deneylerde aynıdır; Ridge farklı hedefe eğitildiği ",
            "için farklıdır.",
        ),
        "",
    ]
    return lines


def _regime_section(config, summary, context):
    universe = _universe(context)
    k_top = config["top_k"]
    days = universe["days"]
    header = [
        "Model",
        "Tümü",
        *[f"{REGIME_NAMES[r]} ({days[r]} g)" for r in REGIMES],
        "Beta-düz. IC",
        "NW t",
        f"İlk {k_top} beta",
        f"İlk {k_top} vol.",
        "Kapsam",
    ]
    lines = [
        "## 3. Rejim ve risk",
        "",
        para(
            "Rejim, etiket penceresinde gerçekleşen QQQ getirisine göre: > +",
            f"{100 * config['diagnostics']['regime_threshold']:.0f}% ↑, < −",
            f"{100 * config['diagnostics']['regime_threshold']:.0f}% ↓, arası yatay. ",
            "Gerçekleşmiş getiriye dayandığı için yalnız tanıdır, sinyal olarak kullanılamaz. ",
            "Beta-düzeltmeli IC'de hedef = hisse getirisi − beta × QQQ getirisi; ",
            f"beta {config['diagnostics']['beta']}. Betası olmayan satırlar yalnız bu ",
            "sütunlardan ve yüksek beta kıyasından düşer. İlk k sütunları: her gün en yüksek ",
            "skorlu hisselerin ortalama beta ve yıllık volatilitesi.",
        ),
        "",
    ]
    for target in config["targets"]:
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        rows = [
            [
                MODEL_NAMES[m],
                num(s.loc[m, "ic_mean"]),
                *[num(s.loc[m, f"ic_{r}"]) for r in REGIMES],
                num(s.loc[m, "ic_beta_mean"]),
                num(s.loc[m, "ic_beta_t_nw"], 2),
                plain(s.loc[m, "top5_beta"]),
                plain(s.loc[m, "top5_vol"]),
                f"{100 * s.loc[m, 'coverage']:.1f}%",
            ]
            for m in _rank_models(s.reset_index())
        ]
        rows.append(
            [
                "Evren (bütün hisseler)",
                "",
                "",
                "",
                "",
                "",
                "",
                plain(universe["beta"]),
                plain(universe["vol"]),
                "",
            ]
        )
        lines += [f"**{TARGET_NAMES[target]}**", "", table(header, rows), ""]
    return lines


def _consistency_section(consistency):
    columns = {
        "mse_beats_sifir": "MSE < sıfır",
        "mse_beats_egitim_ortalamasi": "MSE < ortalama",
        "ic_positive": "IC > 0",
        "ic_beats_momentum_20": "IC > momentum",
        "ic_beats_geri_donus_5": "IC > geri dönüş",
        "ic_beats_en_oynak_20": "IC > en oynak",
        "ic_beats_yuksek_beta": "IC > yüksek beta",
    }
    rows = [
        [TARGET_NAMES[c["target"]], *[f"{c[col]}/{c['folds']}" for col in columns]]
        for _, c in consistency.iterrows()
    ]
    return [
        "## 4. Dönemler arası tutarlılık",
        "",
        table(["Hedef", *columns.values()], rows),
        "",
    ]


def _context_section(config, context, alphas):
    h = config["horizon"]
    grid = config["ridge"]["alpha_grid"]
    targets = list(config["targets"])
    rows = [
        [
            c["fold"],
            f"{c['val_start']:%Y-%m-%d} → {c['val_end']:%Y-%m-%d}",
            c["symbols"],
            pct(c["qqq_mean"]),
            pct(c["stock_mean"]),
            " / ".join(str(c[f"days_{r}"]) for r in REGIMES),
            f"{100 * c['beta_coverage']:.0f}%",
            *[f"{alphas.loc[(t, c['fold']), 'alpha']:.3g}" for t in targets],
        ]
        for _, c in context.iterrows()
    ]
    header = ["Dönem", "Sinyal günleri", "Hisse", f"QQQ {h}g ort.", f"Hisse {h}g ort."]
    header += ["Gün ↑/yatay/↓", "Beta kapsamı", *[f"α {TARGET_NAMES[t]}" for t in targets]]
    return [
        "## 5. Dönem bağlamı",
        "",
        table(header, rows, left=2),
        "",
        para(
            f"α ızgarası {min(grid):g}–{max(grid):g}. α üst sınırdaysa iç validation doğrusal ",
            "sinyal bulamamış, Ridge tahmini eğitim ortalamasına çekilmiştir; sıralaması ise ",
            "hâlâ katsayıların yönünden gelir.",
        ),
        "",
    ]


def _why_section(config, result):
    names = [f["name"] for f in config["folds"]]
    exp = result["exposures"]
    fic = result["feature_ic"].pivot_table(
        index=["feature", "fold"], columns="partition", values="ic_mean"
    )
    rows = []
    for target in config["targets"]:
        for k in names:
            e = exp.loc[exp["target"].eq(target) & exp["fold"].eq(k)].set_index("feature")["corr"]
            top = e.abs().sort_values(ascending=False).index[:3]
            rows.append([TARGET_NAMES[target], k, ", ".join(f"{x} {e[x]:+.2f}" for x in top)])
    lead = exp.assign(a=exp["corr"].abs()).groupby("feature")["a"].mean()
    lead = [x.removeprefix("xs_") for x in lead.sort_values(ascending=False).index[:4]]
    ic_rows = [
        [
            x,
            *[
                f"{num(fic.loc[(x, k), 'train'])} → {num(fic.loc[(x, k), 'validation'])}"
                for k in names
            ],
        ]
        for x in lead
    ]
    return [
        "## 6. Neden? Ridge neye bahis oynuyor?",
        "",
        "Validation yılında Ridge skorunun en çok korele olduğu üç girdi:",
        "",
        table(["Hedef", "Dönem", "Skor–girdi korelasyonu"], rows, left=3),
        "",
        para(
            "Bu özelliklerin tek başına günlük rank IC'si (eğitim → validation). Gün içi sıraya ",
            "çevirmek günlük rank IC'yi değiştirmez. Tablo açıklama içindir, model seçiminde ",
            "kullanılmadı:",
        ),
        "",
        table(["Özellik", *names], ic_rows),
        "",
    ]


def _coefficient_section(config, coefficients):
    names = [f["name"] for f in config["folds"]]
    n = len(names)
    lines = ["## 7. Ridge katsayıları", ""]
    for target in config["targets"]:
        c = coefficients.loc[coefficients["target"].eq(target)].pivot_table(
            index="feature", columns="fold", values="coef"
        )[names]
        # α dönemden döneme değiştiği için büyüklük yerine işaret ve sıra karşılaştırılır.
        rel = c / c.abs().max()
        order = rel.abs().mean(axis=1).sort_values(ascending=False).index[:10]
        rows = []
        for x in order:
            pos = int((c.loc[x] > 0).sum())
            sign = f"{pos}/{n} +" if pos >= n / 2 else f"{n - pos}/{n} −"
            rows.append([x, *[f"{rel.loc[x, k]:+.2f}" for k in names], sign])
        lines += [
            para(
                f"**{TARGET_NAMES[target]}** — her dönemde en büyük katsayıya bölünmüş ",
                "(−1…+1), ilk 10 girdi:",
            ),
            "",
            table(["Girdi", *names, "İşaret"], rows),
            "",
        ]
    lines += [
        "Ham katsayılar (standart sapma başına getiri) `ridge_coefficients.csv` içindedir.",
        "",
    ]
    return lines


TARGET_SETUP = {
    "raw": "`{col}`: hissenin t+1 açılış → t+H kapanış getirisi.",
    "vs_qqq": "`{col}`: hisse getirisi − aynı pencerede QQQ getirisi.",
    "xs": "`{col}`: hisse getirisi − aynı gün aynı bölmedeki hisselerin ortalama getirisi.",
}
FEATURE_SETUP = {
    "raw": "manifestteki 24 `feature_columns`, ham değerleriyle.",
    "xs_rank": para(
        "manifestteki özelliklerden aynı gün bütün hisselerde aynı olanlar (market_*) ",
        "çıkarıldı; kalan {n} özellik her gün hisseler arasında sıraya çevrildi (−0,5…+0,5).",
    ),
}


def _setup_section(config):
    h = config["horizon"]
    lines = [
        "## Kurulum",
        "",
        para(
            f"- Satırlar: veri setindeki `partition_membership.parquet` (H = {h}) ve ",
            "`volume_features_ready = True`. Eğitim etiketleri eğitim sonunu aşmaz (purge).",
        ),
        *[f"- Hedef: {TARGET_SETUP[t].format(col=c)}" for t, c in config["targets"].items()],
        f"- Ridge girdileri: {FEATURE_SETUP[config['feature_set']].format(n=len(config['features']))}",
        para(
            "- Ridge: girdiler ve eğitim hedefi eğitim dönemi %1/%99 kantillerine kırpılır, ",
            "girdiler standartlaştırılır. α eğitimin son 252 seansında (etiketi bu pencereden ",
            "önce biten satırlarla eğitilerek) MSE ile seçilir, sonra bütün eğitime yeniden fit ",
            "edilir.",
        ),
        para(
            "- Skor kıyasları eğitim gerektirmez: momentum = `return_20`, geri dönüş = ",
            "−`return_5`, en oynak = `volatility_20`, yüksek beta = `beta_252`.",
        ),
    ]
    if config.get("reference_model"):
        lines.append(
            "- Referans: Deney 1 Ridge'i (ham 24 girdi, hissenin kendi getirisine eğitilmiş)."
        )
    lines += [
        "- Değerlendirme her zaman kırpılmamış hedefle yapılır.",
        "",
        "## Dosyalar",
        "",
        para(
            "`fold_metrics.csv`, `pooled_metrics.csv`, `consistency.csv`, `period_context.csv`, ",
            "`ridge_coefficients.csv`, `ridge_alpha.csv`, `ridge_exposure.csv`, `feature_ic.csv`, ",
            "`daily_ic.parquet`, `predictions.parquet`, `config.json`. Her çalıştırma ",
            "`output/experiments/runs.csv` kaydına eklenir.",
        ),
        "",
    ]
    return lines


def write_report(output, config, result, summary, consistency, context):
    folds = result["folds"]
    commit = (config["git_commit"] or "?")[:7]
    dirty = " (kaydedilmemiş değişiklik var)" if config["git_dirty"] else ""
    lines = [
        f"# Aşama 1 · {config['title']} (H = {config['horizon']} seans)",
        "",
        para(
            f"Çalıştırma `{config['run_id']}` · commit `{commit}`{dirty} · ",
            f"snapshot `{config['snapshot']}` · veri seti {config['dataset_version']}",
        ),
        "",
        para(
            "Değerlendirme yalnız K1–K4 validation yıllarında yapıldı. ",
            f"**{config['test_start']} ve sonrası (test dönemi) okunmadı.**",
        ),
        "",
        "## Kısa cevap",
        "",
        *_short_answer(config, folds, summary, consistency, context, result["exposures"]),
        *_error_section(config, folds, summary),
        *_rank_section(config, folds, summary),
        *_regime_section(config, summary, context),
        *_consistency_section(consistency),
        *_context_section(config, context, result["alphas"].set_index(["target", "fold"])),
        *_why_section(config, result),
        *_coefficient_section(config, result["coefficients"]),
        *_setup_section(config),
    ]
    (output / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
