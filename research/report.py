"""Aşama 1 raporu: 'Ana model basit kıyasları geçiyor mu, hangi hedefte, hangi dönemde?'"""

from __future__ import annotations

import math

from .data import REGIMES
from .experiment import (
    ERROR_BASELINES,
    LGBM_REFERENCE,
    NEUTRAL_MODEL,
    RANK_BASELINES,
    REFERENCE_MODEL,
    RISK_BASELINES,
)

TARGET_NAMES = {
    "raw": "Hissenin kendi getirisi",
    "vs_qqq": "QQQ'ya göre getiri",
    "xs": "Gün içi ortalamadan arındırılmış getiri",
    "resid": "Betadan arındırılmış artık getiri",
}
MODEL_NAMES = {
    "sifir": "Sıfır",
    "egitim_ortalamasi": "Eğitim ortalaması",
    "momentum_20": "Momentum (return_20)",
    "geri_donus_5": "Geri dönüş (−return_5)",
    "en_oynak_20": "En oynak (volatility_20)",
    "yuksek_beta": "Yüksek beta (beta_252)",
    "ridge": "**Ridge**",
    "lightgbm": "**LightGBM**",
    "ridge_beta_notr": "Ridge · beta-nötrlenmiş skor",
    "ridge_deney1": "Ridge · Deney 1 (referans)",
    "lightgbm_4a": "LightGBM 4a (aynı satırlar)",
}
MAIN_LABELS = {"ridge": "Ridge", "lightgbm": "LightGBM"}
REGIME_NAMES = {"yukari": "QQQ ↑", "yatay": "Yatay", "asagi": "QQQ ↓"}
CRITERIA = {
    "risk_notr": {
        "info": NEUTRAL_MODEL,
        "info_label": "Beta-nötrlenmiş Ridge (bilgi)",
        "passed": "",
        "failed": " Sıradaki adım: Deney 4 (uzun momentum: 60/120/250 seans).",
    },
    "lgbm_4a": {
        "info": REFERENCE_MODEL,
        "info_label": "Ridge · Deney 1 (bilgi)",
        "passed": " Sıradaki adım: Deney 4b (piyasa durumu özellikleri).",
        "failed": (
            " Sonuç raporlanır; Deney 4b yine denenir. 4b de geçemezse Deney 5 "
            "(üç bariyerli hedef)."
        ),
    },
}


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


def _label(config):
    return config.get("main_label") or MAIN_LABELS[config["main_model"]]


def _name(model, config):
    return f"**{_label(config)}**" if model == config["main_model"] else MODEL_NAMES[model]


def _rank_models(summary, main):
    present = set(summary["model"])
    order = (main, NEUTRAL_MODEL, LGBM_REFERENCE, REFERENCE_MODEL, *RANK_BASELINES)
    return [m for m in (*order, *RISK_BASELINES) if m in present]


def _universe(context):
    days = context[[f"days_{r}" for r in REGIMES]].sum(axis=1)
    return {
        "beta": float((context["universe_beta_252"] * days).sum() / days.sum()),
        "vol": float((context["universe_volatility_20"] * days).sum() / days.sum()),
        "days": {r: int(context[f"days_{r}"].sum()) for r in REGIMES},
    }


def criteria_rows(kind, model, target, folds, summary, universe):
    """Önceden yazılı başarı kriterleri; her satır (ad, eşik, değer, geçti)."""
    s = summary.loc[summary["target"].eq(target)].set_index("model")
    f = folds.loc[folds["target"].eq(target) & folds["model"].eq(model)]
    m = s.loc[model]
    momentum = s.loc["momentum_20", "ic_mean"]
    positive = int((f["ic_mean"] > 0).sum())
    ic_label = "Artık IC" if kind == "risk_notr" else "Rank IC"
    down = (
        "QQQ ↓ rejiminde IC",
        "NW t > −2",
        f"{num(m['ic_asagi'])} (t {num(m['ic_asagi_t_nw'], 2)})",
        m["ic_asagi_t_nw"] > -2,
    )
    rows = [
        (
            f"1a. {ic_label} > momentum",
            "birleşik",
            f"{num(m['ic_mean'])} vs {num(momentum)}",
            m["ic_mean"] > momentum,
        ),
        ("1b. Pozitif IC'li dönem", "≥ 3/4", f"{positive}/{len(f)}", positive >= 3),
    ]
    if kind == "risk_notr":
        gap = m["top5_beta"] - universe["beta"]
        return [
            *rows,
            (
                "2a. Skor–beta sıra korelasyonu",
                "−0.20 < r < 0.20",
                num(m["corr_beta"]),
                abs(m["corr_beta"]) < 0.2,
            ),
            (
                "2b. İlk 5 beta − evren",
                "±0.20",
                f"{num(gap, 2)} ({plain(m['top5_beta'])})",
                abs(gap) <= 0.2,
            ),
            (f"3. {down[0]}", *down[1:]),
        ]
    return [
        *rows,
        (f"2. {down[0]}", *down[1:]),
        (
            "3. Skor–beta sıra korelasyonu",
            "−0.30 < r < 0.30",
            num(m["corr_beta"]),
            abs(m["corr_beta"]) < 0.3,
        ),
    ]


def _criteria_lines(config, folds, summary, context):
    spec = CRITERIA[config["criteria"]]
    main = config["main_model"]
    universe = _universe(context)
    present = set(summary["model"])
    infos = [m for m in dict.fromkeys((LGBM_REFERENCE, spec["info"])) if m in present]
    lines = []
    for target in config["targets"]:
        own = criteria_rows(config["criteria"], main, target, folds, summary, universe)
        others = [
            criteria_rows(config["criteria"], m, target, folds, summary, universe) for m in infos
        ]
        rows = [
            [
                a[0],
                a[1],
                *[f"{'✅' if c[3] else '❌'} {c[2]}" for c in (a, *(o[i] for o in others))],
            ]
            for i, a in enumerate(own)
        ]
        passed = all(c[3] for c in own)
        label = _label(config)
        info_labels = [
            spec["info_label"] if m == spec["info"] else f"{MODEL_NAMES[m]} (bilgi)" for m in infos
        ]
        lines += [
            para(
                f"**Önceden yazılı başarı kriteri: {'karşılandı' if passed else 'karşılanmadı'}.** ",
                f"Karar {label} satırına göre verilir; diğer sütunlar bilgi içindir.",
                spec["passed"] if passed else spec["failed"],
            ),
            "",
            table(["Kriter", "Eşik", label, *info_labels], rows, left=2),
            "",
            para(
                '"Düşen piyasada belirgin negatif değil" ölçüsü: QQQ ↓ rejimindeki günlük IC\'lerin ',
                "Newey-West t-istatistiği −2'den büyük. Evren betası: ",
                f"{plain(universe['beta'])}.",
            ),
            "",
            *_thin_ic_note(config, folds, target),
        ]
    return lines


def _ic_cell(row):
    """Dönem IC'si; IC günlerin %90'ından azında hesaplanabildiyse gün sayısıyla."""
    cell = num(row["ic_mean"])
    if not _missing(row.get("ic_days")) and row["ic_days"] < 0.9 * row["val_days"]:
        cell += f" ({int(row['ic_days'])}/{int(row['val_days'])} g)"
    return cell


def _thin_ic_note(config, folds, target):
    main = config["main_model"]
    f = folds.loc[folds["target"].eq(target) & folds["model"].eq(main)]
    thin = f.loc[f["ic_days"] < 0.5 * f["val_days"]]
    if thin.empty:
        return []
    parts = ", ".join(
        f"{r.fold} ({int(r.ic_days)}/{int(r.val_days)} gün)" for r in thin.itertuples()
    )
    return [
        para(
            f"Dikkat: {_label(config)} şu dönemlerde günlerin çoğunda bütün hisselere aynı skoru ",
            f"verdi; o dönemin IC'si az sayıda güne dayanıyor: {parts}. Kural değişmedi; ",
            "sabit skorlu günler IC'ye girmez (bölüm 2'de bilgi için 0 sayılmış hali var).",
        ),
        "",
    ]


def verdict(target, folds, summary, consistency, main):
    """Kural tabanlı kısa hüküm; eşikler raporda açıkça yazılıdır."""
    s = summary.loc[summary["target"].eq(target)].set_index("model")
    c = consistency.loc[consistency["target"].eq(target)].iloc[0]
    f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
    n = int(c["folds"])

    best_error = max(ERROR_BASELINES, key=lambda m: s.loc[m, "r2_oos"])
    error_wins = min(c[f"mse_beats_{m}"] for m in ERROR_BASELINES)
    main_r2 = s.loc[main, "r2_oos"]
    if main_r2 > s.loc[best_error, "r2_oos"] and error_wins >= n - 1:
        error = "geçiyor"
    elif main_r2 < s.loc[best_error, "r2_oos"] and error_wins <= 1:
        error = "geçmiyor"
    else:
        error = "karışık"

    best_rank = max(RANK_BASELINES, key=lambda m: s.loc[m, "ic_mean"])
    rank_wins = min(c[f"ic_beats_{m}"] for m in RANK_BASELINES)
    main_ic = s.loc[main, "ic_mean"]
    if (
        main_ic > s.loc[best_rank, "ic_mean"]
        and main_ic > 0
        and rank_wins >= n - 1
        and c["ic_positive"] >= n - 1
    ):
        rank = "geçiyor"
    elif main_ic < s.loc[best_rank, "ic_mean"] and rank_wins <= 1:
        rank = "geçmiyor"
    else:
        rank = "karışık"

    names = sorted({fold for _, fold in f.index})
    error_folds = [
        k
        for k in names
        if f.loc[(main, k), "mse"] < min(f.loc[(m, k), "mse"] for m in ERROR_BASELINES)
    ]
    rank_folds = [
        k
        for k in names
        if f.loc[(main, k), "ic_mean"] > max(0, *(f.loc[(m, k), "ic_mean"] for m in RANK_BASELINES))
    ]
    return {
        "error": error,
        "rank": rank,
        "error_folds": error_folds,
        "rank_folds": rank_folds,
        "main_r2": main_r2,
        "best_error": best_error,
        "best_error_r2": s.loc[best_error, "r2_oos"],
        "main_ic": main_ic,
        "main_t": s.loc[main, "ic_t_nw"],
        "best_rank": best_rank,
        "best_rank_ic": s.loc[best_rank, "ic_mean"],
    }


def _short_answer(config, folds, summary, consistency, context, exposures):
    names = [f["name"] for f in config["folds"]]
    main = config["main_model"]
    label = _label(config)
    universe = _universe(context)
    lines = _criteria_lines(config, folds, summary, context) if config.get("criteria") else []
    for target in config["targets"]:
        v = verdict(target, folds, summary, consistency, main)
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        lines += [
            f"**{TARGET_NAMES[target]}**",
            para(
                f"- Tahmin hatası: {label} **{v['error']}**. K1–K4 R²: {label} ",
                f"{pct(v['main_r2'])}, en iyi sabit kıyas ({MODEL_NAMES[v['best_error']]}) ",
                f"{pct(v['best_error_r2'])}. İki sabit kıyası birden geçtiği dönemler: ",
                f"{', '.join(v['error_folds']) or 'yok'}.",
            ),
            para(
                f"- Günlük sıralama: {label} **{v['rank']}**. K1–K4 rank IC: {label} ",
                f"{num(v['main_ic'])} (Newey-West t = {num(v['main_t'], 2)}), en iyi skor ",
                f"kıyası ({MODEL_NAMES[v['best_rank']]}) {num(v['best_rank_ic'])}. ",
                "Momentum ve geri dönüşü birlikte geçip pozitif kaldığı dönemler: ",
                f"{', '.join(v['rank_folds']) or 'yok'}.",
            ),
            para(
                "- Rejim ve risk: rank IC QQQ ↑ / yatay / QQQ ↓ = ",
                " / ".join(num(s.loc[main, f"ic_{r}"]) for r in REGIMES),
                f"; ham getiriye göre IC {num(s.loc[main, 'ic_raw_mean'])}, ",
                f"beta-düzeltmeli IC {num(s.loc[main, 'ic_beta_mean'])}; ",
                f"skor–beta sıra korelasyonu {num(s.loc[main, 'corr_beta'], 2)}. ",
                f"Seçilen ilk {config['top_k']} hissenin ortalama betası ",
                f"{plain(s.loc[main, 'top5_beta'])}, volatilitesi ",
                f"{plain(s.loc[main, 'top5_vol'])} ",
                f"(evren: {plain(universe['beta'])} / {plain(universe['vol'])}).",
            ),
        ]
        for ref_name in (LGBM_REFERENCE, REFERENCE_MODEL):
            if ref_name not in s.index:
                continue
            ref = s.loc[ref_name]
            better = "üstünde" if v["main_ic"] > ref["ic_mean"] else "altında"
            lines.append(
                para(
                    f"- {MODEL_NAMES[ref_name]}: K1–K4 rank IC {num(ref['ic_mean'])} ",
                    f"(t = {num(ref['ic_t_nw'], 2)}); bu deneyin ana modeli bunun {better}.",
                )
            )
        lines.append("")
    lead = exposures.assign(a=exposures["corr"].abs()).groupby("feature")["a"].mean()
    lead = lead.sort_values(ascending=False)
    lines += [
        para(
            f"{label} skorunun validation yıllarındaki ortalama mutlak korelasyonu en yüksek ",
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
    main = config["main_model"]
    lines = [
        "## 1. Tahmin hatası",
        "",
        para(
            "R²_oos = 1 − Σ(y − ŷ)² / Σy². Sıfır tahmini tanım gereği 0'dır; pozitif değer ",
            "sıfırdan iyi demektir. Skor kıyasları getiri tahmini olmadığı için bu tabloda yoktur; ",
            "referans satırı yalnız aynı hedefe eğitildiyse gösterilir.",
        ),
        "",
    ]
    for target in config["targets"]:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        models = [*ERROR_BASELINES, main]
        for ref_name in (LGBM_REFERENCE, REFERENCE_MODEL):
            if ref_name in s.index and s.loc[ref_name, "kind"] == "forecast":
                models.append(ref_name)
        rows = [
            [
                _name(m, config),
                *[pct(f.loc[(m, k), "r2_oos"]) for k in names],
                pct(s.loc[m, "r2_oos"]),
                f"{s.loc[m, 'mse']:.5f}",
            ]
            for m in models
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
            "bütün hisselerin ortalaması (maliyetsiz; portföy getirisi değildir). Eşit skorlu ",
            "hisseler kesimde kalan yer oranında ağırlık alır.",
        ),
        "",
    ]
    for target in config["targets"]:
        f = folds.loc[folds["target"].eq(target)].set_index(["model", "fold"])
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        rows = [
            [
                _name(m, config),
                *[_ic_cell(f.loc[(m, k)]) for k in names],
                num(s.loc[m, "ic_mean"]),
                num(s.loc[m, "ic_t_nw"], 2),
                f"{100 * s.loc[m, 'ic_pos_frac']:.0f}%",
                pct(s.loc[m, "top5_minus_all"]),
            ]
            for m in _rank_models(s.reset_index(), config["main_model"])
        ]
        header = ["Model", *names, "K1–K4", "NW t", "IC>0 gün", f"Top{k_top}−tümü"]
        lines += [f"**{TARGET_NAMES[target]}**", "", table(header, rows), ""]
    thin = summary.loc[summary["ic_days"] < summary["val_days"]]
    if not thin.empty:
        lines += [
            para(
                "IC'si bazı günlerde hesaplanamayan satırlar (o gün bütün hisselere aynı skor; ",
                "sıralama yok): ",
                "; ".join(
                    ("" if len(config["targets"]) == 1 else f"{TARGET_NAMES[r.target]} · ")
                    + f"{_name(r.model, config)} {int(r.ic_days)}/{int(r.val_days)} gün, "
                    f"sabit günler 0 sayılırsa K1–K4 IC {num(r.ic_zero_filled)}"
                    for r in thin.itertuples()
                ),
                ". Dönem hücresinde parantez: IC günü / validation günü (%90'ın altındaysa).",
            ),
            "",
        ]
    note = [
        "Not: Günlük rank IC, hedefe aynı gün bütün hisselerde aynı olan bir sabit ",
        "eklenmesinden etkilenmez (QQQ getirisi, gün ortalaması). ",
    ]
    if "resid" in config["targets"]:
        note += [
            "Artık hedefte çıkarılan beta × QQQ hisseden hisseye değiştiği için skor ",
            "kıyaslarının değerleri ham getiri hedefindekinden farklıdır.",
        ]
    else:
        note += [
            "Bu yüzden skor kıyaslarının değerleri ham, QQQ'ya göre ve gün içi hedeflerde ",
            "aynıdır; eğitilen modeller hedefe göre farklıdır.",
        ]
    lines += [para(*note), ""]
    return lines


def _regime_section(config, summary, context):
    universe = _universe(context)
    k_top = config["top_k"]
    days = universe["days"]
    regime_header = [
        "Model",
        "Hedefe göre IC",
        "Ham getiriye göre IC",
        "Beta-düz. IC",
        *[f"{REGIME_NAMES[r]} ({days[r]} g)" for r in REGIMES],
        "NW t (↓)",
    ]
    risk_header = [
        "Model",
        "Skor–beta sıra kor.",
        "Skor–vol. sıra kor.",
        f"İlk {k_top} beta",
        f"İlk {k_top} vol.",
        "Kapsam",
    ]
    threshold = 100 * config["diagnostics"]["regime_threshold"]
    lines = [
        "## 3. Rejim ve risk",
        "",
        para(
            f"Rejim, etiket penceresinde gerçekleşen QQQ getirisine göre: > +{threshold:.0f}% ↑, ",
            f"< −{threshold:.0f}% ↓, arası yatay. Gerçekleşmiş getiriye dayandığı için yalnız ",
            "tanıdır, sinyal olarak kullanılamaz. Ham getiriye göre IC'de hedef hissenin kendi ",
            "getirisidir; beta-düzeltmeli IC'de hisse getirisi − beta × QQQ getirisi. ",
            f"Beta {config['diagnostics']['beta']}; betası olmayan satır bu sütunlardan ve ",
            "yüksek beta kıyasından düşer. Sıra korelasyonları: her gün skor ile beta / ",
            "volatility_20 arasındaki Spearman korelasyonunun ortalaması. İlk k sütunları: her gün ",
            "en yüksek skorlu hisselerin ortalama beta ve yıllık volatilitesi.",
        ),
        "",
    ]
    for target in config["targets"]:
        s = summary.loc[summary["target"].eq(target)].set_index("model")
        models = _rank_models(s.reset_index(), config["main_model"])
        regime_rows = [
            [
                _name(m, config),
                num(s.loc[m, "ic_mean"]),
                num(s.loc[m, "ic_raw_mean"]),
                num(s.loc[m, "ic_beta_mean"]),
                *[num(s.loc[m, f"ic_{r}"]) for r in REGIMES],
                num(s.loc[m, "ic_asagi_t_nw"], 2),
            ]
            for m in models
        ]
        risk_rows = [
            [
                _name(m, config),
                num(s.loc[m, "corr_beta"], 2),
                num(s.loc[m, "corr_vol"], 2),
                plain(s.loc[m, "top5_beta"]),
                plain(s.loc[m, "top5_vol"]),
                f"{100 * s.loc[m, 'coverage']:.1f}%",
            ]
            for m in models
        ]
        risk_rows.append(
            ["Evren (bütün hisseler)", "", "", plain(universe["beta"]), plain(universe["vol"]), ""]
        )
        lines += [
            f"**{TARGET_NAMES[target]}** — rejim",
            "",
            table(regime_header, regime_rows),
            "",
            f"**{TARGET_NAMES[target]}** — risk maruziyeti",
            "",
            table(risk_header, risk_rows),
            "",
        ]
    return lines


def _consistency_section(config, consistency):
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
        f"{_label(config)} satırının dönem bazında kıyasları geçtiği dönem sayısı.",
        "",
        table(["Hedef", *columns.values()], rows),
        "",
    ]


def _tuning(config, result):
    """Dönem bağlamı tablosu için (başlık, hedef×dönem → değer, açıklama)."""
    if result["trees"] is not None:
        trees = result["trees"].set_index(["target", "fold"])["n_trees"]
        lgbm = config["lightgbm"]
        note = para(
            f"Ağaç sayısı: en fazla {lgbm['max_trees']}, early stopping ",
            f"{lgbm['early_stopping_rounds']} tur; eğitimin son 252 seansında (purge'lü) bulunup ",
            "bütün eğitime aynı sayıyla yeniden fit edildi. Çok az ağaç, iç validation'da ilk ",
            "ağaçlardan sonra iyileşme olmadığını gösterir.",
        )
        return "Ağaç", {k: f"{v:d}" for k, v in trees.items()}, note
    alphas = result["alphas"].set_index(["target", "fold"])["alpha"]
    grid = config["ridge"]["alpha_grid"]
    note = para(
        f"α ızgarası {min(grid):g}–{max(grid):g}. α üst sınırdaysa iç validation doğrusal ",
        "sinyal bulamamış, Ridge tahmini eğitim ortalamasına çekilmiştir; sıralaması ise ",
        "hâlâ katsayıların yönünden gelir.",
    )
    return "α", {k: f"{v:.3g}" for k, v in alphas.items()}, note


def _context_section(config, context, result):
    h = config["horizon"]
    targets = list(config["targets"])
    label, tuning, note = _tuning(config, result)
    rows = [
        [
            c["fold"],
            f"{c['val_start']:%Y-%m-%d} → {c['val_end']:%Y-%m-%d}",
            c["symbols"],
            pct(c["qqq_mean"]),
            pct(c["stock_mean"]),
            " / ".join(str(c[f"days_{r}"]) for r in REGIMES),
            f"{100 * c['beta_coverage']:.0f}%",
            *[tuning[(t, c["fold"])] for t in targets],
            *(
                [f"{int(c['excluded_train'])} / {int(c['excluded_validation'])}"]
                if "excluded_train" in c
                else []
            ),
        ]
        for _, c in context.iterrows()
    ]
    header = ["Dönem", "Sinyal günleri", "Hisse", f"QQQ {h}g ort.", f"Hisse {h}g ort."]
    header += ["Gün ↑/yatay/↓", "Beta kapsamı", *[f"{label} {TARGET_NAMES[t]}" for t in targets]]
    if "excluded_train" in context:
        reason = "Piyasa özelliği eksik" if config.get("universe") == "market" else "Betasız"
        header.append(f"{reason}, dışarıda (eğitim / val.)")
    return ["## 5. Dönem bağlamı", "", table(header, rows, left=2), "", note, ""]


def _why_section(config, result):
    names = [f["name"] for f in config["folds"]]
    label = _label(config)
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
    # Tek başına IC yalnız hisseye özgü özelliklerde anlamlı; market_* o gün herkeste aynı.
    known = set(fic.index.get_level_values("feature"))
    lead = [x.removeprefix("xs_") for x in lead.sort_values(ascending=False).index]
    lead = [x for x in lead if x in known and not x.startswith("market_")][:4]
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
        f"## 6. Neden? {label} neye bahis oynuyor?",
        "",
        f"Validation yılında {label} skorunun en çok korele olduğu üç girdi:",
        "",
        table(["Hedef", "Dönem", "Skor–girdi korelasyonu"], rows, left=3),
        "",
        para(
            "Bu özelliklerin tek başına günlük rank IC'si (eğitim → validation). Gün içi sıraya ",
            "çevirmek günlük rank IC'yi değiştirmez. Aynı gün bütün hisselerde aynı olan ",
            "özelliklerin (market_*) günlük rank IC'si tanımsızdır. Tablo açıklama içindir, ",
            "model seçiminde kullanılmadı:",
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


def _direction(delta):
    return "+" if delta > 1e-9 else "−" if delta < -1e-9 else "düz"


def _trees_table(config, result, target, names):
    trees = result["trees"].loc[result["trees"]["target"].eq(target)]
    models = list(dict.fromkeys(trees["model"]))
    t = trees.set_index(["model", "fold"])
    header = ["Dönem"]
    for m in models:
        short = _label(config) if m == config["main_model"] else MODEL_NAMES[m]
        header += [f"{short} ağaç", f"{short} iç l2"]
    rows = [
        [
            k,
            *[
                cell
                for m in models
                for cell in (f"{t.loc[(m, k), 'n_trees']:d}", f"{t.loc[(m, k), 'inner_mse']:.5f}")
            ],
        ]
        for k in names
    ]
    return table(header, rows)


def _curve_table(config, curve, names):
    counts = config["lightgbm"]["tree_curve"]["counts"]
    header = ["Dönem", "Ölçü", *[str(n) for n in counts], "Seçilen"]
    rows = []
    for k in names:
        fold = curve.loc[curve["fold"].eq(k)].set_index("n_trees")
        chosen = int(fold["chosen"].iloc[0])
        for metric, title, fmt in (("l2", "iç l2", "{:.5f}"), ("rank_ic", "iç rank IC", "{:+.3f}")):
            cells = []
            for n in counts:
                cell = fmt.format(fold.loc[n, metric])
                cells.append(f"**{cell}**" if n == chosen else cell)
            rows.append([k, title, *cells, f"{fmt.format(fold.loc[chosen, metric])} (n={chosen})"])
    return table(header, rows, left=2)


def _importance_lines(config, imp, target, names):
    main = config["main_model"]
    part = imp.loc[imp["target"].eq(target) & imp["model"].eq(main)]
    tops = {
        k: part.loc[part["fold"].eq(k)].sort_values("gain_share", ascending=False).head(10)
        for k in names
    }
    rows = [
        [
            str(i + 1),
            *[
                (
                    f"{tops[k].iloc[i]['feature']} {100 * tops[k].iloc[i]['gain_share']:.0f}%"
                    if i < len(tops[k]) and tops[k].iloc[i]["gain_share"] > 0
                    else "—"
                )
                for k in names
            ],
        ]
        for i in range(10)
    ]
    market = []
    for feature in ("market_qqq_return_20", "market_spy_return_20"):
        cells = []
        for k in names:
            fold = part.loc[part["fold"].eq(k)].sort_values("gain_share", ascending=False)
            share = float(fold.set_index("feature").loc[feature, "gain_share"])
            rank = int(fold["feature"].tolist().index(feature)) + 1
            cells.append(f"{k} {100 * share:.0f}%" + (f" ({rank}.)" if share > 0 else ""))
        market.append(f"`{feature}`: " + ", ".join(cells))
    lines = [
        para(
            f"**{TARGET_NAMES[target]}** — {_label(config)} kazanç (gain) payı, dönem başına ",
            "ilk 10 özellik (kullanılmayan özellik —):",
        ),
        "",
        table(["Sıra", *names], rows),
        "",
        "Piyasa özellikleri kullanılıyor mu? " + " · ".join(market) + ".",
        "",
    ]
    if config.get("market_features"):
        new = set(config["market_features"])
        shares = []
        for k in names:
            fold = part.loc[part["fold"].eq(k)]
            used = fold.loc[fold["feature"].isin(new) & fold["gain_share"].gt(0)]
            detail = ", ".join(
                f"{r.feature} {100 * r.gain_share:.0f}%"
                for r in used.sort_values("gain_share", ascending=False).itertuples()
            )
            total = float(fold.loc[fold["feature"].isin(new), "gain_share"].sum())
            shares.append([k, f"{100 * total:.0f}%", detail or "—"])
        lines += [
            f"6 yeni piyasa durumu özelliğinin toplam kazanç payı ({_label(config)}):",
            "",
            table(["Dönem", "Toplam pay", "Kullanılanlar"], shares, left=1),
            "",
        ]
    return lines


def _dependence_table(dep, target, condition, names):
    part = dep.loc[dep["target"].eq(target) & dep["condition"].eq(condition)]
    quantiles = sorted(part["quantile"].unique())
    rows = []
    for k in names:
        fold = part.loc[part["fold"].eq(k)]
        deltas = {}
        for subset, title in (("pozitif", "> 0"), ("negatif", "< 0")):
            sub = fold.loc[fold["subset"].eq(subset)].set_index("quantile")
            if sub.empty:
                rows.append([k, f"{title} (0 satır)", *["—"] * len(quantiles), "—"])
                deltas[subset] = None
                continue
            deltas[subset] = sub.loc[quantiles[-1], "pd"] - sub.loc[quantiles[0], "pd"]
            rows.append(
                [
                    k,
                    f"{title} ({int(sub['rows'].iloc[0])} satır)",
                    *[pct(sub.loc[q, "pd"]) for q in quantiles],
                    pct(deltas[subset]),
                ]
            )
        if None in deltas.values():
            rows[-1].append("— (boş alt küme)")
        else:
            same = _direction(deltas["pozitif"]) == _direction(deltas["negatif"])
            rows[-1].append("aynı" if same else "**farklı**")
        rows[-2].append("")
    header = ["Dönem", f"`{condition}`", *[f"vol q{100 * q:.0f}" for q in quantiles]]
    header += ["Δ (q90 − q10)", "Yön"]
    return table(header, rows, left=2)


def _lgbm_section(config, result):
    names = [f["name"] for f in config["folds"]]
    dep = result["partial_dependence"]
    curve = result["tree_curve"]
    pd_spec = config["lightgbm"]["partial_dependence"]
    lines = ["## 7. LightGBM ayrıntıları", ""]
    for target in config["targets"]:
        lines += [
            f"**{TARGET_NAMES[target]}** — seçilen ağaç sayısı ve iç validation l2",
            "",
            _trees_table(config, result, target, names),
            "",
        ]
        if curve is not None:
            part = curve.loc[curve["target"].eq(target)]
            lines += [
                para(
                    f"**{TARGET_NAMES[target]}** — early stopping incelemesi (4a ayarları, ",
                    "24 özellik). İç bölmede early stopping olmadan "
                    f"{config['lightgbm']['max_trees']} ağaç eğitildi; her ağaç sayısında iç ",
                    "validation l2'si (kırpılmamış hedef, early stopping'in ölçüsü) ve günlük ",
                    "rank IC'si. Kalın hücre ve son sütun 4a'nın seçtiği ağaç sayısı. ",
                    "Yalnız rapor; seçim değişmez.",
                ),
                "",
                _curve_table(config, part, names),
                "",
            ]
        lines += _importance_lines(config, result["importance"], target, names)
        lines += [
            para(
                f"**{TARGET_NAMES[target]}** — basit kısmi bağımlılık: `{pd_spec['feature']}` ",
                "sabit değerlere çekildiğinde ortalama tahmini 20 seanslık getiri, koşul ",
                "özelliği > 0 ve < 0 alt kümelerinde ayrı. Izgara o dönemin validation ",
                "satırlarındaki volatilite kantilleri. Yön iki alt kümede farklıysa model ",
                "piyasa durumuna bağlı (koşullu) bir volatilite ilişkisi kurmuş demektir.",
            ),
            "",
        ]
        for condition in dict.fromkeys(dep["condition"]):
            lines += [_dependence_table(dep, target, condition, names), ""]
    files = ["lgbm_trees.csv", "lgbm_importance.csv", "lgbm_partial_dependence.csv"]
    if curve is not None:
        files.append("lgbm_tree_curve.csv")
    lines += ["Ayrıntılar: " + ", ".join(f"`{f}`" for f in files) + ".", ""]
    return lines


def _coverage_section(config):
    coverage = config.get("market_coverage")
    if not coverage:
        return []
    rows = [[r["fold"], r["partition"], r["rows"], r["missing_any"]] for r in coverage["summary"]]
    missing = [r for r in coverage["detail"] if r["missing"] > 0]
    detail = [
        [
            r["fold"],
            r["partition"],
            f"`{r['feature']}`",
            r["missing"],
            f"{r['first_missing'][:10]} → {r['last_missing'][:10]}",
        ]
        for r in missing
    ]
    excluded = sum(r["missing_any"] for r in coverage["summary"])
    lines = [
        "## 8. Yeni özelliklerin kapsamı",
        "",
        para(
            "Bütün uygun satırlarda sayıldı; doldurma yapılmadı. En az bir yeni özelliği eksik ",
            "olan satır deneyden çıkarılır ve bütün modeller (4a referansı dahil) kalan ortak ",
            f"satırlarda çalışır. Toplam çıkarılan: {excluded} satır.",
        ),
        "",
        table(["Dönem", "Bölme", "Satır", "En az bir yeni özelliği eksik"], rows, left=2),
        "",
    ]
    if detail:
        lines += [
            table(["Dönem", "Bölme", "Özellik", "Eksik satır", "Tarih aralığı"], detail, left=3),
            "",
        ]
    else:
        lines += ["Hiçbir yeni özellikte eksik değer yok; satır kümesi 4a ile aynı.", ""]
    return lines


TARGET_SETUP = {
    "raw": "`{col}`: hissenin t+1 açılış → t+H kapanış getirisi.",
    "vs_qqq": "`{col}`: hisse getirisi − aynı pencerede QQQ getirisi.",
    "xs": "`{col}`: hisse getirisi − aynı gün aynı bölmedeki hisselerin ortalama getirisi.",
    "resid": para(
        "`{col}`: hisse getirisi − beta_t × aynı pencerede ve aynı kuralla QQQ getirisi; ",
        "ardından aynı gün aynı bölmedeki hisselerin ortalamasından arındırılmış.",
    ),
}
FEATURE_SETUP = {
    "raw": "manifestteki 24 `feature_columns`, ham değerleriyle (market_* dahil).",
    "raw_market": para(
        "manifestteki 24 `feature_columns` + 6 piyasa durumu özelliği ",
        "(`market_qqq_return_5`, `market_qqq_return_60`, `market_qqq_sma50_dist`, ",
        "`market_qqq_sma200_dist`, `market_vix_close`, `market_vix_change_5`), ham. Yeni ",
        "özellikler aynı gün bütün hisselerde aynıdır ve sıraya çevrilmedi; QQQ seans takvimi, ",
        "tam pencere, yalnız t kapanışına kadarki veri, doldurma yok.",
    ),
    "xs_rank": para(
        "manifestteki özelliklerden aynı gün bütün hisselerde aynı olanlar (market_*) ",
        "çıkarıldı; kalan {n} özellik her gün hisseler arasında sıraya çevrildi (−0,5…+0,5).",
    ),
}


def _model_setup(config):
    if config["main_model"] == "lightgbm":
        lgbm = config["lightgbm"]
        params = ", ".join(f"{k} {v}" for k, v in lgbm["params"].items())
        return [
            para(
                f"- LightGBM: sabit ayarlar ({params}); validation'a bakılarak değiştirilmez. ",
                "Hedef eğitim dönemi %1/%99 kantillerine kırpılır, özellikler ham kalır. Ağaç ",
                f"sayısı en fazla {lgbm['max_trees']}; eğitimin son 252 seansında (etiketi bu ",
                "pencereden önce biten satırlarla eğitilerek) early stopping ",
                f"({lgbm['early_stopping_rounds']} tur, kırpılmamış hedefin MSE'si) ile bulunur, ",
                "sonra bütün eğitime aynı sayıyla yeniden fit edilir. ",
                f"seed {lgbm['seed']}, deterministic, sütun bazlı histogram.",
            )
        ]
    return [
        para(
            "- Ridge: girdiler ve eğitim hedefi eğitim dönemi %1/%99 kantillerine kırpılır, ",
            "girdiler standartlaştırılır. α eğitimin son 252 seansında (etiketi bu pencereden ",
            "önce biten satırlarla eğitilerek) MSE ile seçilir, sonra bütün eğitime yeniden fit ",
            "edilir.",
        )
    ]


def _setup_section(config):
    h = config["horizon"]
    features = FEATURE_SETUP[config["feature_set"]].format(n=len(config["features"]))
    lines = [
        "## Kurulum",
        "",
        para(
            f"- Satırlar: veri setindeki `partition_membership.parquet` (H = {h}) ve ",
            "`volume_features_ready = True`. Eğitim etiketleri eğitim sonunu aşmaz (purge).",
        ),
        *[f"- Hedef: {TARGET_SETUP[t].format(col=c)}" for t, c in config["targets"].items()],
        f"- Model girdileri: {features}",
        *_model_setup(config),
        para(
            "- Skor kıyasları eğitim gerektirmez: momentum = `return_20`, geri dönüş = ",
            "−`return_5`, en oynak = `volatility_20`, yüksek beta = `beta_252`.",
        ),
    ]
    if config.get("universe") == "beta":
        excluded = config["excluded_rows"]
        n_train = sum(r["excluded"] for r in excluded if r["partition"] == "train")
        n_val = sum(r["excluded"] for r in excluded if r["partition"] == "validation")
        lines.append(
            para(
                "- Evren: yalnız beta_t'si olan satırlar; dışarıda kalan ",
                f"{n_train} eğitim / {n_val} validation satırı (dönem toplamı). Gün içi sıralar ",
                "ve hedef bu evrende hesaplandı.",
            )
        )
    if config.get("universe") == "market":
        excluded = sum(r["missing_any"] for r in config["market_coverage"]["summary"])
        lines.append(
            para(
                "- Evren: yalnız 6 yeni özelliği eksiksiz olan satırlar; dışarıda kalan ",
                f"{excluded} satır (bölüm 8).",
            )
        )
    if config.get("lgbm_reference_model"):
        lines.append(
            para(
                "- LightGBM 4a referansı: 24 özellik, aynı ayarlar, aynı satırlarda yeniden ",
                "eğitildi.",
            )
        )
    if config.get("beta_neutral_row"):
        lines.append(
            para(
                "- Ek satır: Ridge skoru her gün beta_t'ye kesitsel OLS (sabitli) ile çekilip ",
                "kalıntısı alındı. Eğitim değişmez; yalnız değerlendirme.",
            )
        )
    if config.get("reference_model"):
        lines.append(
            "- Referans: Deney 1 Ridge'i (ham 24 girdi, hissenin kendi getirisine eğitilmiş)."
        )
    prefix = "lgbm" if config["main_model"] == "lightgbm" else "ridge"
    model_files = (
        "`lgbm_trees.csv`, `lgbm_importance.csv`, `lgbm_partial_dependence.csv`, "
        "`lgbm_tree_curve.csv`"
        if prefix == "lgbm"
        else "`ridge_coefficients.csv`, `ridge_alpha.csv`"
    )
    if config.get("market_coverage"):
        model_files += ", `market_coverage.csv`, `market_coverage_detail.csv`"
    lines += [
        "- Değerlendirme her zaman kırpılmamış hedefle yapılır.",
        "",
        "## Dosyalar",
        "",
        para(
            "`fold_metrics.csv`, `pooled_metrics.csv`, `consistency.csv`, `period_context.csv`, ",
            f"{model_files}, `{prefix}_exposure.csv`, `feature_ic.csv`, `daily_ic.parquet`, ",
            "`predictions.parquet`, `config.json`. Her çalıştırma `output/experiments/runs.csv` ",
            "kaydına eklenir.",
        ),
        "",
    ]
    return lines


def write_report(output, config, result, summary, consistency, context):
    folds = result["folds"]
    commit = (config["git_commit"] or "?")[:7]
    dirty = " (kaydedilmemiş değişiklik var)" if config["git_dirty"] else ""
    model_section = (
        _lgbm_section(config, result)
        if result["trees"] is not None
        else _coefficient_section(config, result["coefficients"])
    )
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
        *_consistency_section(config, consistency),
        *_context_section(config, context, result),
        *_why_section(config, result),
        *model_section,
        *_coverage_section(config),
        *_setup_section(config),
    ]
    (output / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
