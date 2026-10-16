"""Hazır veri setinden K1–K4 eğitim/validation tablolarını kurar.

Hedefler:
- ``target_return_H``: hissenin t+1 açılışından t+H kapanışına getirisi (veri setinden).
- ``target_vs_qqq_H``: aynı pencerede hisse getirisi − QQQ getirisi.
- ``target_xs_H``: hisse getirisi − aynı gün aynı bölmedeki hisselerin ortalaması.
- ``target_resid_H``: hisse getirisi − beta_t × aynı penceredeki QQQ getirisi, ardından
  aynı gün aynı bölmedeki (betası olan) hisselerin ortalamasından arındırılmış.

Tanı kolonları (model girdisi değildir):
- ``beta_252``: QQQ'ya karşı günlük getirilerle son 252 seanslık beta (en az 200 gözlem);
  yalnız t'ye kadar veri.
- ``beta_adj_return``: hisse getirisi − beta × aynı penceredeki QQQ getirisi.
- ``regime``: etiket penceresinde gerçekleşen QQQ getirisi ±%2 eşiğine göre.
  Gerçekleşmiş getiriye dayandığı için yalnız değerlendirmeyi bölmek içindir.

Piyasa durumu özellikleri (``MARKET_FEATURES``; aynı gün bütün hisselerde aynı, sıraya
çevrilmez): QQQ 5/60 seans getirisi, QQQ'nun SMA50/SMA200'e uzaklığı, ^VIX kapanışı ve
5 seanslık değişimi. QQQ seans takvimi (XNYS) kullanılır; pencereler tam olmalı, eksik
seans NaN bırakır; yalnız t kapanışına kadarki veri. Doldurma yapılmaz.

QQQ getirisi ``dataset_builder.labels.make_labels`` ile aynı kuralla hesaplanır.
Satır seçimi ``partition_membership.parquet`` üzerinden yapılır; purge kuralı oradadır.
Fiyatlar ve veri seti satırları test başlangıcından önce kesilir; 2025+ okunmaz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder.features import _grid, _return
from dataset_builder.labels import make_labels
from dataset_builder.source import discover

BENCHMARK = "QQQ"
BETA_WINDOW = 252
BETA_MIN_OBS = 200
REGIME_THRESHOLD = 0.02
REGIMES = ("yukari", "yatay", "asagi")
DAY_KEYS = ["fold", "partition", "date"]
VIX = "^VIX"
MARKET_FEATURES = [
    "market_qqq_return_5",
    "market_qqq_return_60",
    "market_qqq_sma50_dist",
    "market_qqq_sma200_dist",
    "market_vix_close",
    "market_vix_change_5",
]


@dataclass(frozen=True)
class Panel:
    rows: pd.DataFrame
    features: list[str]
    xs_features: list[str]
    targets: dict[str, str]
    folds: list[dict]
    horizon: int
    test_start: pd.Timestamp
    snapshot: str
    dataset_dir: Path
    dataset_manifest: dict
    excluded: pd.DataFrame | None = None
    market_features: list[str] = field(default_factory=lambda: list(MARKET_FEATURES))

    def split(self, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        part = self.rows.loc[self.rows["fold"].eq(fold)]
        train = part.loc[part["partition"].eq("train")].reset_index(drop=True)
        val = part.loc[part["partition"].eq("validation")].reset_index(drop=True)
        if train.empty or val.empty:
            raise ValueError(f"{fold}: boş eğitim veya validation")
        return train, val

    def subset(self, keep) -> Panel:
        """Yalnız ``keep`` satırları; gün içi sıralar ve hedefler bu evrende yeniden kurulur."""
        excluded = (
            self.rows.assign(excluded=~keep)
            .groupby(["fold", "partition"])["excluded"]
            .agg(rows="size", excluded="sum")
            .reset_index()
        )
        rows = self.rows.loc[keep].copy()
        base = [f.removeprefix("xs_") for f in self.xs_features]
        add_cross_sectional(rows, base, self.targets)
        rows = rows.reset_index(drop=True)
        return replace(self, rows=rows, excluded=excluded)

    def with_beta(self) -> Panel:
        """Yalnız betası olan satırlar."""
        return self.subset(self.rows["beta_252"].notna())

    def with_market(self) -> Panel:
        """Yalnız piyasa durumu özelliklerinin hepsi mevcut olan satırlar."""
        return self.subset(self.rows[self.market_features].notna().all(axis=1))


def benchmark_returns(source, horizon, as_of, symbol=BENCHMARK):
    """Benchmark için hisselerle aynı giriş/çıkış kuralında H seanslık getiri."""
    labels = make_labels(source.load(symbol, as_of), symbol, (horizon,))
    labels = labels.loc[labels[f"label_ready_{horizon}"]]
    return labels.set_index("date")[[f"label_end_{horizon}", f"target_return_{horizon}"]].rename(
        columns={
            f"label_end_{horizon}": "benchmark_label_end",
            f"target_return_{horizon}": "benchmark_return",
        }
    )


def daily_returns(frame, sessions):
    """Geçerli ve eligible seansların düzeltilmiş kapanış getirisi; eksik gün NaN kalır."""
    ok = frame["eligible"].eq(True) & frame["is_session"].eq(True) & frame["adj_close"].gt(0)
    close = frame["adj_close"].where(ok).reindex(sessions)
    return close.pct_change(fill_method=None)


def rolling_beta(stock, bench, window=BETA_WINDOW, min_obs=BETA_MIN_OBS):
    """t günü dahil son ``window`` seansta en az ``min_obs`` ortak gözlem varsa beta.

    Pencere yalnız geriye bakar; t sonrası fiyatlar beta_t'yi etkilemez.
    """
    both = stock.notna() & bench.notna()
    stock, bench = stock.where(both), bench.where(both)
    cov = stock.rolling(window, min_periods=min_obs).cov(bench)
    var = bench.rolling(window, min_periods=min_obs).var()
    return cov / var


def market_state_from_frames(qqq, vix):
    """QQQ ve ^VIX temiz tablolarından tarih bazlı piyasa durumu özellikleri.

    ``dataset_builder.features`` ile aynı kurallar: XNYS seans ızgarası, geçerlilik
    denetimi ve tam pencere. ^VIX QQQ ızgarasına hizalanır; XNYS dışı günleri düşer,
    eksik seansı NaN kalır.
    """
    qx, qvalid = _grid(qqq)
    close = qx["adj_close"].where(qvalid)
    vx, vvalid = _grid(vix)
    vix_close = vx["close"].where(vvalid).reindex(qx.index)
    values = {
        "market_qqq_return_5": _return(close, 5),
        "market_qqq_return_60": _return(close, 60),
        "market_qqq_sma50_dist": close / close.rolling(50, min_periods=50).mean() - 1,
        "market_qqq_sma200_dist": close / close.rolling(200, min_periods=200).mean() - 1,
        "market_vix_close": vix_close,
        "market_vix_change_5": _return(vix_close, 5),
    }
    out = pd.DataFrame(values, index=qx.index)[MARKET_FEATURES]
    out.index.name = "date"
    return out.replace([np.inf, -np.inf], np.nan)


def market_state_features(source, as_of):
    """Test başlangıcından önce kesilmiş fiyatlarla piyasa durumu tablosu."""
    return market_state_from_frames(source.load(BENCHMARK, as_of), source.load(VIX, as_of))


def market_coverage(rows, columns=MARKET_FEATURES):
    """Dönem × bölme başına eksik sayımı; özellik başına eksik satır ve tarih aralığı."""
    missing = rows[columns].isna()
    summary = (
        rows[["fold", "partition"]]
        .assign(missing_any=missing.any(axis=1))
        .groupby(["fold", "partition"])["missing_any"]
        .agg(rows="size", missing_any="sum")
        .reset_index()
    )
    detail = []
    for (fold, partition), part in rows.groupby(["fold", "partition"]):
        for column in columns:
            dates = part.loc[part[column].isna(), "date"]
            detail.append(
                {
                    "fold": fold,
                    "partition": partition,
                    "feature": column,
                    "missing": len(dates),
                    "first_missing": dates.min() if len(dates) else pd.NaT,
                    "last_missing": dates.max() if len(dates) else pd.NaT,
                }
            )
    return summary, pd.DataFrame(detail)


def residual_return(stock_return, beta, bench_return):
    """Betasından beklenenin ötesindeki getiri: hisse − beta × benchmark (aynı pencere)."""
    return stock_return - beta * bench_return


def cross_sectional_rank(rows, columns):
    """Her gün ve bölmede hisseler arası sıra, −0.5…+0.5 aralığına ölçekli."""
    grouped = rows.groupby(DAY_KEYS)
    ranks = grouped[columns].rank(method="average")
    n = grouped[columns[0]].transform("size")
    return (ranks - 1).div(n - 1, axis=0) - 0.5


def add_cross_sectional(rows, base_features, targets):
    """Gün içi sıralı özellikleri ve gün ortalamasından arındırılmış hedefleri yazar.

    Hesap ``rows`` içindeki evrene göredir; satır kümesi değişince yeniden çağrılır.
    """
    rows[[f"xs_{f}" for f in base_features]] = cross_sectional_rank(
        rows, list(base_features)
    ).to_numpy()
    raw = rows[targets["raw"]]
    rows[targets["xs"]] = raw - raw.groupby([rows[k] for k in DAY_KEYS]).transform("mean")
    resid = residual_return(raw, rows["beta_252"], rows["benchmark_return"])
    rows[targets["resid"]] = resid - resid.groupby([rows[k] for k in DAY_KEYS]).transform("mean")


def load_panel(dataset_dir, project, horizon=20) -> Panel:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if horizon not in manifest["horizons"]:
        raise ValueError(f"Veri setinde {horizon} seanslık ufuk yok")
    features = list(manifest["feature_columns"])
    test_start = pd.Timestamp(manifest["test_start"])
    as_of = test_start - pd.Timedelta(days=1)
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
    rows = rows.loc[rows["volume_features_ready"].eq(True)].copy()
    if not np.isfinite(rows[features].to_numpy()).all():
        raise ValueError("Model girdilerinde eksik veya sonsuz değer var")

    source = discover(project, manifest["snapshot"], "main")
    bench = benchmark_returns(source, horizon, as_of)
    rows = rows.join(bench, on="date")
    if rows["benchmark_return"].isna().any():
        missing = rows.loc[rows["benchmark_return"].isna(), "date"].unique()[:5]
        raise ValueError(f"{BENCHMARK} getirisi eksik tarihler: {list(missing)}")
    if not rows["benchmark_label_end"].eq(rows["label_end"]).all():
        raise ValueError(f"{BENCHMARK} ve hisse etiket pencereleri uyuşmuyor")
    targets = {
        "raw": raw_target,
        "vs_qqq": f"target_vs_qqq_{horizon}",
        "xs": f"target_xs_{horizon}",
        "resid": f"target_resid_{horizon}",
    }
    rows[targets["vs_qqq"]] = rows[raw_target] - rows["benchmark_return"]

    bench_frame = source.load(BENCHMARK, as_of)
    sessions = bench_frame.index
    bench_ret = daily_returns(bench_frame, sessions)
    betas = []
    for symbol in sorted(rows["symbol"].unique()):
        beta = rolling_beta(daily_returns(source.load(symbol, as_of), sessions), bench_ret)
        betas.append(pd.DataFrame({"date": sessions, "symbol": symbol, "beta_252": beta.values}))
    rows = rows.merge(pd.concat(betas), on=["date", "symbol"], how="left", validate="many_to_one")
    rows["beta_adj_return"] = residual_return(
        rows[raw_target], rows["beta_252"], rows["benchmark_return"]
    )
    rows = rows.join(market_state_features(source, as_of), on="date")
    rows["regime"] = np.select(
        [
            rows["benchmark_return"] > REGIME_THRESHOLD,
            rows["benchmark_return"] < -REGIME_THRESHOLD,
        ],
        ["yukari", "asagi"],
        "yatay",
    )

    # Aynı gün bütün hisselerde aynı olan özellikler sıraya çevrilince bilgi taşımaz.
    constant = rows.groupby(DAY_KEYS)[features].nunique().le(1).all()
    xs_base = [f for f in features if not constant[f]]
    add_cross_sectional(rows, xs_base, targets)
    xs_features = [f"xs_{f}" for f in xs_base]

    keep = ["date", "symbol", "fold", "partition", "label_end", *features, *xs_features]
    keep += [*targets.values(), "benchmark_return", "beta_252", "beta_adj_return", "regime"]
    keep += MARKET_FEATURES
    rows = rows[keep].sort_values(["fold", "partition", "date", "symbol"]).reset_index(drop=True)
    folds = [f for f in manifest["folds"] if f["name"] in set(rows["fold"])]
    return Panel(
        rows=rows,
        features=features,
        xs_features=xs_features,
        targets=targets,
        folds=folds,
        horizon=horizon,
        test_start=test_start,
        snapshot=manifest["snapshot"],
        dataset_dir=dataset_dir,
        dataset_manifest=manifest,
    )
