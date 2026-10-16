"""Basit kıyaslar, Ridge ve LightGBM.

Bütün modeller aynı arayüzü kullanır: ``fit(train, target)`` ve ``predict(frame)``.
Ölçekleme, kırpma, alpha ve ağaç sayısı seçimi yalnız eğitim satırlarında öğrenilir.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import daily_rank_ic

ALPHA_GRID = tuple(float(a) for a in np.logspace(0, 6, 13))
LGBM_SEED = 20261016
# Sabit ve sığ ayarlar; validation sonuçlarına bakılarak değiştirilmez.
LGBM_PARAMS = {
    "max_depth": 3,
    "num_leaves": 7,
    "learning_rate": 0.03,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
LGBM_MAX_TREES = 2000
LGBM_EARLY_STOPPING = 100
TREE_CURVE = (1, 5, 10, 25, 50, 100, 200, 400, 1000, 2000)


def purged_inner_split(train, inner_sessions=252):
    """Eğitimin son ``inner_sessions`` seansı iç validation; iç eğitim etiketleri ondan önce biter."""
    dates = np.sort(train["date"].unique())
    if len(dates) <= inner_sessions:
        raise ValueError("İç validation için eğitim dönemi çok kısa")
    inner_start = dates[-inner_sessions]
    inner_val = train["date"] >= inner_start
    inner_train = train["label_end"] < inner_start
    return train.loc[inner_train], train.loc[inner_val]


def clipped_target(frame, target, quantile):
    """Hedefi bu satırların kendi %q / %(1−q) kantillerine kırpar."""
    y = frame[target].to_numpy(dtype=float)
    lo, hi = np.quantile(y, [quantile, 1 - quantile])
    return np.clip(y, lo, hi)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Her kolonu eğitimde öğrenilen alt/üst kantillere kırpar."""

    def __init__(self, quantile=0.01):
        self.quantile = quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.quantile(X, self.quantile, axis=0)
        self.upper_ = np.quantile(X, 1 - self.quantile, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class Zero:
    """Getiri tahmini 0: ham hedefte 'değişim yok', QQQ hedefinde 'QQQ kadar'."""

    name = "sifir"
    kind = "forecast"

    def fit(self, train, target):
        return self

    def predict(self, frame):
        return np.zeros(len(frame))


class TrainMean:
    """Eğitim dönemindeki ortalama hedef; bütün satırlara aynı tahmin."""

    name = "egitim_ortalamasi"
    kind = "forecast"

    def fit(self, train, target):
        self.mean_ = float(train[target].mean())
        return self

    def predict(self, frame):
        return np.full(len(frame), self.mean_)


class FeatureScore:
    """Eğitim gerektirmeyen sıralama skoru; getiri birimi değildir.

    Kolon eksikse (ör. beta için yeterli geçmiş yoksa) skor NaN kalır.
    """

    kind = "score"

    def __init__(self, name, column, sign):
        self.name = name
        self.column = column
        self.sign = sign

    def fit(self, train, target):
        return self

    def predict(self, frame):
        return self.sign * frame[self.column].to_numpy(dtype=float)


class RidgeModel:
    """Kırpma → standartlaştırma → Ridge; alpha eğitimin içindeki son yılda seçilir.

    İç validation eğitimin son ``inner_sessions`` seansıdır. İç eğitim satırlarının
    etiketi iç validation başlamadan biter (purge). Seçilen alpha ile bütün eğitim
    dönemine yeniden fit edilir. Eğitim hedefi de eğitim kantillerine kırpılır;
    değerlendirme her zaman kırpılmamış hedefle yapılır.

    ``target_key`` verilirse model deneyin hedefi yerine o hedefe eğitilir; referans
    satırları (ör. Deney 1 Ridge'i) böyle kurulur.
    """

    def __init__(
        self,
        features,
        alphas=ALPHA_GRID,
        quantile=0.01,
        inner_sessions=252,
        *,
        name="ridge",
        kind="forecast",
        target_key=None,
    ):
        self.name = name
        self.kind = kind
        self.target_key = target_key
        self.features = list(features)
        self.alphas = tuple(alphas)
        self.quantile = quantile
        self.inner_sessions = inner_sessions

    def _fit_pipeline(self, frame, target, alpha):
        pipe = make_pipeline(Winsorizer(self.quantile), StandardScaler(), Ridge(alpha=alpha))
        X = frame[self.features].to_numpy(dtype=float)
        return pipe.fit(X, clipped_target(frame, target, self.quantile))

    def inner_split(self, train):
        return purged_inner_split(train, self.inner_sessions)

    def fit(self, train, target):
        inner_train, inner_val = self.inner_split(train)
        y_val = inner_val[target].to_numpy(dtype=float)
        X_val = inner_val[self.features].to_numpy(dtype=float)
        self.inner_mse_ = {}
        for alpha in self.alphas:
            pred = self._fit_pipeline(inner_train, target, alpha).predict(X_val)
            self.inner_mse_[alpha] = float(np.mean((y_val - pred) ** 2))
        self.alpha_ = min(self.inner_mse_, key=self.inner_mse_.get)
        self.pipeline_ = self._fit_pipeline(train, target, self.alpha_)
        return self

    def predict(self, frame):
        return self.pipeline_.predict(frame[self.features].to_numpy(dtype=float))

    def coefficients(self):
        """Standartlaştırılmış özellikte 1 std artışın tahmini getiri etkisi."""
        return pd.Series(self.pipeline_[-1].coef_, index=self.features)


class LGBMModel:
    """LightGBM regresyon; ağaç sayısı eğitimin içindeki purge'lü son yılda bulunur.

    İç validation Ridge'in alpha seçimiyle aynıdır. Early stopping (``early_stopping``
    tur) iç validation'da kırpılmamış hedefin MSE'sine bakar; bulunan ağaç sayısıyla
    bütün eğitim dönemine yeniden fit edilir. Eğitim hedefi kırpılır, özellikler ham
    kalır. Sabit seed, ``deterministic``, sütun bazlı histogram ve tek iş parçacığı aynı
    girdide aynı tahmini üretir. Tek iş parçacığı şarttır: çok iş parçacığında metrik
    toplamı son basamakta oynar ve düz bir validation eğrisinde early stopping farklı tur
    seçebilir. Farklı işletim sistemi veya LightGBM sürümünde küçük farklar olabilir.
    """

    def __init__(
        self,
        features,
        params=LGBM_PARAMS,
        max_trees=LGBM_MAX_TREES,
        early_stopping=LGBM_EARLY_STOPPING,
        quantile=0.01,
        inner_sessions=252,
        seed=LGBM_SEED,
        *,
        name="lightgbm",
        kind="forecast",
        target_key=None,
    ):
        self.name = name
        self.kind = kind
        self.target_key = target_key
        self.features = list(features)
        self.params = dict(params)
        self.max_trees = max_trees
        self.early_stopping = early_stopping
        self.quantile = quantile
        self.inner_sessions = inner_sessions
        self.seed = seed

    def _regressor(self, n_trees):
        return lgb.LGBMRegressor(
            **self.params,
            n_estimators=n_trees,
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbose=-1,
        )

    def _matrix(self, frame):
        return pd.DataFrame(frame[self.features].to_numpy(dtype=float), columns=self.features)

    def inner_split(self, train):
        return purged_inner_split(train, self.inner_sessions)

    def fit(self, train, target):
        inner_train, inner_val = self.inner_split(train)
        inner = self._regressor(self.max_trees).fit(
            self._matrix(inner_train),
            clipped_target(inner_train, target, self.quantile),
            eval_X=(self._matrix(inner_val),),
            eval_y=(inner_val[target].to_numpy(dtype=float),),
            eval_metric="l2",
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )
        self.n_trees_ = int(inner.best_iteration_ or self.max_trees)
        self.inner_mse_ = float(inner.best_score_["valid_0"]["l2"])
        self.model_ = self._regressor(self.n_trees_).fit(
            self._matrix(train), clipped_target(train, target, self.quantile)
        )
        return self

    def predict(self, frame):
        return self.model_.predict(self._matrix(frame))

    def tree_curve(self, train, target, counts=TREE_CURVE):
        """Yalnız inceleme: iç bölmede early stopping olmadan ``max_trees`` ağaç eğitir.

        Her ağaç sayısında iç validation l2'si (kırpılmamış hedef, early stopping'in
        ölçüsü) ve günlük rank IC'si döner. Ağaçlar seed sabit olduğu için early
        stopping'li eğitimin ilk ağaçlarıyla aynıdır. Modelin seçimini değiştirmez.
        """
        inner_train, inner_val = self.inner_split(train)
        full = self._regressor(self.max_trees).fit(
            self._matrix(inner_train), clipped_target(inner_train, target, self.quantile)
        )
        X = self._matrix(inner_val)
        y = inner_val[target].to_numpy(dtype=float)
        rows = []
        for n in counts:
            pred = full.predict(X, num_iteration=n)
            frame = inner_val[["date"]].assign(pred=pred, y=y)
            rows.append(
                {
                    "n_trees": n,
                    "l2": float(np.mean((y - pred) ** 2)),
                    "rank_ic": float(daily_rank_ic(frame).mean()),
                }
            )
        return pd.DataFrame(rows)

    def importance(self):
        """Toplam kazanç (gain) payı; toplam 1."""
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        total = gain.sum()
        return pd.Series(gain / total if total > 0 else gain, index=self.features)


def baseline_models():
    return [
        Zero(),
        TrainMean(),
        FeatureScore("momentum_20", "return_20", 1.0),
        FeatureScore("geri_donus_5", "return_5", -1.0),
        FeatureScore("en_oynak_20", "volatility_20", 1.0),
        FeatureScore("yuksek_beta", "beta_252", 1.0),
    ]
