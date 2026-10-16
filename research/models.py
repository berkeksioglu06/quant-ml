"""Basit kıyaslar ve Ridge.

Bütün modeller aynı arayüzü kullanır: ``fit(train, target)`` ve ``predict(frame)``.
Ölçekleme, kırpma ve alpha seçimi yalnız eğitim satırlarında öğrenilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHA_GRID = tuple(float(a) for a in np.logspace(0, 6, 13))


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
    """Eğitim gerektirmeyen sıralama skoru; getiri birimi değildir."""

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
    """

    name = "ridge"
    kind = "forecast"

    def __init__(self, features, alphas=ALPHA_GRID, quantile=0.01, inner_sessions=252):
        self.features = list(features)
        self.alphas = tuple(alphas)
        self.quantile = quantile
        self.inner_sessions = inner_sessions

    def _fit_pipeline(self, frame, target, alpha):
        y = frame[target].to_numpy(dtype=float)
        lo, hi = np.quantile(y, [self.quantile, 1 - self.quantile])
        pipe = make_pipeline(Winsorizer(self.quantile), StandardScaler(), Ridge(alpha=alpha))
        return pipe.fit(frame[self.features].to_numpy(dtype=float), np.clip(y, lo, hi))

    def inner_split(self, train):
        dates = np.sort(train["date"].unique())
        if len(dates) <= self.inner_sessions:
            raise ValueError("İç validation için eğitim dönemi çok kısa")
        inner_start = dates[-self.inner_sessions]
        inner_val = train["date"] >= inner_start
        inner_train = train["label_end"] < inner_start
        return train.loc[inner_train], train.loc[inner_val]

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


def baseline_models():
    return [
        Zero(),
        TrainMean(),
        FeatureScore("momentum_20", "return_20", 1.0),
        FeatureScore("geri_donus_5", "return_5", -1.0),
    ]
