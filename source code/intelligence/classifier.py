"""
Classifiers for driving style, in the order of the scientific chain:

  1. AmeenRule           published fixed thresholds on OBD acceleration
                         (Ameen et al. 2021) - the no-training REFERENCE
  2. VolatilityClusters  unsupervised k-means on driving-volatility features
                         (Mohammadnazar, Arvin & Khattak 2021); clusters are
                         ordered by volatility and FROZEN to label live data
  3. supervised models   evaluated with session-grouped cross-validation

The evaluation helpers return plain dicts so the experiment can write them to
Markdown/JSON for the thesis.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, silhouette_score)
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import VOLATILITY_COLUMNS

CLASSES_3 = ["calm", "normal", "aggressive"]


class AmeenRule:
    """calm |a| <= 2 m/s^2, normal 2-4, aggressive > 4 (dangerous > 7 merged)."""

    feature = "obd_a_abs_max_ms2"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        a = X[self.feature].to_numpy(float)
        return np.where(a <= C.AMEEN_NORMAL_MS2, "calm",
                        np.where(a <= C.AMEEN_AGGRESSIVE_MS2, "normal", "aggressive"))


class VolatilityClusters:
    def __init__(self, k: int = 3, columns=VOLATILITY_COLUMNS, seed: int = 0):
        self.k, self.columns, self.seed = k, list(columns), seed
        self.scaler = StandardScaler()
        self.km = None
        self.names: dict[int, str] = {}

    def fit(self, X: pd.DataFrame) -> "VolatilityClusters":
        Z = self.scaler.fit_transform(X[self.columns].to_numpy(float))
        self.km = KMeans(n_clusters=self.k, n_init=20, random_state=self.seed).fit(Z)
        # Name clusters by overall volatility of their centre: lowest = calm.
        order = np.argsort(self.km.cluster_centers_.mean(axis=1))
        names = CLASSES_3 if self.k == 3 else [f"level_{i}" for i in range(self.k)]
        self.names = {int(c): names[i] for i, c in enumerate(order)}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X[self.columns].to_numpy(float))
        return np.array([self.names[int(c)] for c in self.km.predict(Z)])

    def silhouette(self, X: pd.DataFrame) -> float:
        Z = self.scaler.transform(X[self.columns].to_numpy(float))
        return float(silhouette_score(Z, self.km.predict(Z)))

    # Freezing: plain JSON, readable and version-independent.
    def to_json(self) -> str:
        return json.dumps({
            "method": "k-means on driving-volatility features (Mohammadnazar et al. 2021)",
            "columns": self.columns,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "centers": self.km.cluster_centers_.tolist(),
            "names": {str(k): v for k, v in self.names.items()},
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "FrozenClusters":
        return FrozenClusters(json.loads(text))


class FrozenClusters:
    """Nearest-centre labelling with centres fixed at training time (slide 57:
    'groups fixed once, so live driving is labelled')."""

    def __init__(self, spec: dict):
        self.columns = spec["columns"]
        self.mean = np.array(spec["scaler_mean"])
        self.scale = np.array(spec["scaler_scale"])
        self.centers = np.array(spec["centers"])
        self.names = {int(k): v for k, v in spec["names"].items()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.columns].to_numpy(float) - self.mean) / self.scale
        d = ((Z[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.array([self.names[int(i)] for i in d.argmin(axis=1)])


def metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def grouped_cv_predict(model_factory, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                       seed: int = 0) -> np.ndarray:
    """Leave-one-drive-out predictions: each drive is predicted by a model
    that never saw any second of that drive.  Deterministic splits; `seed`
    only changes the model's own randomness."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model_factory(seed).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def random_cv_predict(model_factory, X, y, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """The WRONG way (windows of one drive leak into train and test).
    Only kept to show the reader how much it inflates accuracy."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        m = model_factory(seed).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def cluster_agreement(y_true, clusters) -> dict:
    return {"adjusted_rand_index": float(adjusted_rand_score(y_true, clusters))}
