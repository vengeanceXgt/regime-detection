"""Walk-forward evaluation: train on the past only, predict the next block, repeat.

Target at row t is the regime on day t+1 (a genuine one-day-ahead forecast).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .features import make_features
from .hmm import GaussianHMM

warnings.filterwarnings("ignore")


class _Platt:
    """Platt scaling: logistic regression on the logit of the model's probability.
    Smoother than isotonic regression, so it behaves better on small calibration sets."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))[:, None]

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Platt":
        # A calibration window with a single regime cannot identify a mapping: fall back to identity.
        self.identity = len(np.unique(y)) < 2
        if not self.identity:
            self.lr.fit(self._logit(p), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.identity:
            return p
        return self.lr.predict_proba(self._logit(p))[:, 1]


def _fit_calibrator(p_cal: np.ndarray, y_cal: np.ndarray) -> _Platt:
    return _Platt().fit(p_cal, y_cal)


def run_walk_forward(
    df: pd.DataFrame,
    min_train: int = 1500,
    step: int = 250,
    calib_frac: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by date with y (next-day regime) and one prob column per model."""
    ret = df["ret"]
    feats = make_features(ret)
    y_next = df["regime"].shift(-1)
    valid = feats.notna().all(axis=1) & y_next.notna()
    feats, y_next, ret_v = feats[valid], y_next[valid].astype(int), ret[valid]
    n = len(feats)
    cols = ["hmm", "gbm", "gbm_calibrated", "mlp", "mlp_calibrated"]
    out = {c: np.full(n, np.nan) for c in cols}

    start = min_train
    while start < n:
        end = min(start + step, n)
        tr = slice(0, start)
        X_tr, y_tr = feats.iloc[tr].values, y_next.iloc[tr].values

        # HMM sees raw returns only; fit on the training window, filter forward through the test block.
        hmm = GaussianHMM(seed=seed).fit(ret_v.iloc[tr].values)
        p_hmm = hmm.predict_next_proba(ret_v.iloc[:end].values)[:, -1]
        out["hmm"][start:end] = p_hmm[start:end]

        # Hold out the most recent part of training data to calibrate (still strictly before the test block).
        cut = int(start * (1 - calib_frac))
        X_fit, y_fit, X_cal, y_cal = X_tr[:cut], y_tr[:cut], X_tr[cut:], y_tr[cut:]
        X_te = feats.iloc[start:end].values

        gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
        gbm.fit(X_fit, y_fit)
        out["gbm"][start:end] = gbm.predict_proba(X_te)[:, 1]
        cal_g = _fit_calibrator(gbm.predict_proba(X_cal)[:, 1], y_cal)
        out["gbm_calibrated"][start:end] = cal_g.predict(gbm.predict_proba(X_te)[:, 1])

        scaler = StandardScaler().fit(X_fit)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=400, random_state=seed)
        mlp.fit(scaler.transform(X_fit), y_fit)
        out["mlp"][start:end] = mlp.predict_proba(scaler.transform(X_te))[:, 1]
        cal_m = _fit_calibrator(mlp.predict_proba(scaler.transform(X_cal))[:, 1], y_cal)
        out["mlp_calibrated"][start:end] = cal_m.predict(mlp.predict_proba(scaler.transform(X_te))[:, 1])

        start = end

    res = pd.DataFrame(out, index=feats.index)
    res["y"] = y_next.values
    return res.dropna()
