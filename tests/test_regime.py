import numpy as np

from src.data import simulate_regimes
from src.features import make_features
from src.hmm import GaussianHMM
from src.metrics import brier, ece, log_loss


def test_hmm_recovers_parameters():
    df = simulate_regimes(n=4000, seed=1)
    hmm = GaussianHMM().fit(df["ret"].values)
    # state 1 must be the turbulent one, with std near 0.022 vs 0.007
    assert abs(np.sqrt(hmm.var[0]) - 0.007) < 0.002
    assert abs(np.sqrt(hmm.var[1]) - 0.022) < 0.005
    assert np.allclose(hmm.A.sum(1), 1.0)
    assert hmm.A[0, 0] > 0.95 and hmm.A[1, 1] > 0.85


def test_filtering_is_causal():
    """Probabilities at time t must not change if future data is appended."""
    df = simulate_regimes(n=1500, seed=2)
    hmm = GaussianHMM().fit(df["ret"].values[:1000])
    x = df["ret"].values
    p_short = hmm.filter_proba(x[:1200])
    p_long = hmm.filter_proba(x[:1500])
    assert np.allclose(p_short, p_long[:1200], atol=1e-9)


def test_filter_rows_are_distributions():
    df = simulate_regimes(n=800, seed=3)
    hmm = GaussianHMM().fit(df["ret"].values)
    p = hmm.predict_next_proba(df["ret"].values)
    assert np.allclose(p.sum(1), 1.0) and (p >= 0).all()


def test_features_are_causal():
    df = simulate_regimes(n=500, seed=4)
    full = make_features(df["ret"])
    cut = make_features(df["ret"].iloc[:300])
    assert np.allclose(full.iloc[:300].dropna().values, cut.dropna().values)


def test_metrics_sanity():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier(y, perfect) == 0.0
    assert ece(y, perfect) == 0.0
    assert log_loss(y, np.array([0.5] * 4)) > log_loss(y, np.array([0.1, 0.1, 0.9, 0.9]))
