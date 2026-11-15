"""Data sources: a synthetic regime-switching generator (ground truth known) and an
optional real-data loader (yfinance / CSV) for out-of-sample sanity checks."""
import numpy as np
import pandas as pd


def simulate_regimes(
    n: int = 4000,
    p_stay: tuple[float, float] = (0.985, 0.95),
    mu: tuple[float, float] = (0.0006, -0.0008),
    sigma: tuple[float, float] = (0.007, 0.022),
    seed: int = 0,
) -> pd.DataFrame:
    """Two-regime Markov-switching returns. Regime 0 = calm/bull, regime 1 = turbulent/bear.

    Returns a DataFrame with columns: ret (daily log return), regime (true state).
    """
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        stay = p_stay[states[t - 1]]
        states[t] = states[t - 1] if rng.random() < stay else 1 - states[t - 1]
    mu_a, sigma_a = np.array(mu), np.array(sigma)
    ret = rng.normal(mu_a[states], sigma_a[states])
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame({"ret": ret, "regime": states}, index=idx)


def load_prices(ticker: str = "^NSEI", start: str = "2008-01-01") -> pd.DataFrame:
    """Real data (needs network + yfinance). No ground-truth regime exists, so label by
    a volatility rule: regime 1 when 20-day realised vol is in the top quartile of its history
    *so far* (expanding quantile, no look-ahead)."""
    import yfinance as yf

    px = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"].squeeze()
    ret = np.log(px).diff().dropna().rename("ret").to_frame()
    vol = ret["ret"].rolling(20).std()
    thresh = vol.expanding(min_periods=250).quantile(0.75)
    ret["regime"] = (vol > thresh).astype(int)
    return ret.dropna()
