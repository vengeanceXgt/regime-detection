import numpy as np


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == y))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Return (mean predicted prob, observed frequency, count) per equal-width bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(out)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: count-weighted gap between confidence and frequency."""
    rel = reliability(y, p, bins)
    return float(np.sum(rel[:, 2] * np.abs(rel[:, 0] - rel[:, 1])) / rel[:, 2].sum())
