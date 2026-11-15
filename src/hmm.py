"""Two-state (or K-state) Gaussian HMM, implemented from scratch in NumPy.

Includes Baum-Welch (EM) training in log space and the *forward* (filtering) algorithm,
which gives P(state_t | returns_1..t): it uses no future data, so it is safe for walk-forward use.
"""
import numpy as np


def _log_gauss(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(x_t | mu_k, var_k) -> shape (T, K)."""
    return -0.5 * (np.log(2 * np.pi * var)[None, :] + (x[:, None] - mu[None, :]) ** 2 / var[None, :])


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 60, tol: float = 1e-5, seed: int = 0):
        self.K, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed

    # ---- inference -------------------------------------------------------------------------
    def _forward(self, logB: np.ndarray):
        T, K = logB.shape
        log_alpha = np.empty((T, K))
        log_A = np.log(self.A)
        log_alpha[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + logB[t]
        return log_alpha

    def _backward(self, logB: np.ndarray):
        T, K = logB.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A)
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    # ---- training --------------------------------------------------------------------------
    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x = np.asarray(x, dtype=float)
        T, K = len(x), self.K
        rng = np.random.default_rng(self.seed)
        # Initialise by splitting on |return| quantiles so states start separated by volatility.
        order = np.argsort(np.abs(x))
        chunks = np.array_split(order, K)
        self.mu = np.array([x[c].mean() for c in chunks]) + rng.normal(0, 1e-5, K)
        self.var = np.array([x[c].var() + 1e-10 for c in chunks])
        self.A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(self.A, 0.9)
        self.pi = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            logB = _log_gauss(x, self.mu, self.var)
            log_alpha = self._forward(logB)
            log_beta = self._backward(logB)
            ll = _logsumexp(log_alpha[-1], axis=0)

            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)

            log_A = np.log(self.A)
            log_xi = (
                log_alpha[:-1, :, None]
                + log_A[None]
                + (logB[1:] + log_beta[1:])[:, None, :]
            )
            log_xi -= _logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
            xi = np.exp(log_xi)

            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma * x[:, None]).sum(0) / w
            self.var = (gamma * (x[:, None] - self.mu[None, :]) ** 2).sum(0) / w + 1e-10

            if abs(ll - prev_ll) < self.tol * abs(ll):
                break
            prev_ll = ll
        # Label states by volatility so that state K-1 is always the highest-variance regime.
        perm = np.argsort(self.var)
        self.mu, self.var, self.pi = self.mu[perm], self.var[perm], self.pi[perm]
        self.A = self.A[np.ix_(perm, perm)]
        self.log_likelihood_ = ll
        return self

    # ---- prediction ------------------------------------------------------------------------
    def filter_proba(self, x: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t) for every t (no look-ahead)."""
        logB = _log_gauss(np.asarray(x, dtype=float), self.mu, self.var)
        log_alpha = self._forward(logB)
        log_alpha -= _logsumexp(log_alpha, axis=1)[:, None]
        return np.exp(log_alpha)

    def predict_next_proba(self, x: np.ndarray) -> np.ndarray:
        """P(state_{t+1} | x_1..x_t): filtered distribution pushed one step through A."""
        return self.filter_proba(x) @ self.A
