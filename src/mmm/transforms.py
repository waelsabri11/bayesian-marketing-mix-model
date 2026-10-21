"""Media transforms: geometric adstock and Hill saturation (NumPy and PyTensor versions)."""
import numpy as np
import pytensor.tensor as pt

L_MAX = 8  # carry-over window in weeks


def lag_matrix(x: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """(T, C) -> (T, L, C) where [:, l, :] is x shifted by l weeks (zero-padded)."""
    T, C = x.shape
    out = np.zeros((T, l_max, C))
    for lag in range(l_max):
        out[lag:, lag, :] = x[: T - lag]
    return out


def adstock_weights(alpha, l_max: int = L_MAX):
    """Normalised geometric weights alpha**l / sum(alpha**l), shape (L, C)."""
    lags = np.arange(l_max)[:, None]
    w = alpha[None, :] ** lags
    return w / w.sum(axis=0, keepdims=True)


def geometric_adstock(x: np.ndarray, alpha, l_max: int = L_MAX) -> np.ndarray:
    """NumPy geometric adstock. Weights sum to 1, so a constant spend x stays x at steady state."""
    w = adstock_weights(np.asarray(alpha, float), l_max)
    return np.einsum("tlc,lc->tc", lag_matrix(x, l_max), w)


def hill(x, k, s):
    """Hill saturation x^s / (x^s + k^s): k is the half-saturation point, s the shape."""
    if isinstance(x, np.ndarray) or np.isscalar(x):
        xs = np.maximum(x, 0.0) ** s
        return xs / (xs + k**s)
    xs = pt.maximum(x, 1e-9) ** s
    return xs / (xs + k**s)


def pt_adstock(x_lag: np.ndarray, alpha):
    """PyTensor adstock from a precomputed lag matrix (T, L, C)."""
    w = adstock_weights(alpha, x_lag.shape[1])
    return (pt.as_tensor_variable(x_lag) * w[None, :, :]).sum(axis=1)


def fourier(t: np.ndarray, period: float = 52.1775, order: int = 2) -> np.ndarray:
    """Yearly Fourier features for a week index t."""
    cols = []
    for n in range(1, order + 1):
        cols += [np.sin(2 * np.pi * n * t / period), np.cos(2 * np.pi * n * t / period)]
    return np.column_stack(cols)
