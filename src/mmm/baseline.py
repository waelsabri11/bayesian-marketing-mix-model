"""Ridge-regression MMM baseline (Robyn-style): per-channel adstock decay and Hill
half-saturation chosen by random search with time-series cross-validation, then Ridge."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .model import Prepared
from .transforms import geometric_adstock, hill


def _features(p: Prepared, alpha, k) -> np.ndarray:
    x = p.data.spend / p.spend_scale
    media = hill(geometric_adstock(x, alpha), np.asarray(k), 1.0)
    return np.column_stack([media, p.t, p.f, p.z])


def _mape(y, yhat):
    return float(np.mean(np.abs(y - yhat) / np.abs(y)))


def fit_ridge(p: Prepared, n_iter: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    C = len(p.data.channels)
    tr = p.train_idx
    y = p.data.y / p.y_scale
    cv = TimeSeriesSplit(n_splits=4, test_size=max(8, len(tr) // 10))
    best = None
    for _ in range(n_iter):
        alpha, k = rng.uniform(0.0, 0.9, C), rng.uniform(0.1, 1.0, C)
        lam = 10 ** rng.uniform(-3, 1)
        X = _features(p, alpha, k)[tr]
        err = np.mean([_mape(y[tr][te], Ridge(lam).fit(X[a], y[tr][a]).predict(X[te]))
                       for a, te in cv.split(X)])
        if best is None or err < best["cv_mape"]:
            best = {"cv_mape": err, "alpha": alpha, "k": k, "lam": lam}
    X = _features(p, best["alpha"], best["k"])
    model = Ridge(best["lam"]).fit(X[tr], y[tr])
    coef = model.coef_[:C]
    contrib = X[:, :C] * coef * p.y_scale
    best.update(
        model=model,
        pred=model.predict(X) * p.y_scale,
        roas=contrib[tr].sum(0) / p.data.spend[tr].sum(0),
        coef=coef,
    )
    return best
