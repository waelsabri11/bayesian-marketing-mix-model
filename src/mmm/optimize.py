"""Budget reallocation under posterior uncertainty.

Fix the total weekly budget, maximise the posterior-mean steady-state media contribution,
keep each channel within [lo, hi] x its current spend (we do not trust the curves far outside
the observed range), then report the gain distribution across posterior draws."""
import numpy as np
from scipy.optimize import minimize

from .model import response


def optimise_budget(current: np.ndarray, params: dict, spend_scale: np.ndarray, y_scale: float,
                    lo: float = 0.5, hi: float = 1.5) -> dict:
    total = current.sum()

    def neg_expected(x):
        return -response(x, params, spend_scale, y_scale).sum(axis=1).mean()

    res = minimize(
        neg_expected, current, method="SLSQP",
        bounds=[(lo * c, hi * c) for c in current],
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    opt = res.x
    base = response(current, params, spend_scale, y_scale).sum(axis=1)
    new = response(opt, params, spend_scale, y_scale).sum(axis=1)
    gain = new - base
    return {
        "current": current, "optimal": opt, "success": bool(res.success),
        "base_draws": base, "new_draws": new, "gain_draws": gain,
        "expected_gain": float(gain.mean()),
        "gain_interval": (float(np.quantile(gain, 0.05)), float(np.quantile(gain, 0.95))),
        "expected_gain_pct": float(gain.mean() / base.mean() * 100),
        "p_improvement": float((gain > 0).mean()),
    }


def scenario(spend: np.ndarray, current: np.ndarray, params: dict, spend_scale, y_scale) -> dict:
    """Compare an arbitrary weekly allocation with the current one (used by the Streamlit app)."""
    base = response(current, params, spend_scale, y_scale).sum(axis=1)
    new = response(spend, params, spend_scale, y_scale).sum(axis=1)
    gain = new - base
    return {
        "base_mean": float(base.mean()), "new_mean": float(new.mean()),
        "new_interval": (float(np.quantile(new, 0.05)), float(np.quantile(new, 0.95))),
        "gain_mean": float(gain.mean()),
        "gain_interval": (float(np.quantile(gain, 0.05)), float(np.quantile(gain, 0.95))),
        "p_improvement": float((gain > 0).mean()),
    }
