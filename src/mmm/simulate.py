"""Simulated weekly MMM dataset with known ground truth (used to test parameter recovery)."""
import numpy as np
import pandas as pd

from .transforms import fourier, geometric_adstock, hill

# Ground truth. beta is the maximum weekly contribution in units of BASE_LEVEL.
TRUTH = {
    "channels": ["tv", "search", "social", "display"],
    "alpha": [0.70, 0.20, 0.40, 0.30],
    "k": [0.50, 0.40, 0.60, 0.35],
    "s": [2.0, 1.0, 1.5, 1.2],
    "beta": [0.22, 0.12, 0.12, 0.04],
}
BASE_LEVEL = 1_000_000.0
NOISE_SD = 0.03  # in units of BASE_LEVEL


def simulate(T: int = 208, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    season = np.sin(2 * np.pi * t / 52.1775)

    # TV: six-week flights roughly every ten weeks.
    tv = np.where((t % 10) < 6, rng.normal(60_000, 12_000, T), 0.0)
    # Search: always-on and follows demand seasonality (a realistic confounder).
    search = 20_000 * (1 + 0.3 * season) + rng.normal(0, 2_000, T)
    # Social: random-walk budget; display is bought alongside it (corr ~0.8).
    social = 30_000 + np.cumsum(rng.normal(0, 1_500, T))
    social = social - social.min() + 8_000
    display = 0.4 * social + rng.normal(0, 2_500, T) + 3_000
    spend = np.clip(np.column_stack([tv, search, social, display]), 0, None)

    promo = (rng.random(T) < 0.15).astype(float)
    price_index = 1 + 0.05 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.01, T)

    x = spend / spend.max(axis=0)
    sat = hill(geometric_adstock(x, TRUTH["alpha"]), np.array(TRUTH["k"]), np.array(TRUTH["s"]))
    contrib = sat * np.array(TRUTH["beta"])  # (T, C), units of BASE_LEVEL

    base = (
        0.60
        + 0.10 * t / T
        + fourier(t, order=2) @ np.array([0.06, 0.02, 0.01, 0.0])
        + 0.05 * promo
        - 0.80 * (price_index - 1)
    )
    y = (base + contrib.sum(axis=1) + rng.normal(0, NOISE_SD, T)) * BASE_LEVEL

    df = pd.DataFrame(spend, columns=TRUTH["channels"])
    df.insert(0, "date", pd.date_range("2021-01-04", periods=T, freq="W-MON"))
    df["promo"] = promo
    df["price_index"] = price_index
    df["sales"] = y

    contrib_cur = contrib * BASE_LEVEL
    truth = dict(TRUTH)
    truth["roas"] = (contrib_cur.sum(axis=0) / spend.sum(axis=0)).tolist()
    truth["contribution_share"] = (contrib_cur.sum(axis=0) / y.sum()).tolist()
    truth["base_level"] = BASE_LEVEL
    return df, truth


def true_response(weekly_spend: np.ndarray, spend_max: np.ndarray) -> np.ndarray:
    """True steady-state weekly contribution (currency) for a weekly spend vector."""
    k, s, beta = (np.array(TRUTH[n]) for n in ("k", "s", "beta"))
    return BASE_LEVEL * beta * hill(weekly_spend / spend_max, k, s)
