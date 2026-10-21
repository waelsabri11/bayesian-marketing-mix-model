"""Bayesian MMM in plain PyMC: geometric adstock -> Hill saturation, trend, seasonality, controls.

Scaling: target / its training mean, each channel's spend / its max, controls standardised.
So beta_c is the maximum weekly contribution of channel c as a fraction of average sales,
k_c is the half-saturation point as a fraction of the channel's peak weekly spend.
"""
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pymc as pm

from .data import MMMData
from .transforms import fourier, hill, lag_matrix, pt_adstock

PRIORS = {
    "alpha": (2.0, 2.0),        # Beta(a, b): decay, prior mean 0.5
    "k": (0.5, 0.25),           # Gamma(mu, sigma): half-saturation, in units of peak spend
    "s": (1.5, 0.6),            # Gamma(mu, sigma): Hill shape
    "beta": 0.3,                # HalfNormal sigma: max weekly effect / average sales
    "intercept": (0.6, 0.3),    # Normal
    "trend": 0.3,               # Normal(0, .)
    "season": 0.2,              # Normal(0, .)
    "control": 0.2,             # Normal(0, .)
    "sigma": 0.1,               # HalfNormal
}


@dataclass
class LiftTest:
    """A geo/holdout experiment: moving weekly spend from x0 to x1 for `weeks` weeks
    changed the outcome by `lift` (currency, total) with standard error `se`."""
    channel: str
    x0: float
    x1: float
    weeks: int
    lift: float
    se: float


@dataclass
class Prepared:
    data: MMMData
    train_idx: np.ndarray
    y_scale: float
    spend_scale: np.ndarray
    ctrl_mean: np.ndarray
    ctrl_std: np.ndarray
    x_lag: np.ndarray = field(repr=False)
    z: np.ndarray = field(repr=False)
    f: np.ndarray = field(repr=False)
    t: np.ndarray = field(repr=False)


def prepare(data: MMMData, n_train: int | None = None) -> Prepared:
    T = len(data.y)
    n_train = T if n_train is None else n_train
    train = np.arange(n_train)
    spend_scale = data.spend[train].max(axis=0)
    mu_c, sd_c = data.controls[train].mean(0), data.controls[train].std(0) + 1e-9
    return Prepared(
        data=data, train_idx=train, y_scale=float(data.y[train].mean()), spend_scale=spend_scale,
        ctrl_mean=mu_c, ctrl_std=sd_c,
        x_lag=lag_matrix(data.spend / spend_scale),
        z=(data.controls - mu_c) / sd_c,
        f=fourier(np.arange(T), order=2),
        t=np.arange(T) / n_train,
    )


def build_model(p: Prepared, lift_tests: list[LiftTest] | None = None, priors: dict = PRIORS) -> pm.Model:
    d = p.data
    coords = {"channel": d.channels, "control": d.control_names, "fourier": [f"f{i}" for i in range(p.f.shape[1])],
              "time": np.arange(len(d.y))}
    with pm.Model(coords=coords) as m:
        alpha = pm.Beta("alpha", *priors["alpha"], dims="channel")
        k = pm.Gamma("k", mu=priors["k"][0], sigma=priors["k"][1], dims="channel")
        s = pm.Gamma("s", mu=priors["s"][0], sigma=priors["s"][1], dims="channel")
        beta = pm.HalfNormal("beta", priors["beta"], dims="channel")
        intercept = pm.Normal("intercept", *priors["intercept"])
        trend = pm.Normal("trend", 0, priors["trend"])
        g_season = pm.Normal("g_season", 0, priors["season"], dims="fourier")
        g_ctrl = pm.Normal("g_ctrl", 0, priors["control"], dims="control")
        sigma = pm.HalfNormal("sigma", priors["sigma"])

        media = beta * hill(pt_adstock(p.x_lag, alpha), k, s)
        contrib = pm.Deterministic("contrib", media, dims=("time", "channel"))
        mu = pm.Deterministic(
            "mu", intercept + trend * p.t + p.f @ g_season + p.z @ g_ctrl + contrib.sum(axis=1), dims="time"
        )
        y_obs = d.y[p.train_idx] / p.y_scale
        pm.Normal("y", mu[p.train_idx], sigma, observed=y_obs)

        # Calibration: each lift test is an extra likelihood term on the channel's response curve.
        # At steady state normalised adstock leaves a constant spend unchanged, so the weekly
        # effect of moving from x0 to x1 is beta * (hill(x1) - hill(x0)).
        for i, lt in enumerate(lift_tests or []):
            c = d.channels.index(lt.channel)
            sc = p.spend_scale[c]
            pred = lt.weeks * beta[c] * (hill(lt.x1 / sc, k[c], s[c]) - hill(lt.x0 / sc, k[c], s[c]))
            pm.Normal(f"lift_{i}", pred, lt.se / p.y_scale, observed=lt.lift / p.y_scale)
    return m


def fit(model: pm.Model, draws: int = 1000, tune: int = 1500, chains: int = 4, seed: int = 42):
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, target_accept=0.99,
                          random_seed=seed, progressbar=False)
    return idata


def draws(idata, var: str) -> np.ndarray:
    """Posterior draws stacked over chains: shape (n_draws, ...)."""
    v = idata.posterior[var].values
    return v.reshape(-1, *v.shape[2:])


def diagnostics(idata) -> dict:
    names = ["alpha", "k", "s", "beta", "intercept", "trend", "g_season", "g_ctrl", "sigma"]
    summ = az.summary(idata, var_names=names, kind="diagnostics", round_to="none")
    return {
        "max_r_hat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "n_draws": int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]),
    }


def interval(x: np.ndarray, prob: float = 0.9, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    q = (1 - prob) / 2
    return np.quantile(x, q, axis=axis), np.quantile(x, 1 - q, axis=axis)


def predictive(idata, p: Prepared, seed: int = 0) -> np.ndarray:
    """Posterior predictive draws of y (currency) for every week, incl. held-out weeks."""
    rng = np.random.default_rng(seed)
    mu, sig = draws(idata, "mu"), draws(idata, "sigma")
    return (mu + rng.normal(size=mu.shape) * sig[:, None]) * p.y_scale


def roas(idata, p: Prepared, idx: np.ndarray | None = None) -> np.ndarray:
    """ROAS draws per channel: modelled revenue contribution / spend over the weeks in idx."""
    idx = p.train_idx if idx is None else idx
    c = draws(idata, "contrib")[:, idx, :].sum(axis=1) * p.y_scale
    return c / p.data.spend[idx].sum(axis=0)


def response(weekly_spend: np.ndarray, params: dict, p_scale: np.ndarray, y_scale: float) -> np.ndarray:
    """Steady-state weekly contribution (currency) per draw and channel: (D, C).
    weekly_spend: (C,) or (D, C). params: beta, k, s arrays of shape (D, C)."""
    x = np.broadcast_to(weekly_spend / p_scale, params["beta"].shape)
    return y_scale * params["beta"] * hill(x, params["k"], params["s"])


def param_draws(idata, n: int | None = None, seed: int = 0) -> dict:
    out = {v: draws(idata, v) for v in ("alpha", "k", "s", "beta")}
    if n is not None and n < len(out["beta"]):
        sel = np.random.default_rng(seed).choice(len(out["beta"]), n, replace=False)
        out = {v: a[sel] for v, a in out.items()}
    return out
