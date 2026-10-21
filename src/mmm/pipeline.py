"""End-to-end run: simulated recovery study + Robyn demo data. Writes everything to results/.

    uv run mmm-run            # full run (~5 model fits, a few minutes on a laptop CPU)
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from . import model as M
from .baseline import fit_ridge
from .data import ROOT, from_simulation, load_robyn
from .optimize import optimise_budget
from .simulate import BASE_LEVEL, TRUTH, simulate, true_response

RES = ROOT / "results"
FIG = RES / "figures"
SEED = 42
HOLDOUT = 26  # weeks


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def _mape(y, yhat):
    return float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)


def prior_check(p: M.Prepared, tag: str) -> dict:
    with M.build_model(p):
        prior = pm.sample_prior_predictive(500, random_seed=SEED)
    yp = prior.prior_predictive["y"].values.reshape(-1, len(p.train_idx)) * p.y_scale
    lo, hi = np.quantile(yp, [0.05, 0.95], axis=0)
    y = p.data.y[p.train_idx]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(p.data.dates[p.train_idx], lo, hi, alpha=0.3, label="prior predictive 90%")
    ax.plot(p.data.dates[p.train_idx], y, "k", lw=1, label="observed")
    ax.set_title(f"Prior predictive check ({tag})")
    ax.legend()
    _save(fig, f"{tag}_prior_predictive.png")
    return {"observed_inside_prior_90": float(((y >= lo) & (y <= hi)).mean()),
            "prior_90_width_vs_mean": float(np.mean(hi - lo) / y.mean())}


def posterior_check(idata, p: M.Prepared, tag: str) -> dict:
    yp = M.predictive(idata, p)
    lo, hi = M.interval(yp)
    y = p.data.y
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(p.data.dates, lo, hi, alpha=0.3, label="posterior predictive 90%")
    ax.plot(p.data.dates, yp.mean(0), lw=1, label="posterior mean")
    ax.plot(p.data.dates, y, "k", lw=0.8, label="observed")
    ax.set_title(f"Posterior predictive check ({tag})")
    ax.legend()
    _save(fig, f"{tag}_posterior_predictive.png")
    tr = p.train_idx
    return {"in_sample_mape_pct": _mape(y[tr], yp.mean(0)[tr]),
            "in_sample_90_coverage": float(((y[tr] >= lo[tr]) & (y[tr] <= hi[tr])).mean())}


def trace_plot(idata, tag: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    for ax, v in zip(axes.ravel(), ["alpha", "k", "s", "beta"]):
        x = idata.posterior[v].values  # (chain, draw, C)
        for c in range(x.shape[0]):
            ax.plot(x[c, :, :], lw=0.3, alpha=0.6)
        ax.set_title(f"{v} (all chains, all channels)")
    _save(fig, f"{tag}_trace.png")


def holdout(data, tag: str) -> dict:
    T = len(data.y)
    p = M.prepare(data, n_train=T - HOLDOUT)
    idata = M.fit(M.build_model(p), seed=SEED)
    test = np.arange(T - HOLDOUT, T)
    yp = M.predictive(idata, p)
    lo, hi = M.interval(yp)
    ridge = fit_ridge(p, seed=SEED)
    y = data.y
    fig, ax = plt.subplots(figsize=(9, 3.2))
    d = data.dates
    ax.fill_between(d[test], lo[test], hi[test], alpha=0.3, label="Bayesian MMM 90%")
    ax.plot(d[test], yp.mean(0)[test], label="Bayesian MMM mean")
    ax.plot(d[test], ridge["pred"][test], "--", label="Ridge MMM")
    ax.plot(d[test], y[test], "k", lw=1, label="observed")
    ax.set_title(f"{HOLDOUT}-week holdout forecast ({tag})")
    ax.legend(fontsize=8)
    _save(fig, f"{tag}_holdout.png")
    return {
        "bayes_holdout_mape_pct": _mape(y[test], yp.mean(0)[test]),
        "ridge_holdout_mape_pct": _mape(y[test], ridge["pred"][test]),
        "bayes_holdout_90_coverage": float(((y[test] >= lo[test]) & (y[test] <= hi[test])).mean()),
        "diagnostics": M.diagnostics(idata),
    }


def roas_table(idata, p, extra: dict | None = None) -> pd.DataFrame:
    r = M.roas(idata, p)
    lo, hi = M.interval(r)
    share = M.draws(idata, "contrib")[:, p.train_idx].sum(1) * p.y_scale / p.data.y[p.train_idx].sum()
    df = pd.DataFrame({
        "channel": p.data.channels,
        "spend_share": p.data.spend.sum(0) / p.data.spend.sum(),
        "contribution_share_mean": share.mean(0),
        "roas_mean": r.mean(0), "roas_p5": lo, "roas_p95": hi,
        "p_roas_gt_1": (r > 1).mean(0),
    })
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def response_plot(params, p, tag, current, truth_fn=None):
    C = len(p.data.channels)
    fig, axes = plt.subplots(1, C, figsize=(3 * C, 2.8))
    for c, ax in enumerate(axes):
        grid = np.linspace(0, 1.5 * p.spend_scale[c], 60)
        X = np.tile(current, (len(grid), 1))
        X[:, c] = grid
        r = np.stack([M.response(x, params, p.spend_scale, p.y_scale)[:, c] for x in X], axis=1)
        lo, hi = M.interval(r)
        ax.fill_between(grid, lo, hi, alpha=0.3)
        ax.plot(grid, r.mean(0), label="posterior")
        if truth_fn is not None:
            ax.plot(grid, [truth_fn(x)[c] for x in X], "k--", lw=1, label="truth")
        ax.axvline(current[c], color="grey", ls=":", lw=1)
        ax.set_title(p.data.channels[c])
        ax.set_xlabel("weekly spend")
    axes[0].set_ylabel("weekly contribution")
    if truth_fn is not None:
        axes[0].legend(fontsize=8)
    _save(fig, f"{tag}_response_curves.png")


def budget_block(idata, p, tag) -> tuple[dict, dict]:
    params = M.param_draws(idata, n=1000, seed=SEED)
    current = p.data.spend[-52:].mean(0)
    opt = optimise_budget(current, params, p.spend_scale, p.y_scale)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(opt["gain_draws"], bins=50)
    ax.axvline(0, color="k", lw=1)
    ax.set_title(f"Weekly gain from reallocation ({tag})\nP(gain > 0) = {opt['p_improvement']:.2f}")
    _save(fig, f"{tag}_budget_gain.png")
    alloc = pd.DataFrame({"channel": p.data.channels, "current_weekly": current, "optimal_weekly": opt["optimal"],
                          "change_pct": (opt["optimal"] / current - 1) * 100})
    alloc.to_csv(RES / f"{tag}_budget_allocation.csv", index=False)
    out = {k: opt[k] for k in ("expected_gain", "gain_interval", "expected_gain_pct", "p_improvement", "success")}
    out["weekly_media_contribution_current"] = float(opt["base_draws"].mean())
    out["weekly_budget"] = float(current.sum())
    return out, params


def run_simulated(meta: dict):
    df, truth = simulate(seed=0)
    df.to_csv(RES / "simulated_data.csv", index=False)
    data = from_simulation(df, TRUTH["channels"])
    p = M.prepare(data)
    out = {"prior_check": prior_check(p, "sim")}

    t0 = time.time()
    idata = M.fit(M.build_model(p), seed=SEED)
    out["fit_seconds"] = time.time() - t0
    out["diagnostics"] = M.diagnostics(idata)
    out["posterior_check"] = posterior_check(idata, p, "sim")
    trace_plot(idata, "sim")

    # Parameter recovery: true value vs posterior mean and 90% interval.
    rows = []
    for v in ("alpha", "k", "s", "beta", "roas"):
        if v == "roas":
            d, true = M.roas(idata, p), np.array(truth["roas"])
        else:
            d = M.draws(idata, v) * (p.y_scale / BASE_LEVEL if v == "beta" else 1)
            true = np.array(TRUTH[v])
        lo, hi = M.interval(d)
        for c, ch in enumerate(TRUTH["channels"]):
            rows.append({"param": v, "channel": ch, "true": true[c], "post_mean": d.mean(0)[c],
                         "p5": lo[c], "p95": hi[c], "covered": bool(lo[c] <= true[c] <= hi[c])})
    rec = pd.DataFrame(rows)
    rec.to_csv(RES / "sim_recovery.csv", index=False)
    out["recovery"] = {
        "covered_90": int(rec["covered"].sum()), "n_params": len(rec),
        "by_param": rec.groupby("param")["covered"].sum().astype(int).to_dict(),
        "roas_mean_abs_error": float((rec.query("param=='roas'")["post_mean"]
                                      - rec.query("param=='roas'")["true"]).abs().mean()),
    }
    _recovery_plot(rec)

    # Lift-test calibration on search, the channel confounded with seasonality.
    c = TRUTH["channels"].index("search")
    x0 = data.spend[:, c].mean()
    on = np.zeros(len(TRUTH["channels"]))
    on[c] = x0
    weekly_true = true_response(on, data.spend.max(0))[c]  # true weekly effect of 0 -> x0
    weeks, rng = 4, np.random.default_rng(7)
    se = 0.15 * weekly_true * weeks
    lift = M.LiftTest("search", x0=0.0, x1=x0, weeks=weeks, lift=weekly_true * weeks + rng.normal(0, se), se=se)
    idata_cal = M.fit(M.build_model(p, lift_tests=[lift]), seed=SEED)
    r0, r1 = M.roas(idata, p), M.roas(idata_cal, p)
    cal = pd.DataFrame({"channel": TRUTH["channels"], "true_roas": truth["roas"],
                        "uncalibrated_mean": r0.mean(0), "uncalibrated_p5": M.interval(r0)[0],
                        "uncalibrated_p95": M.interval(r0)[1],
                        "calibrated_mean": r1.mean(0), "calibrated_p5": M.interval(r1)[0],
                        "calibrated_p95": M.interval(r1)[1]})
    cal.to_csv(RES / "sim_calibration.csv", index=False)
    out["calibration"] = {"lift_test": lift.__dict__, "diagnostics": M.diagnostics(idata_cal),
                          "search": cal.iloc[c].to_dict()}
    _calibration_plot(r0[:, c], r1[:, c], truth["roas"][c])

    out["holdout"] = holdout(data, "sim")
    ridge_full = fit_ridge(p, seed=SEED)
    tab = roas_table(idata, p, {"true_roas": truth["roas"], "ridge_roas": ridge_full["roas"]})
    tab.to_csv(RES / "sim_roas.csv", index=False)

    # Budget optimisation, scored against the TRUE response curves.
    budget, params = budget_block(idata_cal, p, "sim")
    alloc = pd.read_csv(RES / "sim_budget_allocation.csv")
    smax = data.spend.max(0)
    true_gain = true_response(alloc["optimal_weekly"].to_numpy(), smax).sum() - true_response(
        alloc["current_weekly"].to_numpy(), smax).sum()
    budget["true_gain_of_recommendation"] = float(true_gain)
    # Oracle: the best reallocation under the TRUE curves, same budget and bounds.
    oracle_params = {v: np.array(TRUTH[v])[None] for v in ("k", "s", "beta")}
    oracle = optimise_budget(alloc["current_weekly"].to_numpy(), oracle_params, smax, BASE_LEVEL)
    budget["oracle_true_gain"] = oracle["expected_gain"]
    budget["model"] = "calibrated (with search lift test)"
    out["budget"] = budget
    response_plot(params, p, "sim", data.spend[-52:].mean(0), truth_fn=lambda x: true_response(x, smax))
    meta["simulated"] = out


def _recovery_plot(rec):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, (v, g) in zip(axes, rec.groupby("param", sort=False)):
        y = np.arange(len(g))
        ax.errorbar(g["post_mean"], y, xerr=[g["post_mean"] - g["p5"], g["p95"] - g["post_mean"]], fmt="o",
                    label="posterior 90%")
        ax.plot(g["true"], y, "rx", ms=8, label="true")
        ax.set_yticks(y, g["channel"])
        ax.set_title(v)
    axes[0].legend(fontsize=8)
    _save(fig, "sim_recovery.png")


def _calibration_plot(before, after, true):
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(0, max(np.quantile(before, 0.995), np.quantile(after, 0.995)), 60)
    ax.hist(before, bins=bins, alpha=0.5, density=True, label="MMM only")
    ax.hist(after, bins=bins, alpha=0.5, density=True, label="MMM + lift test")
    ax.axvline(true, color="k", ls="--", label="true ROAS")
    ax.set_title("Search ROAS posterior: effect of a lift-test calibration")
    ax.legend(fontsize=8)
    _save(fig, "sim_calibration.png")


def run_robyn(meta: dict):
    data = load_robyn()
    p = M.prepare(data)
    out = {"weeks": len(data.y), "channels": data.channels, "controls": data.control_names,
           "prior_check": prior_check(p, "robyn")}
    t0 = time.time()
    idata = M.fit(M.build_model(p), seed=SEED)
    out["fit_seconds"] = time.time() - t0
    out["diagnostics"] = M.diagnostics(idata)
    out["posterior_check"] = posterior_check(idata, p, "robyn")
    trace_plot(idata, "robyn")
    ridge_full = fit_ridge(p, seed=SEED)
    tab = roas_table(idata, p, {"ridge_roas": ridge_full["roas"]})
    tab.to_csv(RES / "robyn_roas.csv", index=False)
    out["roas"] = tab.to_dict(orient="records")
    out["holdout"] = holdout(data, "robyn")
    budget, params = budget_block(idata, p, "robyn")
    out["budget"] = budget
    out["allocation"] = pd.read_csv(RES / "robyn_budget_allocation.csv").to_dict(orient="records")
    response_plot(params, p, "robyn", data.spend[-52:].mean(0))
    # Thinned posterior for the Streamlit app (small file, committed).
    small = M.param_draws(idata, n=500, seed=SEED)
    np.savez(RES / "robyn_posterior_params.npz", **small, spend_scale=p.spend_scale, y_scale=p.y_scale,
             current=data.spend[-52:].mean(0), channels=np.array(data.channels))
    meta["robyn"] = out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sim", "robyn"], default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    path = RES / "metrics.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    if args.only in (None, "sim"):
        run_simulated(meta)
    if args.only in (None, "robyn"):
        run_robyn(meta)
    path.write_text(json.dumps(_jsonable(meta), indent=2))
    print(json.dumps(_jsonable(meta), indent=2)[:4000])


if __name__ == "__main__":
    main()
