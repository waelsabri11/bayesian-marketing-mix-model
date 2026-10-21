"""Budget scenario simulator on the Robyn-demo posterior.

    uv run streamlit run app/streamlit_app.py

Reads results/robyn_posterior_params.npz (500 posterior draws), so no model refit is needed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import streamlit as st

from mmm.model import response
from mmm.optimize import scenario

PATH = Path(__file__).resolve().parents[1] / "results" / "robyn_posterior_params.npz"

st.set_page_config(page_title="MMM budget simulator", layout="wide")
st.title("Budget scenario simulator")
if not PATH.exists():
    st.error("Run `uv run mmm-run` first to produce results/robyn_posterior_params.npz.")
    st.stop()

z = np.load(PATH)
channels = [str(c) for c in z["channels"]]
params = {k: z[k] for k in ("beta", "k", "s")}
scale, y_scale, current = z["spend_scale"], float(z["y_scale"]), z["current"]

st.caption("Steady-state weekly revenue contribution of media, with 90% credible intervals "
           f"from {len(params['beta'])} posterior draws. Baseline = average weekly spend over the last 52 weeks.")

cols = st.columns(len(channels))
spend = np.array([
    col.slider(ch, 0.0, float(1.5 * scale[i]), float(round(current[i])), step=float(scale[i]) / 100)
    for i, (col, ch) in enumerate(zip(cols, channels))
])

res = scenario(spend, current, params, scale, y_scale)
a, b, c = st.columns(3)
a.metric("Weekly budget", f"{spend.sum():,.0f}", f"{spend.sum() - current.sum():+,.0f} vs current")
b.metric("Expected media revenue / week", f"{res['new_mean']:,.0f}",
         f"{res['gain_mean']:+,.0f} ({res['gain_interval'][0]:+,.0f} to {res['gain_interval'][1]:+,.0f})")
c.metric("P(better than current)", f"{res['p_improvement']:.0%}")
if np.any(spend > scale) or np.any(spend < 0.5 * current):
    st.warning("Some channels are outside the historically observed range; the curves are extrapolating there.")

fig, axes = plt.subplots(1, len(channels), figsize=(3 * len(channels), 2.6))
for i, ax in enumerate(axes):
    grid = np.linspace(0, 1.5 * scale[i], 50)
    r = np.stack([response(np.where(np.arange(len(channels)) == i, g, spend), params, scale, y_scale)[:, i]
                  for g in grid], axis=1)
    ax.fill_between(grid, *np.quantile(r, [0.05, 0.95], axis=0), alpha=0.3)
    ax.plot(grid, r.mean(0))
    ax.axvline(current[i], color="grey", ls=":", lw=1)
    ax.axvline(spend[i], color="C3", lw=1)
    ax.set_title(channels[i])
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
