import numpy as np
import pymc as pm
import pytensor
import pytest

from mmm import model as M
from mmm.baseline import fit_ridge
from mmm.data import from_simulation
from mmm.optimize import optimise_budget
from mmm.simulate import TRUTH, simulate
from mmm.transforms import adstock_weights, geometric_adstock, hill, lag_matrix, pt_adstock


@pytest.fixture(scope="module")
def sim():
    df, truth = simulate(T=104, seed=1)
    return df, truth, M.prepare(from_simulation(df, TRUTH["channels"]))


def test_adstock_weights_sum_to_one():
    w = adstock_weights(np.array([0.0, 0.5, 0.9]))
    assert np.allclose(w.sum(axis=0), 1.0)
    assert np.allclose(w[:, 0], [1] + [0] * (w.shape[0] - 1))  # alpha=0 means no carry-over


def test_adstock_steady_state_and_carryover():
    x = np.full((30, 1), 2.0)
    assert np.allclose(geometric_adstock(x, [0.6])[10:], 2.0)
    pulse = np.zeros((10, 1))
    pulse[0] = 1.0
    out = geometric_adstock(pulse, [0.5])[:, 0]
    assert out[0] > out[1] > out[2] > 0 and np.isclose(out[1] / out[0], 0.5)


def test_numpy_and_pytensor_adstock_agree():
    rng = np.random.default_rng(0)
    x = rng.random((20, 3))
    alpha = np.array([0.1, 0.5, 0.8])
    pt_val = pt_adstock(lag_matrix(x), pytensor.shared(alpha)).eval()
    assert np.allclose(pt_val, geometric_adstock(x, alpha))


def test_hill_properties():
    assert np.isclose(hill(0.4, 0.4, 2.0), 0.5)
    xs = np.linspace(0, 5, 50)
    h = hill(xs, 0.5, 1.5)
    assert np.all(np.diff(h) >= 0) and h[0] == 0 and h[-1] < 1


def test_simulation_is_reproducible(sim):
    df, truth, _ = sim
    df2, truth2 = simulate(T=104, seed=1)
    assert df.equals(df2) and truth == truth2
    assert all(r > 0 for r in truth["roas"])


def test_model_builds_and_prior_predictive_is_finite(sim):
    _, _, p = sim
    lift = M.LiftTest("search", x0=0.0, x1=20_000.0, weeks=4, lift=200_000.0, se=30_000.0)
    with M.build_model(p, lift_tests=[lift]) as m:
        prior = pm.sample_prior_predictive(20, random_seed=0)
    assert "lift_0" in m.named_vars
    assert np.isfinite(prior.prior_predictive["y"].values).all()


def test_ridge_baseline_runs(sim):
    _, _, p = sim
    r = fit_ridge(p, n_iter=10)
    assert r["pred"].shape == p.data.y.shape and r["cv_mape"] < 0.2


def test_budget_optimiser_respects_constraints():
    current = np.array([10.0, 10.0])
    # Channel 0 saturates early, channel 1 is nearly linear: money should move to channel 1.
    params = {"beta": np.array([[1.0, 1.0]]), "k": np.array([[0.05, 2.0]]), "s": np.array([[1.0, 1.0]])}
    out = optimise_budget(current, params, spend_scale=np.array([20.0, 20.0]), y_scale=100.0)
    assert np.isclose(out["optimal"].sum(), 20.0)
    assert np.all(out["optimal"] >= 5 - 1e-6) and np.all(out["optimal"] <= 15 + 1e-6)
    assert out["optimal"][1] > 10 and out["expected_gain"] > 0 and out["p_improvement"] == 1.0


def test_streamlit_app_renders():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    if not (app.parents[1] / "results" / "robyn_posterior_params.npz").exists():
        pytest.skip("run `uv run mmm-run` first")
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 3
