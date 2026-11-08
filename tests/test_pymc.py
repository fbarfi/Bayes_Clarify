"""from_pymc against from_linear on a small PyMC logit (skipped without PyMC)."""

import numpy as np
import pandas as pd
import pytest

pm = pytest.importorskip("pymc")
import bayes_clarify as bc  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    N = 200
    df = pd.DataFrame({"x": rng.normal(size=N), "d": rng.integers(0, 2, N)})
    eta = -0.5 + 0.8 * df.x - 0.6 * df.d
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    cols = ["x", "d"]
    with pm.Model() as model:
        X = pm.Data("X", df[cols].to_numpy(float))
        y = pm.Data("y", df["y"].to_numpy())
        a = pm.Normal("a", 0, 2)
        b = pm.Normal("b", 0, 2, shape=2)
        p = pm.Deterministic("p", pm.math.invlogit(a + X @ b))
        pm.Bernoulli("y_obs", p=p, observed=y, shape=X.shape[0])
        idata = pm.sample(200, tune=200, chains=2, random_seed=1, progressbar=False)
    return model, idata, df, cols


def test_pymc_matches_linear(fitted):
    model, idata, df, cols = fitted
    post = bc.Posterior.from_pymc(
        model, idata, var="p", data=df,
        data_map=lambda d: {"X": d[cols].to_numpy(float), "y": np.zeros(len(d), dtype=int)})
    lin = bc.Posterior.from_linear(idata, df, coef="b", intercept="a", columns=cols,
                                   link="logit")
    for f in (lambda q: bc.sim_ame(q, "x"), lambda q: bc.sim_ame(q, "d", contrast="diff"),
              lambda q: bc.sim_setx(q, {"x": [0.0, 1.0]}),
              lambda q: bc.sim_predictions(q, "x", n=5)):
        assert np.allclose(f(post).values, f(lin).values, atol=1e-6)
    # the model data are restored afterwards
    assert model["X"].get_value().shape == (len(df), 2)
