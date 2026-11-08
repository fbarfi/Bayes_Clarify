"""bayes_clarify with a PyMC model.  Requires pymc >= 5 and arviz.

    python examples/pymc_logit.py
"""

import numpy as np
import pandas as pd
import pymc as pm

import bayes_clarify as bc

rng = np.random.default_rng(42)
N = 1000
df = pd.DataFrame({"logoil": rng.normal(0, 1, N), "polity": rng.normal(0, 1, N),
                   "majorpower": rng.integers(0, 2, N)})
eta = -1.5 + 0.4 * df.logoil - 0.3 * df.polity + 0.8 * df.majorpower
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
cols = ["logoil", "polity", "majorpower"]

# inputs as pm.Data containers, expected outcome as a Deterministic
with pm.Model() as model:
    X = pm.Data("X", df[cols].to_numpy(float))
    y = pm.Data("y", df["y"].to_numpy())
    a = pm.Normal("a", 0, 2.5)
    b = pm.Normal("b", 0, 2.5, shape=len(cols))
    p = pm.Deterministic("p", pm.math.invlogit(a + X @ b))
    pm.Bernoulli("y_obs", p=p, observed=y, shape=X.shape[0])
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=1)

# route 1: predict through the PyMC model itself
post = bc.Posterior.from_pymc(
    model, idata, var="p", data=df,
    data_map=lambda d: {"X": d[cols].to_numpy(float), "y": np.zeros(len(d), dtype=int)},
    n_draws=1000)

# route 2: the same model from the coefficient draws (faster, plain numpy)
post_lin = bc.Posterior.from_linear(idata, df, coef="b", intercept="a", columns=cols,
                                    link="logit", n_draws=1000)

print(bc.sim_params(post_lin, ["a", "b"], labels={"b": cols}).summary())
print(bc.sim_setx(post_lin, {"majorpower": 0}, x1={"majorpower": 1}).summary())
print(bc.sim_ame(post, "majorpower", contrast="diff").summary(interval="hdi"))
print(bc.sim_ame(post_lin, "logoil", by="majorpower").summary())
bc.sim_predictions(post_lin, "logoil", n=25).plot().savefig("logoil_levels.png", dpi=200)
bc.sim_predictions(post_lin, "logoil", contrast="slopes").plot().savefig("logoil_slopes.png", dpi=200)
