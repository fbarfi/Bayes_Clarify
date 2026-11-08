"""Quickstart: any model, any sampler.

All bayes_clarify needs is (1) posterior draws as {name: (S, ...)} arrays,
(2) the data, and (3) the expected outcome as a function of the parameters.
Here the draws are simulated so the script runs as is; with a real fit, get
them from your sampler (see the other examples).

    python examples/quickstart_any_model.py
"""

import numpy as np
import pandas as pd

import bayes_clarify as bc

rng = np.random.default_rng(0)

# ---- data: country-years with a continuous and a binary covariate
N = 2000
data = pd.DataFrame({"gdp": rng.normal(size=N),
                     "democracy": rng.integers(0, 2, N),
                     "region": rng.choice(["Africa", "Americas", "Asia", "Europe"], N)})

# ---- posterior draws of an ordered logit with 3 categories (stand-in for a fit)
S = 2000
draws = {"beta": rng.normal([0.6, -0.4], 0.08, (S, 2)),        # gdp, democracy
         "region": rng.normal([0.0, 0.2, 0.4, -0.3], 0.1, (S, 4)),
         "cut": np.sort(rng.normal([-0.5, 1.2], 0.1, (S, 2)), axis=1)}
REGIONS = ["Africa", "Americas", "Asia", "Europe"]


def expit(z):
    return 1 / (1 + np.exp(-z))


def predict(p, df):
    """Expected outcome of every unit under every draw: here the three
    category probabilities, shape (s, N, 3)."""
    r = pd.Categorical(df["region"], categories=REGIONS).codes
    eta = (p["beta"][:, :1] * df["gdp"].to_numpy()
           + p["beta"][:, 1:2] * df["democracy"].to_numpy()
           + p["region"][:, r])
    c1 = expit(p["cut"][:, :1] - eta)
    c2 = expit(p["cut"][:, 1:2] - eta)
    return np.stack([c1, c2 - c1, 1 - c2], axis=-1)


post = bc.Posterior(draws, predict, data, outcome_names=["none", "low", "high"])

# expected outcome at chosen profiles (other covariates at typical values)
print(bc.sim_setx(post, {"gdp": [-1, 0, 1], "democracy": 1}).summary().round(3))

# average marginal effect of gdp, overall and by region
print(bc.sim_ame(post, "gdp", by="region").summary().round(4))

# democracy: average predictions at 0 and 1, and their difference
ame = bc.sim_ame(post, "democracy", contrast="diff")
print(ame.summary(interval="hdi", level=0.94).round(4))

# average predicted outcomes across values of gdp; a figure
bc.sim_predictions(post, "gdp", n=25).plot().savefig("predictions_gdp.png", dpi=150)

# a LaTeX table
print(ame[["Diff [high]", "Diff [low]"]].to_latex(
    level=0.94, interval="hdi", point="mean", caption="Effect of democracy.",
    label="tab:democracy", labels={"Diff [high]": "high", "Diff [low]": "low"}))
