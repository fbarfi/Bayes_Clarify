"""bayes_clarify with Stan (CmdStanPy).  Requires cmdstanpy and CmdStan.

    python examples/stan_cmdstanpy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

import bayes_clarify as bc

STAN = """
data { int<lower=0> N; int<lower=0> K; matrix[N, K] X; array[N] int<lower=0, upper=1> y; }
parameters { real a; vector[K] b; }
model { a ~ normal(0, 2.5); b ~ normal(0, 2.5); y ~ bernoulli_logit(a + X * b); }
"""

rng = np.random.default_rng(1)
N = 1000
df = pd.DataFrame({"x1": rng.normal(size=N), "x2": rng.integers(0, 2, N)})
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.5 * df.x1 + 0.7 * df.x2))))
cols = ["x1", "x2"]

Path("logit.stan").write_text(STAN)
fit = CmdStanModel(stan_file="logit.stan").sample(
    data={"N": N, "K": 2, "X": df[cols].to_numpy(), "y": df["y"].to_numpy()}, seed=1)

draws = bc.draws_from_cmdstanpy(fit, ["a", "b"])            # {"a": (S,), "b": (S, 2)}
post = bc.Posterior.from_linear(draws, df, coef="b", intercept="a", columns=cols,
                                link="logit")
print(bc.sim_ame(post, "x1").summary())
print(bc.sim_ame(post, "x2", contrast="diff").summary())
