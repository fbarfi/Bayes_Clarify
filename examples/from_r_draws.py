"""bayes_clarify with draws exported from R (brms, rstanarm, rstan, JAGS ...).

In R, write the draws to a CSV file, one row per draw:

    library(brms); library(posterior)
    fit <- brm(y ~ x1 + x2, family = bernoulli(), data = d)
    write.csv(as_draws_df(fit), "draws.csv", row.names = FALSE)
    write.csv(d, "data.csv", row.names = FALSE)

Columns named with numeric indices (beta[1], beta.1, ...) are gathered into
arrays; brms names such as b_Intercept, b_x1 stay scalars.

    python examples/from_r_draws.py
"""

import numpy as np
import pandas as pd

import bayes_clarify as bc

draws = bc.draws_from_table("draws.csv")
data = pd.read_csv("data.csv")


def predict(p, df):
    eta = (p["b_Intercept"][:, None]
           + p["b_x1"][:, None] * df["x1"].to_numpy()
           + p["b_x2"][:, None] * df["x2"].to_numpy())
    return 1 / (1 + np.exp(-eta))


post = bc.Posterior(draws, predict, data)
print(bc.sim_ame(post, "x1").summary())
print(bc.sim_ame(post, "x2", contrast="diff").summary())
