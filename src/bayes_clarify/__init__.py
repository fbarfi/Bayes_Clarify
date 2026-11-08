"""bayes_clarify: quantities of interest from Bayesian fits.

A Python counterpart of the R package clarify (Greifer, Worthington, Iacus &
King), built on posterior draws instead of simulation from a normal
approximation to the sampling distribution. It works with draws from any
sampler (PyMC, Stan, NumPyro, R/brms exports, ...) and any model whose expected
outcome you can write as a function of the parameters.

    clarify (R)              bayes_clarify
    -----------------------  -----------------------------------------------
    sim(fit, n)              Posterior(draws, predict, data)
                             Posterior.from_linear(...), Posterior.from_pymc(...)
                             draws from draws_from_idata / _cmdstanpy /
                             _numpyro / _table
    sim_apply(s, FUN)        sim_apply(post, fn), sim_params(post, names)
    sim_setx(s, x, x1)       sim_setx(post, x, x1)
    sim_ame(s, var, ...)     sim_ame(post, var, contrast=..., by=..., subset=...)
    sim_adrf(s, var, ...)    sim_predictions(post, var, contrast="levels" | "slopes")
                             (sim_adrf is an alias)
    transform(est, ...)      est.transform(name=lambda d: ...)
    cbind(est1, est2)        Draws.concat(est1, est2)
    summary(est)             est.summary(level, interval="eti" | "hdi")
    plot(est)                est.plot()
                             est.to_latex(...)
"""

from .draws import Draws, eti_bounds, hdi_bounds
from .posterior import (LINKS, Posterior, PymcPosterior, draws_from_idata,
                        typical_values)
from .quantities import (CONTRASTS, sim_adrf, sim_ame, sim_apply, sim_params,
                         sim_predictions, sim_setx)
from .sources import draws_from_cmdstanpy, draws_from_numpyro, draws_from_table

__all__ = ["Draws", "Posterior", "PymcPosterior", "typical_values",
           "draws_from_idata", "draws_from_cmdstanpy", "draws_from_numpyro",
           "draws_from_table", "sim_apply", "sim_params", "sim_setx", "sim_ame",
           "sim_predictions", "sim_adrf", "hdi_bounds", "eti_bounds", "LINKS",
           "CONTRASTS"]
__version__ = "0.1.0"
