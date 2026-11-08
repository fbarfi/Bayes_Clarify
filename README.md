# bayes_clarify

Quantities of interest from Bayesian fits: expected outcomes at chosen
profiles, first differences, average marginal effects, average predictions at
each level of a variable and their contrasts, and average predicted outcomes
across the values of a variable, each with its full posterior.

The design follows the R package **clarify** (Greifer, Worthington, Iacus and
King), which implements the simulation approach of King, Tomz and Wittenberg
(2000). clarify draws parameters from a normal approximation to the sampling
distribution of a frequentist estimate and pushes each draw through the model.
A Bayesian fit already has posterior draws, so bayes_clarify pushes those
through the model instead. The quantities, the "set the data, predict, average
over units" logic and the function names follow clarify.

It works with draws from **any sampler** (PyMC, Stan, NumPyro, R packages such
as brms) and **any model** whose expected outcome you can write as a function of
its parameters, including multi-category outcomes.

## Install

```
pip install git+https://github.com/fbarfi/bayes_clarify.git
pip install "bayes_clarify[pymc] @ git+https://github.com/fbarfi/bayes_clarify.git"   # with PyMC
```

Requires Python 3.9+, numpy, pandas, scipy, matplotlib.

## Quick start

```python
import numpy as np
import bayes_clarify as bc

# 1. posterior draws, {name: (S, ...)}, from any sampler
draws = bc.draws_from_idata(idata)            # or _cmdstanpy, _numpyro, _table

# 2. the expected outcome of every unit under every draw: (s, N) or (s, N, C)
def predict(p, df):
    eta = p["a"][:, None] + p["b"] @ df[["x", "d"]].to_numpy(float).T
    return 1 / (1 + np.exp(-eta))

# 3. the posterior, the data, the rule
post = bc.Posterior(draws, predict, df)

bc.sim_setx(post, {"x": [-1, 0, 1], "d": 1}).summary()     # expected outcome at profiles
bc.sim_setx(post, {"d": 0}, x1={"d": 1}).summary()          # first difference
bc.sim_ame(post, "x").summary()                             # average marginal effect
bc.sim_ame(post, "d", contrast="diff").summary()            # E[Y(1)], E[Y(0)], difference
bc.sim_predictions(post, "x", n=25).plot()                  # across the values of x
```

`examples/quickstart_any_model.py` runs as is (an ordered logit with three
categories, no sampler needed).

## Getting the draws

| source | function |
|---|---|
| ArviZ `InferenceData` (PyMC, NumPyro, Pyro, CmdStanPy, PyStan, Bambi, emcee... all convert to it) | `draws_from_idata(idata, var_names=None)` |
| CmdStanPy fit | `draws_from_cmdstanpy(fit, var_names=None)` |
| NumPyro `MCMC`, or its `get_samples()` dict | `draws_from_numpyro(mcmc, var_names=None)` |
| A table of draws, one row per draw: a DataFrame or a CSV file (Stan output, `posterior::as_draws_df()` or `brms` written from R, JAGS/BUGS matrices) | `draws_from_table(df_or_path, var_names=None)` |
| Anything else | a dict `{name: array}` whose first axis indexes the draws |

`draws_from_table` gathers columns with numeric indices (`beta[1]`,
`beta[2,3]`, `beta.1`) into arrays, keeps other names as scalars
(`b_Intercept`, `r_country[USA,Intercept]`) and drops sampler bookkeeping
(`.chain`, `.draw`, `lp__`...).

## Building a `Posterior`

A `Posterior` holds the draws, the data the quantities average over (usually
the estimation sample), and a rule

```
predict(params, data) -> (s, N)  or  (s, N, C)
```

where `params` is a dict of arrays whose first axis indexes `s` draws, `data`
a DataFrame of `N` units, and the output the **expected outcome** of each unit
under each draw. With `C` columns (category probabilities, several outcome
summaries) every quantity is reported for each column; `outcome_names=` labels
them.

Three ways to build one:

* **`Posterior(draws, predict, data)`**: any model; you write `predict` in
  numpy. The most general route and the fastest.
* **`Posterior.from_linear(draws, data, coef=, columns=, intercept=, link=)`**:
  GLM-type models, E[y] = link⁻¹(a + X b); links `identity`, `logit`,
  `probit`, `log`, `cloglog`, or a function.
* **`Posterior.from_pymc(model, idata, var=, data=, data_map=)`**: predicts
  through the PyMC model itself (inputs as `pm.Data`, the expected outcome a
  `pm.Deterministic`); see `examples/pymc_logit.py`.

Options: `n_draws=` uses a random subset of the draws; `max_elems=` bounds the
block of unit-level predictions held in memory (predictions are averaged over
units chunk by chunk, so large panels with many draws fit in memory).

## Quantities of interest

| function | quantity |
|---|---|
| `sim_setx(post, x, x1=None)` | expected outcome at profiles `x` (a list makes a grid); variables not named are held at typical values (the mean of a numeric variable with more than two values, otherwise the most frequent value). With `x1`, the first difference. The profiles used are in `.info["profiles"]`. |
| `sim_ame(post, var, contrast=None, reference=None, by=None, subset=None, weights=None)` | discrete `var`: average adjusted predictions E[Y(level)] with every unit set to each level, and contrasts against `reference` (`diff`, `ratio`, `log_ratio`, `odds_ratio`, `log_odds_ratio`). Continuous `var`: average marginal effect, the derivative averaged over units. |
| `sim_predictions(post, var, at=None, n=21, contrast="levels" \| "slopes")` | average predicted outcomes across values of `var` (every unit set to each value in turn), or the average marginal effect at each value. clarify name: `sim_adrf`. |
| `sim_apply(post, fn)` | any function of the parameters, draw by draw. |
| `sim_params(post, names)` | the parameters themselves. |

`by=` computes a quantity within the levels of a column; `subset=` restricts
the units (boolean array, function of the data, or a `DataFrame.eval` string
such as `"year >= 1990"`); `weights=` weights the units.

Every function returns a `Draws` object, the posterior draws of the
quantities:

```python
est = bc.sim_ame(post, "d", contrast="diff")
est.summary(level=0.94, interval="hdi", point="mean")   # estimate, sd, interval, Pr(>0), Pr(<0)
est.transform(RR=lambda d: d["E[Y(1)]"] / d["E[Y(0)]"])  # new quantities draw by draw
bc.Draws.concat(est, other)                               # join results
est.plot()
est.to_latex(level=0.94, interval="hdi", caption="...", label="tab:...")  # booktabs
est.to_frame()                                            # the draws as a DataFrame
```

## Differences from clarify

* The draws are posterior draws, so there is no normal approximation and no
  choice of the number of simulations (`n_draws=` thins the posterior).
* Intervals are credible intervals: equal-tailed (`interval="eti"`, the
  quantile interval clarify reports) or highest-density (`"hdi"`).
* Instead of p-values, `summary()` reports the posterior probabilities
  `Pr(>0)` and `Pr(<0)` (or relative to `null=`).

As in clarify, the units averaged over are taken as fixed: the uncertainty is
that of the parameters.

## Rules worth knowing

* A variable is treated as **discrete** in `sim_ame` if it is text,
  categorical, boolean or has at most two distinct values (or if `values=` is
  given), otherwise as continuous.
* **Derivatives** are central differences of step `eps` (default 1e-4 times the
  standard deviation of the variable).
* The **grid** of `sim_predictions` is `n` evenly spaced values from the minimum
  to the maximum of the variable in the units averaged over, unless `at=` is
  given.
* **Dummy-coded factors** are separate columns, each held at its own most
  frequent value by `sim_setx`; set them explicitly if that gives an impossible
  combination.

## Tests

```
pip install ".[test]"
pytest
```

The tests check every quantity against its value computed by hand from
synthetic posterior draws (logit and ordered-logit models, contrasts, groups,
subsets, weights, chunking, intervals), the readers of draws, and, when PyMC is
installed, `from_pymc` against `from_linear` on a fitted PyMC model.

`docs/package_map.html` shows how the parts fit together.

## Citation

See `CITATION.cff`. Please also cite clarify and King, Tomz and Wittenberg
(2000), *American Journal of Political Science* 44(2): 347–361.

## License

MIT.
