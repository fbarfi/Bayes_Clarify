"""Posterior draws from any sampler, as {name: (S, ...)} arrays.

A Posterior needs the posterior draws as a dict of numpy arrays whose first
axis indexes the S draws (chains stacked). These functions produce that dict
from the usual sources; none of them imports the sampler, so the package works
without any of them installed.

    draws_from_idata(idata)          ArviZ InferenceData (PyMC, NumPyro, Pyro,
                                     CmdStanPy, PyStan, Bambi, emcee... all
                                     convert to it), an xarray Dataset, or a
                                     DataTree
    draws_from_cmdstanpy(fit)        a CmdStanPy CmdStanMCMC fit
    draws_from_numpyro(mcmc)         a NumPyro MCMC object, or the dict its
                                     get_samples() returns
    draws_from_table(df)             a table with one column per scalar: a
                                     DataFrame or a CSV path, e.g. Stan output,
                                     brms::as_draws_df() or posterior::as_draws_df()
                                     written from R, JAGS/BUGS matrices
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .posterior import draws_from_idata  # noqa: F401  (re-exported)

ArrayDict = Dict[str, np.ndarray]

#: columns of a draws table that are bookkeeping, not parameters
_META = re.compile(r"^(\.(chain|iteration|draw)|chain__|iter__|draw__|lp__|accept_stat__|"
                   r"stepsize__|treedepth__|n_leapfrog__|divergent__|energy__|"
                   r"chain|iteration|draw)$")
#: name[1], name[1,2], name.1, name.1.2 (numeric indices only)
_INDEXED = re.compile(r"^(?P<base>.+?)(?:\[(?P<br>\d+(?:,\d+)*)\]|\.(?P<dot>\d+(?:\.\d+)*))$")


def _pick(d: ArrayDict, var_names: Optional[Sequence[str]]) -> ArrayDict:
    if var_names is None:
        return d
    missing = [v for v in var_names if v not in d]
    if missing:
        raise KeyError(f"not in the draws: {missing}; have {sorted(d)[:20]}")
    return {v: d[v] for v in var_names}


def draws_from_cmdstanpy(fit, var_names: Optional[Sequence[str]] = None) -> ArrayDict:
    """{name: (S, ...)} from a CmdStanPy fit (chains stacked)."""
    if var_names is not None:
        return {v: np.asarray(fit.stan_variable(v)) for v in var_names}
    return {k: np.asarray(v) for k, v in fit.stan_variables().items()}


def draws_from_numpyro(mcmc_or_samples, var_names: Optional[Sequence[str]] = None) -> ArrayDict:
    """{name: (S, ...)} from a NumPyro MCMC object (chains stacked) or from the
    dict of samples it returns."""
    if hasattr(mcmc_or_samples, "get_samples"):
        samples = mcmc_or_samples.get_samples(group_by_chain=False)
    else:
        samples = mcmc_or_samples
    return _pick({k: np.asarray(v) for k, v in samples.items()}, var_names)


def draws_from_table(table: Union[pd.DataFrame, str, Path],
                     var_names: Optional[Sequence[str]] = None,
                     **read_csv_kw) -> ArrayDict:
    """{name: (S, ...)} from a table with one row per draw and one column per
    scalar parameter.

    Columns named with numeric indices, beta[1], beta[2,3] (Stan, brms) or
    beta.1, beta.2.3 (as R and CSV readers often rewrite them), are gathered
    into one array per name, in index order; the indices may start at 0 or 1.
    Other columns (b_Intercept, sigma, r_country[USA,Intercept]) stay scalars.
    Bookkeeping columns (.chain, .iteration, .draw, lp__ and the other
    sampler diagnostics) are dropped. Comment lines starting with # are
    skipped when reading a CSV (Stan output).
    """
    if isinstance(table, (str, Path)):
        read_csv_kw.setdefault("comment", "#")
        table = pd.read_csv(table, **read_csv_kw)
    df = table.loc[:, [c for c in table.columns if not _META.match(str(c))]]
    groups: Dict[str, list] = {}
    scalars: ArrayDict = {}
    for c in df.columns:
        m = _INDEXED.match(str(c))
        if m:
            idx = tuple(int(i) for i in re.split(r"[,.]", m.group("br") or m.group("dot")))
            groups.setdefault(m.group("base"), []).append((idx, c))
        else:
            scalars[str(c)] = df[c].to_numpy(float)
    out: ArrayDict = dict(scalars)
    S = len(df)
    for base, items in groups.items():
        if base in out:
            raise ValueError(f"{base!r} appears both as a scalar and with indices.")
        dims = {len(i) for i, _ in items}
        if len(dims) != 1:
            raise ValueError(f"{base!r} has indices of different lengths.")
        idx = np.array([i for i, _ in items])
        lo = idx.min(axis=0)
        shape = tuple(idx.max(axis=0) - lo + 1)
        if int(np.prod(shape)) != len(items):
            raise ValueError(f"{base!r}: {len(items)} columns do not fill a "
                             f"{shape} array.")
        arr = np.empty((S,) + shape)
        for i, c in items:
            arr[(slice(None),) + tuple(np.array(i) - lo)] = df[c].to_numpy(float)
        out[base] = arr
    return _pick(out, var_names)
