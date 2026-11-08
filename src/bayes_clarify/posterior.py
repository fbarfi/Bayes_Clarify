"""Posterior: the posterior draws, the data, and how to predict from them.

clarify::sim() draws parameters from a normal approximation to the sampling
distribution of a frequentist fit. Here the parameters are the posterior
draws themselves; a Posterior holds them with a prediction rule

    predict(params, data) -> (s, N) or (s, N, C) array

where params is a dict of arrays whose first axis indexes s posterior draws,
data is a DataFrame of N units, and the output is the expected outcome of each
unit under each draw (C columns for a model with several outcome quantities,
e.g. category probabilities). Every quantity of interest (sim_setx, sim_ame,
sim_predictions) is computed by changing the data and averaging these predictions,
draw by draw.

Three ways to build one:

    Posterior(draws, predict, data)               any model: you write predict
    Posterior.from_linear(...)                    GLM-type models: link(a + X b)
    Posterior.from_pymc(model, idata, var, ...)   a PyMC model with pm.Data
                                                  containers and a Deterministic
                                                  expected outcome
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayDict = Dict[str, np.ndarray]


# ------------------------------------------------------------------ links
def _expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _probit(x):
    from scipy.special import ndtr
    return ndtr(x)


def _cloglog(x):
    return -np.expm1(-np.exp(x))


LINKS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "logit": _expit,
    "probit": _probit,
    "log": np.exp,
    "cloglog": _cloglog,
}


# ------------------------------------------------------------ idata utils
def draws_from_idata(idata, var_names: Optional[Sequence[str]] = None,
                     group: str = "posterior") -> ArrayDict:
    """{name: (chain*draw, ...)} from an ArviZ InferenceData (or DataTree),
    an xarray Dataset of the posterior, or a dict of arrays already flattened
    to (S, ...)."""
    if isinstance(idata, dict):
        out = {k: np.asarray(v) for k, v in idata.items()
               if var_names is None or k in var_names}
        return out
    post = getattr(idata, group, None)
    if post is None:
        try:
            post = idata[group]
        except Exception:
            post = idata                          # a Dataset of the posterior
    names = list(var_names) if var_names is not None else list(post.data_vars)
    out = {}
    for v in names:
        da = post[v]
        dims = list(da.dims)
        if dims[:2] != ["chain", "draw"]:
            da = da.transpose("chain", "draw", ...)
        a = np.asarray(da.values)
        out[v] = a.reshape((-1,) + a.shape[2:])
    return out


def typical_values(data: pd.DataFrame) -> Dict[str, object]:
    """The value at which sim_setx holds a variable not set by the user: the
    mean of a numeric variable with more than two distinct values, otherwise
    (binary, categorical, boolean, text) the most frequent value."""
    out = {}
    for c in data.columns:
        s = data[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        if (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                and s.nunique() > 2):
            out[c] = float(s.mean())
        else:
            out[c] = s.mode().iloc[0]
    return out


# ------------------------------------------------------------- Posterior
class Posterior:
    """Posterior draws, the data of the fit, and a prediction rule.

    draws         {name: (S, ...)} posterior draws (see draws_from_idata)
    predict       predict(params, data) -> (s, N) or (s, N, C); params holds
                  the same keys as draws, sliced to s draws
    data          DataFrame of the N units the quantities average over
                  (usually the estimation sample)
    outcome_names names of the C outcome columns, if predict returns 3-D
    n_draws       use this many draws, picked at random (all if None)
    seed          for that pick
    max_elems     largest s*N*C array predict is asked to return at once;
                  the draws are processed in chunks below that size
    """

    def __init__(self, draws: ArrayDict, predict: Callable, data: pd.DataFrame,
                 outcome_names: Optional[Sequence[str]] = None,
                 n_draws: Optional[int] = None, seed: int = 0,
                 max_elems: float = 2e7):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        self.draws = {k: np.asarray(v) for k, v in draws.items()}
        sizes = {v.shape[0] for v in self.draws.values()}
        if len(sizes) > 1:
            raise ValueError(f"draws differ in their number of draws: {sizes}")
        S_all = sizes.pop() if sizes else 0
        self._S_all = S_all
        if n_draws is not None and S_all and n_draws < S_all:
            rng = np.random.default_rng(seed)
            self._idx = np.sort(rng.choice(S_all, int(n_draws), replace=False))
        else:
            self._idx = np.arange(S_all)
        self._predict = predict
        self.data = data.reset_index(drop=True)
        self.outcome_names = list(outcome_names) if outcome_names else None
        self.max_elems = float(max_elems)

    # ------------------------------------------------------------ sizes
    @property
    def n_draws(self) -> int:
        return int(self._idx.size)

    def params(self, sl: slice = slice(None)) -> ArrayDict:
        """The posterior draws in use (all of them, or a slice)."""
        idx = self._idx[sl]
        return {k: v[idx] for k, v in self.draws.items()}

    # ---------------------------------------------------------- predict
    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """(S, N) or (S, N, C) expected outcomes of the units in data (the
        Posterior data if None), one row per posterior draw."""
        df = self.data if data is None else data.reset_index(drop=True)
        S, N = self.n_draws, len(df)
        first = np.asarray(self._predict(self.params(slice(0, 1)), df), float)
        per_draw = max(1, int(np.prod(first.shape[1:])))
        chunk = max(1, int(self.max_elems // per_draw))
        if first.shape[:2] != (1, N):
            raise ValueError(f"predict returned shape {first.shape} for 1 draw "
                             f"and {N} units; expected (1, {N}[, C]).")
        out = np.empty((S,) + first.shape[1:], float)
        out[:1] = first
        for a in range(1, S, chunk):
            b = min(S, a + chunk)
            out[a:b] = self._predict(self.params(slice(a, b)), df)
        return out

    def predict_many(self, datas: Sequence[pd.DataFrame]) -> List[np.ndarray]:
        """predict() for several counterfactual versions of the data."""
        return [self.predict(d) for d in datas]

    def average(self, data: pd.DataFrame, W: np.ndarray) -> np.ndarray:
        """(S, G) or (S, G, C): the predictions of the units in data averaged
        with each column of W (N, G), draw by draw. Only one chunk of draws of
        unit-level predictions is held in memory at a time."""
        df = data.reset_index(drop=True)
        W = np.asarray(W, float)
        if W.shape[0] != len(df):
            raise ValueError(f"W has {W.shape[0]} rows for {len(df)} units.")
        S = self.n_draws

        def avg(pr):
            return np.moveaxis(np.tensordot(pr, W, axes=([1], [0])), -1, 1)

        first = np.asarray(self._predict(self.params(slice(0, 1)), df), float)
        if first.shape[:2] != (1, len(df)):
            raise ValueError(f"predict returned shape {first.shape} for 1 draw "
                             f"and {len(df)} units; expected (1, {len(df)}[, C]).")
        per_draw = max(1, int(np.prod(first.shape[1:])))
        chunk = max(1, int(self.max_elems // per_draw))
        a0 = avg(first)
        out = np.empty((S,) + a0.shape[1:], float)
        out[:1] = a0
        for a in range(1, S, chunk):
            b = min(S, a + chunk)
            out[a:b] = avg(np.asarray(self._predict(self.params(slice(a, b)), df), float))
        return out

    def average_many(self, datas: Sequence[pd.DataFrame], W: np.ndarray) -> List[np.ndarray]:
        """average() for several counterfactual versions of the data."""
        return [self.average(d, W) for d in datas]

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.n_draws} draws, {len(self.data)} "
                f"units, parameters {list(self.draws)[:8]})")

    # ------------------------------------------------------ constructors
    @classmethod
    def from_idata(cls, idata, predict: Callable, data: pd.DataFrame,
                   var_names: Optional[Sequence[str]] = None, **kw) -> "Posterior":
        """Posterior from an ArviZ InferenceData and your own predict."""
        return cls(draws_from_idata(idata, var_names), predict, data, **kw)

    @classmethod
    def from_linear(cls, idata, data: pd.DataFrame, coef: str,
                    columns: Sequence[str], intercept: Optional[str] = None,
                    link: Union[str, Callable] = "identity",
                    offset: Optional[str] = None, **kw) -> "Posterior":
        """E[y] = link^{-1}(intercept + sum_k coef[k] * data[columns[k]]).

        coef       name of the (S, K) coefficient draws, in the order of columns
        columns    the K columns of data (numeric; dummy-code factors yourself,
                   as the model was fitted)
        intercept  name of the (S,) intercept draws, if any
        link       "identity", "logit", "probit", "log", "cloglog" or a function
                   mapping the linear predictor to E[y]
        offset     a column of data added to the linear predictor
        """
        inv = LINKS[link] if isinstance(link, str) else link
        names = [coef] + ([intercept] if intercept else [])
        draws = draws_from_idata(idata, names)
        cols = list(columns)
        if draws[coef].reshape(draws[coef].shape[0], -1).shape[1] != len(cols):
            raise ValueError(f"{coef} has {draws[coef].shape[1:]} coefficients "
                             f"for {len(cols)} columns.")

        def predict(p, df):
            X = df[cols].to_numpy(float)
            b = p[coef].reshape(p[coef].shape[0], -1)
            eta = b @ X.T
            if intercept:
                eta = eta + p[intercept].reshape(-1, 1)
            if offset:
                eta = eta + df[offset].to_numpy(float)[None, :]
            return inv(eta)

        return cls(draws, predict, data, **kw)

    @classmethod
    def from_pymc(cls, model, idata, var: str, data: pd.DataFrame,
                  data_map: Callable[[pd.DataFrame], Dict[str, np.ndarray]],
                  coords_map: Optional[Callable[[pd.DataFrame], dict]] = None,
                  outcome_names: Optional[Sequence[str]] = None,
                  n_draws: Optional[int] = None, seed: int = 0,
                  max_rows: int = 200_000) -> "PymcPosterior":
        """Posterior of a PyMC model, predicting through the model itself.

        model       the pm.Model; its inputs must be pm.Data containers
        idata       the InferenceData returned by pm.sample (with its
                    constant_data group, which PyMC keeps by default)
        var         name of a pm.Deterministic holding the expected outcome
                    (e.g. p = pm.Deterministic("p", pm.math.invlogit(eta)))
        data        DataFrame of the estimation sample
        data_map    data_map(df) -> {container name: array} for pm.set_data;
                    must include every container whose length is N (the
                    observed outcome too, if it is a container: any values
                    of the right length will do, they are not used)
        coords_map  coords_map(df) -> {dim: labels}, if containers have dims
                    whose length changes with N
        n_draws     use about this many draws (evenly thinned per chain)
        max_rows    counterfactual data sets are stacked and predicted in one
                    call up to this many rows (one compile per call)

        Each prediction calls pm.set_data and pm.sample_posterior_predictive(
        idata, var_names=[var], predictions=True); PyMC recomputes var from
        the posterior draws because the data it depends on have changed. The
        model data are restored afterwards.
        """
        post = draws_from_idata(idata)
        obj = PymcPosterior(post, None, data, outcome_names=outcome_names)
        obj.model, obj.var = model, var
        obj.data_map, obj.coords_map = data_map, coords_map
        obj.seed, obj.max_rows = seed, int(max_rows)
        obj.idata = _thin_idata(idata, n_draws)
        # the draws held for sim_apply follow the same thinning
        obj.draws = draws_from_idata(obj.idata)
        obj._idx = np.arange(next(iter(obj.draws.values())).shape[0])
        return obj


def _thin_idata(idata, n_draws: Optional[int]):
    """Keep about n_draws draws: every k-th draw of each chain."""
    if n_draws is None:
        return idata
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    C, D = int(post.sizes["chain"]), int(post.sizes["draw"])
    if n_draws >= C * D:
        return idata
    step = max(1, int(np.ceil(C * D / n_draws)))
    try:
        return idata.isel(draw=slice(None, None, step))
    except Exception:
        logger.warning("could not thin the InferenceData; all %d draws used.",
                       C * D)
        return idata


class PymcPosterior(Posterior):
    """Posterior whose predictions come from the PyMC model (from_pymc)."""

    def _pymc_predict(self, df: pd.DataFrame) -> np.ndarray:
        import pymc as pm
        new = self.data_map(df)
        coords = self.coords_map(df) if self.coords_map else None
        with self.model:
            pm.set_data(new, coords=coords)
            pp = pm.sample_posterior_predictive(
                self.idata, var_names=[self.var], predictions=True,
                progressbar=False, random_seed=self.seed)
        pred = pp.predictions[self.var] if hasattr(pp, "predictions") \
            else pp["predictions"][self.var]
        dims = list(pred.dims)
        if dims[:2] != ["chain", "draw"]:
            pred = pred.transpose("chain", "draw", ...)
        a = np.asarray(pred.values, float)
        return a.reshape((-1,) + a.shape[2:])

    def _restore(self):
        import pymc as pm
        with self.model:
            pm.set_data(self.data_map(self.data),
                        coords=self.coords_map(self.data) if self.coords_map else None)

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.predict_many([self.data if data is None else data])[0]

    def average_many(self, datas: Sequence[pd.DataFrame], W: np.ndarray) -> List[np.ndarray]:
        W = np.asarray(W, float)
        return [np.moveaxis(np.tensordot(p, W, axes=([1], [0])), -1, 1)
                for p in self.predict_many(datas)]

    def average(self, data: pd.DataFrame, W: np.ndarray) -> np.ndarray:
        return self.average_many([data], W)[0]

    def predict_many(self, datas: Sequence[pd.DataFrame]) -> List[np.ndarray]:
        """Stack the data sets (up to max_rows rows) and predict each stack in
        one call to sample_posterior_predictive."""
        datas = [d.reset_index(drop=True) for d in datas]
        out: List[Optional[np.ndarray]] = [None] * len(datas)
        try:
            i = 0
            while i < len(datas):
                j, rows = i, 0
                while j < len(datas) and (j == i or rows + len(datas[j]) <= self.max_rows):
                    rows += len(datas[j])
                    j += 1
                stack = pd.concat(datas[i:j], ignore_index=True)
                pred = self._pymc_predict(stack)
                a = 0
                for k in range(i, j):
                    n = len(datas[k])
                    out[k] = pred[:, a:a + n]
                    a += n
                i = j
        finally:
            self._restore()
        return out
