"""Quantities of interest from posterior draws, after the R package clarify.

    sim_apply   any function of the parameters, draw by draw
    sim_params  the draws of named parameters, as quantities
    sim_setx    expected outcome at chosen covariate profiles (others held at
                typical values), and the first difference between two profiles
    sim_ame     average adjusted predictions at each level of a variable and
                their contrasts, or the average marginal effect of a continuous
                variable
    sim_predictions
                average adjusted predictions across values of a continuous
                variable ("levels"), or the average marginal effect at each
                value ("slopes"); clarify calls it sim_adrf, an alias here

Each averages the model predictions over the units of Posterior.data draw by
draw, so the result is a set of posterior draws of the quantity (a Draws).
Uncertainty is the posterior uncertainty of the parameters; the units being
averaged over are taken as fixed, as in clarify.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .draws import Draws
from .posterior import Posterior, typical_values

Subset = Union[None, str, np.ndarray, pd.Series, Callable[[pd.DataFrame], np.ndarray]]

CONTRASTS: Dict[str, tuple] = {
    # name: (label, f(a, b)) with a the level, b the reference
    "diff": ("Diff", lambda a, b: a - b),
    "ratio": ("RR", lambda a, b: a / b),
    "log_ratio": ("logRR", lambda a, b: np.log(a) - np.log(b)),
    "odds_ratio": ("OR", lambda a, b: (a / (1 - a)) / (b / (1 - b))),
    "log_odds_ratio": ("logOR", lambda a, b: np.log(a / (1 - a)) - np.log(b / (1 - b))),
}


# ------------------------------------------------------------- helpers
def _mask(df: pd.DataFrame, subset: Subset) -> np.ndarray:
    if subset is None:
        return np.ones(len(df), bool)
    if isinstance(subset, str):
        m = df.eval(subset)
    elif callable(subset):
        m = subset(df)
    else:
        m = subset
    m = np.asarray(m, bool).reshape(-1)
    if m.size != len(df):
        raise ValueError(f"subset has {m.size} entries for {len(df)} units.")
    if not m.any():
        raise ValueError("subset selects no unit.")
    return m


def _weights(df: pd.DataFrame, weights) -> np.ndarray:
    if weights is None:
        return np.ones(len(df))
    w = df[weights].to_numpy(float) if isinstance(weights, str) else np.asarray(weights, float)
    if w.shape != (len(df),) or np.any(w < 0):
        raise ValueError("weights must be non-negative, one per unit.")
    return w


def _groups(df: pd.DataFrame, subset: Subset, by: Optional[str], w: np.ndarray):
    """(labels, W): one label per group (None without by) and the (N, G)
    matrix whose column g holds the normalised weights of group g."""
    m = _mask(df, subset)
    if by is None:
        parts = [(None, m)]
    else:
        levels = pd.unique(df.loc[m, by].dropna())
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        parts = [(f"{by}={lv}", m & (df[by] == lv).to_numpy()) for lv in levels]
    out = []
    for lab, mm in parts:
        ww = np.where(mm, w, 0.0)
        tot = ww.sum()
        if tot <= 0:
            raise ValueError(f"no positive weight in group {lab}.")
        out.append((lab, ww / tot))
    return [lab for lab, _ in out], np.column_stack([ww for _, ww in out])


def _set(df: pd.DataFrame, var: str, value) -> pd.DataFrame:
    """Copy of df with every unit at var = value (dtype kept where possible)."""
    d = df.copy()
    col = df[var]
    if isinstance(col.dtype, pd.CategoricalDtype):
        d[var] = pd.Categorical([value] * len(d), categories=col.cat.categories,
                                ordered=col.cat.ordered)
    elif pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        d[var] = np.full(len(d), value, dtype=float if isinstance(value, float)
                         or not pd.api.types.is_integer_dtype(col) else col.dtype)
    else:
        d[var] = pd.Series([value] * len(d), index=d.index).astype(col.dtype)
    return d


def _shift(df: pd.DataFrame, var: str, delta: float) -> pd.DataFrame:
    d = df.copy()
    d[var] = df[var].to_numpy(float) + delta
    return d


def _outcomes(post: Posterior, C: Optional[int]) -> List[Optional[str]]:
    if C is None:
        return [None]
    names = post.outcome_names or [str(c) for c in range(C)]
    if len(names) != C:
        raise ValueError(f"{len(names)} outcome names for {C} outcome columns.")
    return names


def _name(base: str, group: Optional[str], outcome: Optional[str]) -> str:
    s = base
    if outcome is not None:
        s += f" [{outcome}]"
    if group is not None:
        s += f" | {group}"
    return s


def _cols(a: np.ndarray) -> List[np.ndarray]:
    """(S,) -> [(S,)]; (S, C) -> C arrays."""
    return [a] if a.ndim == 1 else [a[:, c] for c in range(a.shape[1])]


def _is_discrete(s: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
            or s.dropna().nunique() <= 2)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _batches(post: Posterior, n_sets: int) -> int:
    """How many counterfactual data sets to predict per call."""
    rows = getattr(post, "max_rows", None)
    if rows is None:
        return 1
    return max(1, int(rows // max(1, len(post.data))))


# ------------------------------------------------------------ sim_apply
def sim_apply(post: Posterior, fn: Callable, names: Optional[Sequence[str]] = None,
              vectorized: bool = True) -> Draws:
    """Any function of the parameters, draw by draw (clarify: sim_apply).

    vectorized=True : fn(params) receives {name: (S, ...)} and returns (S,),
                      (S, K) or {quantity name: (S,)}
    vectorized=False: fn(params) receives one draw {name: array} and returns a
                      scalar, a K-vector or {quantity name: scalar}
    """
    P = post.params()
    if vectorized:
        out = fn(P)
    else:
        S = post.n_draws
        rows = [fn({k: v[s] for k, v in P.items()}) for s in range(S)]
        if isinstance(rows[0], dict):
            out = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        else:
            out = np.array(rows, float)
    if isinstance(out, dict):
        names = list(out)
        vals = np.column_stack([np.asarray(out[k], float).reshape(-1) for k in names])
    else:
        vals = np.asarray(out, float)
        vals = vals[:, None] if vals.ndim == 1 else vals.reshape(vals.shape[0], -1)
        if names is None:
            names = [f"q{j}" for j in range(vals.shape[1])]
    return Draws(vals, list(names), "apply")


def sim_params(post: Posterior, var_names: Sequence[str],
               labels: Optional[Dict[str, Sequence[str]]] = None) -> Draws:
    """The posterior draws of named parameters as quantities; a vector
    parameter beta becomes beta[0], beta[1], ... (or labels["beta"])."""
    P = post.params()
    vals, names = [], []
    for v in var_names:
        a = P[v].reshape(P[v].shape[0], -1)
        lab = (labels or {}).get(v)
        if lab is not None and len(lab) != a.shape[1]:
            raise ValueError(f"{len(lab)} labels for {a.shape[1]} entries of {v}.")
        for j in range(a.shape[1]):
            names.append(v if a.shape[1] == 1 else
                         (f"{v}[{lab[j]}]" if lab is not None else f"{v}[{j}]"))
            vals.append(a[:, j])
    return Draws(np.column_stack(vals), names, "apply")


# ------------------------------------------------------------- sim_setx
def sim_setx(post: Posterior, x: Optional[Dict[str, object]] = None,
             x1: Optional[Dict[str, object]] = None) -> Draws:
    """Expected outcome at covariate profiles (clarify: sim_setx).

    x   {variable: value or list of values}; a list makes a grid over all
        combinations. Variables not named are held at typical values: the mean
        of a numeric variable with more than two distinct values, the most
        frequent value otherwise (binary, categorical, text).
    x1  a second profile, given as changes from x; x must then describe one
        profile. Adds the first difference FD = E[Y|x1] - E[Y|x].

    The profiles actually used are in result.info["profiles"].
    """
    df = post.data
    x = dict(x or {})
    for k in list(x) + list((x1 or {})):
        if k not in df.columns:
            raise KeyError(f"{k!r} is not a column of the data.")
    typ = typical_values(df)
    keys = list(x)
    grid = [dict(zip(keys, combo)) for combo in itertools.product(
        *[v if isinstance(v, (list, tuple, np.ndarray, pd.Index)) else [v]
          for v in x.values()])] or [{}]
    if x1 is not None and len(grid) != 1:
        raise ValueError("with x1, x must describe a single profile.")
    profiles = [{**typ, **g} for g in grid]
    labels = []
    varying = [k for k in keys if len({repr(g[k]) for g in grid}) > 1]
    for g in grid:
        shown = varying or keys
        labels.append("E[Y|" + (", ".join(f"{k}={_fmt(g[k])}" for k in shown)
                                if shown else "typical") + "]")
    if x1 is not None:
        profiles.append({**profiles[0], **x1})
        labels = ["E[Y|x]", "E[Y|x1]"]

    # one row per profile, built from the data so the dtypes are kept
    rows = df.iloc[[0] * len(profiles)].reset_index(drop=True).copy()
    for c in df.columns:
        vals = [p[c] for p in profiles]
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            rows[c] = pd.Categorical(vals, categories=df[c].cat.categories,
                                     ordered=df[c].cat.ordered)
        else:
            try:
                rows[c] = pd.Series(vals).astype(df[c].dtype)
            except (TypeError, ValueError):
                rows[c] = pd.Series(vals)
    pred = post.predict(rows)                               # (S, P[, C])
    C = pred.shape[2] if pred.ndim == 3 else None
    outs = _outcomes(post, C)
    vals, names, meta = [], [], []
    for c, oname in enumerate(outs):
        pc = pred if C is None else pred[:, :, c]
        for p, lab in enumerate(labels):
            vals.append(pc[:, p])
            names.append(_name(lab, None, oname))
            meta.append({"profile": p, "outcome": oname})
        if x1 is not None:
            vals.append(pc[:, 1] - pc[:, 0])
            names.append(_name("FD", None, oname))
            meta.append({"profile": "FD", "outcome": oname})
    prof = pd.DataFrame(profiles, index=labels)
    return Draws(np.column_stack(vals), names, "setx", meta, {"profiles": prof})


# -------------------------------------------------------------- sim_ame
def sim_ame(post: Posterior, var: str, subset: Subset = None,
            by: Optional[str] = None, contrast: Optional[str] = None,
            reference=None, values: Optional[Sequence] = None,
            eps: Optional[float] = None, weights=None) -> Draws:
    """Average adjusted predictions or the average marginal effect of var
    (clarify: sim_ame).

    Discrete var (text, categorical, boolean, or two distinct values): every
    unit is set to each level in turn and the predictions are averaged,
    giving E[Y(level)] for each level. With contrast ("diff", "ratio",
    "log_ratio", "odds_ratio", "log_odds_ratio") each level is compared with
    reference (the first level by default). values restricts or orders the
    levels.

    Continuous var: E[dY/d(var)], the average over units of the derivative
    of the prediction, by central differences of step eps (default 1e-4 times
    the standard deviation of var).

    subset  units to average over: boolean array, function of the data, or a
            DataFrame.eval string ("year >= 1990")
    by      a column; the quantities are computed within each of its levels
    weights a column name or array of unit weights for the averages
    """
    df = post.data
    if var not in df.columns:
        raise KeyError(f"{var!r} is not a column of the data.")
    w = _weights(df, weights)
    labels, W = _groups(df, subset, by, w)
    discrete = values is not None or _is_discrete(df[var])

    if discrete:
        levels = list(values) if values is not None else list(pd.unique(df[var].dropna()))
        if values is None:
            try:
                levels = sorted(levels)
            except TypeError:
                pass
        if contrast is not None and contrast not in CONTRASTS:
            raise ValueError(f"contrast must be one of {list(CONTRASTS)}.")
        ref = levels[0] if reference is None else reference
        if contrast is not None and ref not in levels:
            raise ValueError(f"reference {ref!r} is not among the levels {levels}.")
        datas = [_set(df, var, lv) for lv in levels]
        avgs = {}                                   # (level, group) -> (S[, C])
        step = _batches(post, len(datas))
        for i in range(0, len(datas), step):
            for lv, ag in zip(levels[i:i + step],
                              post.average_many(datas[i:i + step], W)):
                for g, lab in enumerate(labels):
                    avgs[(lv, lab)] = ag[:, g]
        first = next(iter(avgs.values()))
        C = first.shape[1] if first.ndim == 2 else None
        outs = _outcomes(post, C)
        vals, names, meta = [], [], []
        for lab in labels:
            for c, oname in enumerate(outs):
                pick = (lambda a: a) if C is None else (lambda a, c=c: a[:, c])
                for lv in levels:
                    vals.append(pick(avgs[(lv, lab)]))
                    names.append(_name(f"E[Y({_fmt(lv)})]", lab, oname))
                    meta.append({"level": lv, "group": lab, "outcome": oname})
                if contrast is not None:
                    tag, f = CONTRASTS[contrast]
                    for lv in levels:
                        if lv == ref:
                            continue
                        base = tag if len(levels) == 2 else f"{tag}({_fmt(lv)} vs {_fmt(ref)})"
                        vals.append(f(pick(avgs[(lv, lab)]), pick(avgs[(ref, lab)])))
                        names.append(_name(base, lab, oname))
                        meta.append({"contrast": contrast, "level": lv,
                                     "reference": ref, "group": lab, "outcome": oname})
        return Draws(np.column_stack(vals), names, "ame", meta,
                     {"var": var, "levels": levels, "contrast": contrast})

    # continuous: average derivative by central differences
    x = df[var].to_numpy(float)
    h = eps if eps is not None else 1e-4 * (np.nanstd(x) or 1.0)
    up, dn = post.average_many([_shift(df, var, h / 2), _shift(df, var, -h / 2)], W)
    deriv = (up - dn) / h                            # (S, G[, C]): averaging is linear
    C = deriv.shape[2] if deriv.ndim == 3 else None
    outs = _outcomes(post, C)
    vals, names, meta = [], [], []
    for g, lab in enumerate(labels):
        a = deriv[:, g]
        for c, oname in enumerate(outs):
            vals.append(a if C is None else a[:, c])
            names.append(_name(f"E[dY/d({var})]", lab, oname))
            meta.append({"group": lab, "outcome": oname})
    return Draws(np.column_stack(vals), names, "ame", meta, {"var": var, "eps": h})


# ------------------------------------------------------- sim_predictions
_CURVE = {"levels": "levels", "slopes": "slopes",
          "adrf": "levels", "amef": "slopes"}         # clarify names accepted


def sim_predictions(post: Posterior, var: str, at: Optional[Sequence[float]] = None,
             n: int = 21, contrast: str = "levels", subset: Subset = None,
             by: Optional[str] = None, eps: Optional[float] = None,
             weights=None) -> Draws:
    """Average adjusted predictions across values of a continuous var, or the
    average marginal effect at each value (clarify: sim_adrf).

    contrast="levels": E[Y(v)] for each v in at, every unit set to var = v
    contrast="slopes": E[dY/d(var) | var = v], the average marginal effect
                     function, by central differences of step eps around v
    at defaults to n evenly spaced values from the minimum to the maximum of
    var in the units averaged over.
    """
    if contrast not in _CURVE:
        raise ValueError("contrast must be 'levels' or 'slopes'.")
    contrast = _CURVE[contrast]
    df = post.data
    if var not in df.columns:
        raise KeyError(f"{var!r} is not a column of the data.")
    w = _weights(df, weights)
    labels, W = _groups(df, subset, by, w)
    x = df[var].to_numpy(float)
    m = _mask(df, subset)
    if at is None:
        at = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), int(n))
    at = np.asarray(at, float)
    h = eps if eps is not None else 1e-4 * (np.nanstd(x) or 1.0)

    if contrast == "levels":
        datas = [_set(df, var, float(v)) for v in at]
    else:
        datas = [d for v in at for d in (_set(df, var, float(v) + h / 2),
                                         _set(df, var, float(v) - h / 2))]
    res: List[np.ndarray] = []                       # per grid value: avgs by group
    step = _batches(post, len(datas))
    if contrast == "slopes":
        step += step % 2                             # keep +/- pairs together
    preds_iter = []                                  # per data set: (G, S[, C])
    for i in range(0, len(datas), step):
        preds_iter.extend([np.moveaxis(ag, 1, 0)
                           for ag in post.average_many(datas[i:i + step], W)])
    if contrast == "levels":
        res = preds_iter
    else:
        res = [(preds_iter[2 * k] - preds_iter[2 * k + 1]) / h for k in range(len(at))]
    C = res[0].shape[2] if res[0].ndim == 3 else None
    outs = _outcomes(post, C)
    vals, names, meta = [], [], []
    for g, lab in enumerate(labels):
        for c, oname in enumerate(outs):
            for k, v in enumerate(at):
                a = res[k][g]
                vals.append(a if C is None else a[:, c])
                base = (f"E[Y({var}={_fmt(v)})]" if contrast == "levels"
                        else f"E[dY/d({var})|{var}={_fmt(v)}]")
                names.append(_name(base, lab, oname))
                meta.append({"grid": float(v), "group": lab, "outcome": oname})
    return Draws(np.column_stack(vals), names, contrast, meta,
                 {"var": var, "at": at, "eps": h if contrast == "slopes" else None})


#: the name clarify uses
sim_adrf = sim_predictions
