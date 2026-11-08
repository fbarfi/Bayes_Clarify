"""Draws: posterior draws of one or more quantities of interest.

Every function of the package returns a Draws object: an (S, K) array of S
posterior draws of K named quantities, with metadata saying how they were
made (used by plot). It plays the role of the clarify_est object of the R
package clarify; the draws are posterior draws, not simulations from a normal
approximation to the sampling distribution.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


_TEX = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
        "$": r"\$", "{": r"\{", "}": r"\}", "^": r"\^{}", "~": r"\~{}",
        "|": r"$|$", "<": r"$<$", ">": r"$>$"}


def _escape(s: str) -> str:
    """Escape LaTeX special characters in a quantity name (>= and <= become
    the math symbols)."""
    s = str(s).replace(">=", "\0ge").replace("<=", "\0le")
    out = "".join(_TEX.get(ch, ch) for ch in s)
    return out.replace("\0ge", r"$\geq$").replace("\0le", r"$\leq$")


def hdi_bounds(x: np.ndarray, level: float) -> tuple:
    """Shortest interval containing a share `level` of the draws x (1-D)."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    n = x.size
    if n == 0:
        return np.nan, np.nan
    m = int(np.floor(level * n))
    if m < 1 or m >= n:
        return float(x[0]), float(x[-1])
    widths = x[m:] - x[:n - m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + m])


def eti_bounds(x: np.ndarray, level: float) -> tuple:
    """Equal-tailed interval: the (1-level)/2 and (1+level)/2 quantiles."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(x, [a, 1.0 - a])
    return float(lo), float(hi)


class _Columns:
    """Column access by name for transform(): d["E[Y(1)]"] -> (S,) array."""

    def __init__(self, values: np.ndarray, names: Sequence[str]):
        self._v = values
        self._ix = {n: i for i, n in enumerate(names)}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._ix:
            raise KeyError(f"no quantity named {name!r}; have {list(self._ix)}")
        return self._v[:, self._ix[name]]

    def keys(self) -> List[str]:
        return list(self._ix)


class Draws:
    """S posterior draws of K named quantities.

    values : (S, K) array
    names  : K names
    kind   : "apply", "setx", "ame", "levels" or "slopes" (how they were made)
    meta   : per-quantity metadata for plot (grid value, group, outcome)
    """

    def __init__(self, values: np.ndarray, names: Sequence[str],
                 kind: str = "apply", meta: Optional[List[dict]] = None,
                 info: Optional[dict] = None):
        v = np.asarray(values, float)
        if v.ndim == 1:
            v = v[:, None]
        if v.ndim != 2:
            raise ValueError("values must be (S, K).")
        if v.shape[1] != len(names):
            raise ValueError(f"{v.shape[1]} columns but {len(names)} names.")
        if len(set(names)) != len(names):
            raise ValueError("quantity names must be unique.")
        self.values = v
        self.names = list(names)
        self.kind = kind
        self.meta = list(meta) if meta is not None else [{} for _ in names]
        self.info = dict(info) if info else {}

    # ------------------------------------------------------------ access
    @property
    def n_draws(self) -> int:
        return self.values.shape[0]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, key):
        """d["name"] -> (S,) draws; d[["a", "b"]] -> Draws with those columns."""
        if isinstance(key, str):
            return self.values[:, self.names.index(key)]
        idx = [self.names.index(k) for k in key]
        return Draws(self.values[:, idx], [self.names[i] for i in idx],
                     self.kind, [self.meta[i] for i in idx], self.info)

    def to_frame(self) -> pd.DataFrame:
        """The draws as an (S, K) DataFrame."""
        return pd.DataFrame(self.values, columns=self.names)

    def __repr__(self) -> str:
        return (f"Draws(kind={self.kind!r}, {self.n_draws} draws of "
                f"{len(self)} quantities: {self.names[:6]}"
                f"{' ...' if len(self) > 6 else ''})")

    # ----------------------------------------------------------- summary
    def summary(self, level: float = 0.95, interval: str = "eti",
                point: str = "median", null: float = 0.0) -> pd.DataFrame:
        """One row per quantity.

        estimate         posterior median (point="median") or mean ("mean")
        sd               posterior standard deviation
        lower, upper     credible interval of probability `level`: equal-tailed
                         (interval="eti") or highest density ("hdi")
        Pr(>null), Pr(<null)
                         posterior probabilities that the quantity lies above /
                         below `null` (the Bayesian counterpart of the p-value
                         that clarify reports)
        """
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1).")
        bounds = {"eti": eti_bounds, "hdi": hdi_bounds}.get(interval)
        if bounds is None:
            raise ValueError("interval must be 'eti' or 'hdi'.")
        rows = []
        for j, name in enumerate(self.names):
            x = self.values[:, j]
            ok = x[np.isfinite(x)]
            est = np.median(ok) if point == "median" else np.mean(ok)
            lo, hi = bounds(ok, level)
            rows.append({"quantity": name, "estimate": float(est),
                         "sd": float(np.std(ok, ddof=1)) if ok.size > 1 else np.nan,
                         "lower": lo, "upper": hi,
                         f"Pr(>{null:g})": float(np.mean(ok > null)),
                         f"Pr(<{null:g})": float(np.mean(ok < null))})
        out = pd.DataFrame(rows).set_index("quantity")
        out.attrs.update(level=level, interval=interval, point=point)
        return out

    def to_latex(self, level: float = 0.95, interval: str = "eti",
                 point: str = "median", null: float = 0.0, digits: int = 3,
                 caption: Optional[str] = None, label: Optional[str] = None,
                 prob: bool = True, labels: Optional[Dict[str, str]] = None) -> str:
        """A booktabs table of summary(): quantity, point estimate, SD, the
        interval, and Pr(>null) if prob. labels renames quantities (LaTeX
        allowed in the new names); other names are escaped. Needs booktabs."""
        tab = self.summary(level=level, interval=interval, point=point, null=null)
        f = (lambda v: f"${v:.{digits}f}$" if np.isfinite(v) else "")
        pct = f"{100 * level:g}\\%"
        head = "Mean" if point == "mean" else "Median"
        lines = [r"\begin{table}[!h]", r"\centering", r"\footnotesize"]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines += [r"\begin{tabular}{lrrrr" + ("r" if prob else "") + "}", r"\toprule",
                  rf"Quantity & {head} & SD & \multicolumn{{2}}{{c}}{{{pct} "
                  + interval.upper() + "}" + (rf" & $\Pr(>{null:g})$" if prob else "") + r"\\",
                  r"\cmidrule(lr){4-5}",
                  r" & & & lower & upper" + (" &" if prob else "") + r"\\", r"\midrule"]
        for name, r in tab.iterrows():
            shown = (labels or {}).get(name, _escape(name))
            cells = [shown, f(r["estimate"]), f(r["sd"]), f(r["lower"]), f(r["upper"])]
            if prob:
                cells.append(f"${r[f'Pr(>{null:g})']:.2f}$")
            lines.append(" & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)

    # --------------------------------------------------------- transform
    def transform(self, **new: Callable[[_Columns], np.ndarray]) -> "Draws":
        """Add quantities computed draw by draw from existing ones, e.g.

            est.transform(RR=lambda d: d["E[Y(1)]"] / d["E[Y(0)]"])

        (clarify: transform(est, RR = `E[Y(1)]` / `E[Y(0)]`)). The function
        receives the columns by name and must return an (S,) array.
        """
        cols = _Columns(self.values, self.names)
        vals, names, meta = [self.values], list(self.names), list(self.meta)
        for name, fn in new.items():
            if name in names:
                raise ValueError(f"{name!r} already exists.")
            v = np.asarray(fn(cols), float).reshape(-1)
            if v.size != self.n_draws:
                raise ValueError(f"{name!r}: function returned {v.size} values "
                                 f"for {self.n_draws} draws.")
            vals.append(v[:, None])
            names.append(name)
            meta.append({"derived": True})
        return Draws(np.hstack(vals), names, self.kind, meta, self.info)

    @staticmethod
    def concat(*draws: "Draws") -> "Draws":
        """Join Draws made from the same posterior draws (clarify: cbind)."""
        S = {d.n_draws for d in draws}
        if len(S) != 1:
            raise ValueError("all Draws must have the same number of draws.")
        kinds = {d.kind for d in draws}
        return Draws(np.hstack([d.values for d in draws]),
                     [n for d in draws for n in d.names],
                     kinds.pop() if len(kinds) == 1 else "mixed",
                     [m for d in draws for m in d.meta])

    # -------------------------------------------------------------- plot
    def plot(self, level: float = 0.95, interval: str = "eti",
             point: str = "median", ax=None, **kw):
        """Curve across the values of a covariate with its band (kind "levels"/"slopes"), otherwise
        point estimates with credible intervals, one row per quantity."""
        from .plotting import plot_draws
        return plot_draws(self, level=level, interval=interval, point=point,
                          ax=ax, **kw)
