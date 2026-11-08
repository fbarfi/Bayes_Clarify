"""Plots of Draws (clarify: plot.clarify_est)."""

from __future__ import annotations

import numpy as np


def plot_draws(est, level=0.95, interval="eti", point="median", ax=None,
               color="C0", **kw):
    import matplotlib.pyplot as plt

    tab = est.summary(level=level, interval=interval, point=point)
    pct = f"{int(round(100 * level))}%"

    if est.kind in ("levels", "slopes"):
        keys = sorted({(m.get("group"), m.get("outcome")) for m in est.meta},
                      key=lambda t: (str(t[0]), str(t[1])))
        if ax is None:
            fig, ax = plt.subplots(figsize=kw.pop("figsize", (6.5, 4)))
        for i, (g, o) in enumerate(keys):
            idx = [j for j, m in enumerate(est.meta)
                   if m.get("group") == g and m.get("outcome") == o and "grid" in m]
            x = np.array([est.meta[j]["grid"] for j in idx])
            order = np.argsort(x)
            idx = [idx[j] for j in order]
            x = x[order]
            r = tab.iloc[idx]
            col = f"C{i}" if len(keys) > 1 else color
            lab = " | ".join(s for s in (o, g) if s is not None) or None
            ax.fill_between(x, r["lower"], r["upper"], color=col, alpha=0.2, lw=0)
            ax.plot(x, r["estimate"], color=col, lw=1.8, label=lab)
        if est.kind == "slopes":
            ax.axhline(0, color="grey", lw=0.7, ls=":")
        var = est.info.get("var", "")
        ax.set_xlabel(var)
        ax.set_ylabel(r"$E[Y(%s)]$" % var if est.kind == "levels"
                      else r"$E[dY/d(%s)]$" % var)
        ax.set_title(f"posterior {point}, {pct} {interval.upper()}", fontsize=9,
                     loc="left")
        if len(keys) > 1:
            ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        return ax.figure

    K = len(est.names)
    if ax is None:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (6.5, 0.45 * K + 1.2)))
    y = np.arange(K)[::-1]
    ax.hlines(y, tab["lower"], tab["upper"], color=color, lw=2)
    ax.plot(tab["estimate"], y, "o", color=color, ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels(est.names, fontsize=9)
    ax.set_title(f"posterior {point}, {pct} {interval.upper()}", fontsize=9,
                 loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    return ax.figure
