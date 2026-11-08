"""Checks of bayes_clarify against quantities computed by hand.

Run:  python -m pytest

The posterior draws are synthetic (normal around fixed values), so every
quantity has an exact value per draw that the package must reproduce.
"""

import numpy as np
import pandas as pd

import bayes_clarify as bc

RNG = np.random.default_rng(1)
N, S = 400, 600


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data():
    return pd.DataFrame({
        "x": RNG.normal(0.5, 1.2, N),
        "d": RNG.integers(0, 2, N),                         # binary, int
        "g": RNG.choice(["a", "b", "c"], N),                # categorical text
        "w": RNG.uniform(0.5, 2.0, N),
    })


DATA = make_data()
DRAWS = {"alpha": RNG.normal(-0.3, 0.1, S),
         "beta": RNG.normal([0.8, -0.6], 0.1, (S, 2)),       # x, d
         "gam": RNG.normal([0.4, -0.5], 0.1, (S, 2))}        # g = b, c


def eta(p, df):
    gb = (df["g"] == "b").to_numpy(float)
    gc = (df["g"] == "c").to_numpy(float)
    return (p["alpha"][:, None] + p["beta"][:, :1] * df["x"].to_numpy(float)
            + p["beta"][:, 1:2] * df["d"].to_numpy(float)
            + p["gam"][:, :1] * gb + p["gam"][:, 1:2] * gc)


def predict_logit(p, df):
    return expit(eta(p, df))


def post(**kw):
    return bc.Posterior(DRAWS, predict_logit, DATA, **kw)


def test_setx_matches_hand():
    est = bc.sim_setx(post(), {"x": [0.0, 1.0], "d": 1})
    # g held at its most frequent value
    gmode = DATA["g"].mode().iloc[0]
    for xv in (0.0, 1.0):
        row = pd.DataFrame({"x": [xv], "d": [1], "g": [gmode], "w": [1.0]})
        hand = predict_logit(DRAWS, row)[:, 0]
        assert np.allclose(est[f"E[Y|x={xv:.4g}]"], hand)
    assert list(est.info["profiles"]["g"]) == [gmode, gmode]


def test_setx_first_difference():
    est = bc.sim_setx(post(), {"x": 0.0}, x1={"x": 1.0})
    assert np.allclose(est["FD"], est["E[Y|x1]"] - est["E[Y|x]"])
    assert est.info["profiles"].loc["E[Y|x]", "d"] in (0, 1)


def test_ame_continuous_is_average_derivative():
    est = bc.sim_ame(post(), "x")
    p = predict_logit(DRAWS, DATA)
    hand = (p * (1 - p) * DRAWS["beta"][:, :1]).mean(1)
    assert np.allclose(est["E[dY/d(x)]"], hand, atol=1e-6)


def test_ame_binary_with_contrasts():
    est = bc.sim_ame(post(), "d", contrast="diff")
    d0, d1 = DATA.copy(), DATA.copy()
    d0["d"], d1["d"] = 0, 1
    e0, e1 = predict_logit(DRAWS, d0).mean(1), predict_logit(DRAWS, d1).mean(1)
    assert np.allclose(est["E[Y(0)]"], e0) and np.allclose(est["E[Y(1)]"], e1)
    assert np.allclose(est["Diff"], e1 - e0)
    orr = bc.sim_ame(post(), "d", contrast="odds_ratio")["OR"]
    assert np.allclose(orr, (e1 / (1 - e1)) / (e0 / (1 - e0)))


def test_ame_multilevel_reference_by_subset_weights():
    est = bc.sim_ame(post(), "g", contrast="ratio", reference="b",
                     by="d", subset="x > 0", weights="w")
    m = (DATA["x"] > 0).to_numpy()
    for dv in (0, 1):
        sel = m & (DATA["d"] == dv).to_numpy()
        w = np.where(sel, DATA["w"], 0.0)
        w = w / w.sum()
        e = {}
        for lv in ("a", "b", "c"):
            dd = DATA.copy()
            dd["g"] = lv
            e[lv] = predict_logit(DRAWS, dd) @ w
        assert np.allclose(est[f"E[Y(a)] | d={dv}"], e["a"])
        assert np.allclose(est[f"RR(c vs b) | d={dv}"], e["c"] / e["b"])
    assert "RR(b vs b) | d=0" not in est.names


def test_predictions_levels_and_slopes():
    at = [-1.0, 0.0, 2.0]
    adrf = bc.sim_predictions(post(), "x", at=at)
    amef = bc.sim_predictions(post(), "x", at=at, contrast="slopes")
    for v in at:
        dd = DATA.copy()
        dd["x"] = v
        p = predict_logit(DRAWS, dd)
        assert np.allclose(adrf[f"E[Y(x={v:.4g})]"], p.mean(1))
        assert np.allclose(amef[f"E[dY/d(x)|x={v:.4g}]"],
                           (p * (1 - p) * DRAWS["beta"][:, :1]).mean(1), atol=1e-6)
    grid = bc.sim_predictions(post(), "x", n=5)
    assert np.isclose(grid.meta[0]["grid"], DATA["x"].min())
    assert np.isclose(grid.meta[-1]["grid"], DATA["x"].max())


def test_multi_outcome_ordered_logit():
    cut = np.array([-0.5, 0.7])

    def predict_ord(p, df):
        e = eta(p, df)
        c1 = expit(cut[0] - e)
        c2 = expit(cut[1] - e)
        return np.stack([c1, c2 - c1, 1 - c2], axis=-1)        # (s, N, 3)

    po = bc.Posterior(DRAWS, predict_ord, DATA, outcome_names=["low", "mid", "high"])
    ame = bc.sim_ame(po, "x")
    tot = sum(ame[f"E[dY/d(x)] [{k}]"] for k in ("low", "mid", "high"))
    assert np.allclose(tot, 0, atol=1e-8)                       # probabilities sum to 1
    adrf = bc.sim_predictions(po, "x", n=4)
    assert len(adrf) == 12
    sx = bc.sim_setx(po, {"d": [0, 1]})
    assert np.allclose(sx["E[Y|d=0] [low]"] + sx["E[Y|d=0] [mid]"]
                       + sx["E[Y|d=0] [high]"], 1)


def test_chunking_and_thinning():
    a = bc.sim_ame(post(), "x").values
    b = bc.sim_ame(post(max_elems=1000), "x").values           # many chunks
    assert np.allclose(a, b)
    p = post(n_draws=100, seed=3)
    assert p.n_draws == 100 and bc.sim_ame(p, "x").n_draws == 100


def test_from_linear_equals_custom():
    data = DATA.assign(gb=(DATA["g"] == "b").astype(float),
                       gc=(DATA["g"] == "c").astype(float))
    draws = {"a": DRAWS["alpha"],
             "b": np.column_stack([DRAWS["beta"], DRAWS["gam"]])}
    pl = bc.Posterior.from_linear(draws, data, coef="b", intercept="a",
                                  columns=["x", "d", "gb", "gc"], link="logit")
    assert np.allclose(bc.sim_ame(pl, "x").values, bc.sim_ame(post(), "x").values)


def test_apply_params_transform_concat_summary():
    p = post()
    v = bc.sim_apply(p, lambda P: {"b_x": P["beta"][:, 0], "b_d": P["beta"][:, 1]})
    nv = bc.sim_apply(p, lambda P: P["beta"][0], vectorized=False, names=["b_x"])
    assert np.allclose(v["b_x"], nv["b_x"])
    sp = bc.sim_params(p, ["alpha", "beta"], labels={"beta": ["x", "d"]})
    assert sp.names == ["alpha", "beta[x]", "beta[d]"]
    est = bc.sim_ame(p, "d").transform(RR=lambda d: d["E[Y(1)]"] / d["E[Y(0)]"])
    assert np.allclose(est["RR"], est["E[Y(1)]"] / est["E[Y(0)]"])
    both = bc.Draws.concat(est, sp)
    tab = both.summary(interval="hdi")
    assert list(tab.columns) == ["estimate", "sd", "lower", "upper", "Pr(>0)", "Pr(<0)"]
    assert (tab["lower"] <= tab["estimate"]).all() and (tab["estimate"] <= tab["upper"]).all()


def test_intervals():
    z = np.random.default_rng(0).normal(size=200_000)
    lo, hi = bc.eti_bounds(z, 0.95)
    assert abs(lo + 1.96) < 0.02 and abs(hi - 1.96) < 0.02
    lo2, hi2 = bc.hdi_bounds(z, 0.95)
    assert abs(lo2 + 1.96) < 0.02 and abs(hi2 - 1.96) < 0.02
    e = np.random.default_rng(0).exponential(size=200_000)
    lo, hi = bc.hdi_bounds(e, 0.9)
    assert abs(lo) < 0.01 and abs(hi - np.log(10)) < 0.02     # HDI of Exp(1)
    a, b = bc.eti_bounds(e, 0.9)
    assert hi - lo < b - a


def test_plots_run():
    import matplotlib
    matplotlib.use("Agg")
    bc.sim_predictions(post(), "x", n=7, by="d").plot()
    bc.sim_ame(post(), "g", contrast="diff").plot()


def test_clarify_names_still_work():
    a = bc.sim_predictions(post(), "x", at=[0.0, 1.0], contrast="slopes").values
    b = bc.sim_adrf(post(), "x", at=[0.0, 1.0], contrast="amef").values
    assert np.allclose(a, b)
    assert bc.sim_adrf(post(), "x", at=[0.0]).kind == "levels"


def test_to_latex():
    est = bc.sim_ame(post(), "d", contrast="diff")
    tex = est.to_latex(level=0.94, interval="hdi", point="mean", caption="c", label="l",
                       labels={"Diff": r"$\Delta$"})
    assert r"\begin{tabular}{lrrrrr}" in tex and r"94\% HDI" in tex
    assert r"$\Delta$ &" in tex and "E[Y(0)]" in tex
    assert tex.count(r"\\") >= 5
